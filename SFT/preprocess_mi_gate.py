"""Offline MI Gate preprocessing for HatefulMemes.

Pipeline (run once, before SFT):

    1. Re-extract SigLIP2 features for every sample in ``data.json`` in id
       order. Reuses the legacy PCA fit (mean + components) from the existing
       feature file so the new features land in the same 512-dim space the
       trained estimators were trained on.
    2. Build a deterministic stratified train / val / test split over the
       samples that have a usable image, and write ``split`` back into
       ``data.json``.
    3. Score every train sample's R / U_V / U_T / S using the trained
       estimators in ``Estimator/saved_estimators/`` and write per-sample
       scores back to ``data.json`` (``mi_r``, ``mi_u1``, ``mi_u2``,
       ``mi_s``).
    4. Apply the MI Gate on the train split. Treat samples whose dominant
       interaction term is U_V (image-uniqueness) as the *valid set*; pick
       ``floor(tau * |valid|)`` of them, ranked by U_V score. ``tau`` is
       therefore the fraction of valid (U1-dominant) samples that receive
       the caption augmentation, *not* the fraction of all train samples.
       For selected samples set ``mi_gate_text = original_text + " " + caption``;
       otherwise keep ``mi_gate_text = original_text``. Mark each sample with
       ``mi_gate_in_subset`` (bool).

Run from repo root:

    python SFT/preprocess_mi_gate.py --tau 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTIMATOR_DIR = os.path.join(REPO_ROOT, "Estimator")
for p in (REPO_ROOT, ESTIMATOR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from Estimator.entropy_estimator import MargKernel  # noqa: E402
from Estimator.mi_estimator import LSMI_estimation  # noqa: E402
from Estimator.utils import cls_network, feature_dataset, setup_seed  # noqa: E402
from Features.utils_features import (  # noqa: E402
    build_siglip2,
    extract_image_features_siglip2,
    extract_text_features_siglip2,
    stratified_split_indices,
)


# ---------------------------------------------------------------------------
# Estimator loading
# ---------------------------------------------------------------------------
def load_trained_estimators(ckpt_path: str, device: str):
    """Re-build estimator architectures from the saved meta and load weights."""
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    in1, in2 = int(meta["input_size_1"]), int(meta["input_size_2"])
    embed = int(meta["embed_size"])
    n_classes = int(meta["n_classes"])

    discriminator = [
        cls_network(in1, embed, n_classes).to(device),
        cls_network(in2, embed, n_classes).to(device),
        cls_network(in1 + in2, embed, n_classes).to(device),
    ]
    for model, sd in zip(discriminator, ckpt["discriminator_state_dicts"]):
        model.load_state_dict(sd)
        model.eval()

    entropy = [MargKernel(dim=in1).to(device), MargKernel(dim=in2).to(device)]
    for model, sd in zip(entropy, ckpt["entropy_state_dicts"]):
        model.load_state_dict(sd)
        model.eval()

    return discriminator, entropy, meta


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_id_aligned_features(
    data: List[dict],
    image_dir: str,
    siglip_model_id: str,
    device: str,
    image_batch_size: int = 32,
    text_batch_size: int = 128,
    text_field: str = "original_text",
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Extract SigLIP2 image and text features for samples with a usable image.

    Returns
    -------
    image_features: [M, D]  raw SigLIP2 image features (M = #valid samples)
    text_features:  [M, D]  raw SigLIP2 text features
    valid_mask:     [N]     boolean mask aligned to ``data`` order; M = mask.sum()
    """
    print(f"[features] loading {siglip_model_id} on {device}")
    model, tokenizer, image_processor = build_siglip2(siglip_model_id, device=device)

    images: List[Image.Image] = []
    texts: List[str] = []
    valid = np.zeros(len(data), dtype=bool)
    missing = 0

    for i, sample in enumerate(data):
        path = os.path.join(image_dir, f"{sample['id']}.png")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[features] skip id={sample['id']}: {exc}")
            missing += 1
            continue
        images.append(img)
        texts.append(str(sample.get(text_field, "")))
        valid[i] = True

    print(f"[features] valid samples: {valid.sum()} / {len(data)} (missing={missing})")
    print("[features] extracting image features")
    img_feats = extract_image_features_siglip2(
        images, model, image_processor, batch_size=image_batch_size, device=device
    )
    print("[features] extracting text features")
    txt_feats = extract_text_features_siglip2(
        texts, model, tokenizer, batch_size=text_batch_size, device=device
    )

    del model, images, texts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return img_feats, txt_feats, valid


def apply_pca(features: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Project features into the legacy PCA space: (X - mean) @ components."""
    return (features.float() - mean.unsqueeze(0)) @ components


# ---------------------------------------------------------------------------
# Splits + scoring
# ---------------------------------------------------------------------------
def assign_splits(
    targets: np.ndarray,
    valid_mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> np.ndarray:
    """Stratified train/val/test split over valid samples; invalid → 'excluded'."""
    valid_idx = np.where(valid_mask)[0]
    valid_targets = targets[valid_idx]
    rel_train, rel_val, rel_test = stratified_split_indices(valid_targets, train_ratio, val_ratio, seed)

    splits = np.full(len(targets), "excluded", dtype=object)
    splits[valid_idx[rel_train]] = "train"
    splits[valid_idx[rel_val]] = "val"
    splits[valid_idx[rel_test]] = "test"
    return splits


def score_per_sample(
    discriminator,
    entropy_estimator,
    img_features: torch.Tensor,
    txt_features: torch.Tensor,
    targets: torch.Tensor,
    device: str,
    n_classes: int,
    batch_size: int = 256,
    class_priors: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """Run LSMI estimation in id order; return per-sample R/U1/U2/S as numpy."""

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.device = device
    cfg.n_classes = n_classes
    cfg.class_priors = class_priors
    cfg.input_size_1 = img_features.shape[1]
    cfg.input_size_2 = txt_features.shape[1]

    ds = feature_dataset(img_features, txt_features, targets)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    R, U1, U2, S, per_sample = LSMI_estimation(
        loader, discriminator, entropy_estimator, cfg=cfg, return_per_sample=True
    )
    return {k: v.numpy() for k, v in per_sample.items()}


# ---------------------------------------------------------------------------
# MI Gate
# ---------------------------------------------------------------------------
def summarize_interactions(
    per_sample: Dict[str, np.ndarray],
    label: str,
    indices: Optional[np.ndarray] = None,
) -> None:
    """Print expected R/U1/U2/S and dominance proportions over a (sub)set."""
    keys = ("r", "u1", "u2", "s")
    pretty = ("R", "U1", "U2", "S")
    if indices is None:
        n = len(per_sample[keys[0]])
        slc = slice(None)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        n = len(indices)
        slc = indices
    if n == 0:
        print(f"[summary][{label}] empty subset")
        return
    means = {k: float(per_sample[k][slc].mean()) for k in keys}
    stack = np.stack([per_sample[k][slc] for k in keys], axis=1)
    argmax = stack.argmax(axis=1)
    counts = np.bincount(argmax, minlength=4)
    print(f"[summary][{label}] N={n}")
    print(f"  E[R]={means['r']:+.4f}  E[U1]={means['u1']:+.4f}  "
          f"E[U2]={means['u2']:+.4f}  E[S]={means['s']:+.4f}")
    print("  dominance proportions:")
    for name, c in zip(pretty, counts):
        print(f"    {name}-dominant: {int(c)}/{n} ({100.0 * c / n:5.1f}%)")


def mi_gate_select(
    per_sample: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Return absolute indices (into the full data list) of samples to caption.

    A sample is *valid* when ``argmax(r, u1, u2, s) == u1``. The gate keeps
    ``floor(tau * |valid|)`` of the valid samples, ranked by ``u1`` (descending).
    Note: ``tau`` is the fraction of *valid* (U1-dominant) train samples that
    receive the caption, not the fraction of all train samples.
    """
    r = per_sample["r"][train_indices]
    u1 = per_sample["u1"][train_indices]
    u2 = per_sample["u2"][train_indices]
    s = per_sample["s"][train_indices]
    stack = np.stack([r, u1, u2, s], axis=1)
    valid_local = np.where(stack.argmax(axis=1) == 1)[0]
    print(f"[mi_gate] u1-dominant samples: {len(valid_local)} / {len(train_indices)}")

    n_target = int(np.floor(tau * len(valid_local)))
    if n_target <= 0:
        print(f"[mi_gate] selected for captioning: 0 (target = floor({tau}*{len(valid_local)}) = 0)")
        return np.array([], dtype=np.int64)
    ranked_local = valid_local[np.argsort(-u1[valid_local])]
    chosen_local = ranked_local[:n_target]
    chosen_abs = train_indices[chosen_local]
    print(f"[mi_gate] selected for captioning: {len(chosen_abs)} (target = floor({tau}*{len(valid_local)}) = {n_target})")
    return chosen_abs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MI Gate preprocessing")
    parser.add_argument("--data_path", default=os.path.join(REPO_ROOT, "data", "data.json"))
    parser.add_argument("--image_dir", default=os.path.join(REPO_ROOT, "data", "images"))
    parser.add_argument("--legacy_features", default=os.path.join(REPO_ROOT, "Features/hateful_memes_features_siglip2_pca512.pt"),
                        help="Used for PCA mean/components so new features match the estimator's training space.")
    parser.add_argument("--features_out", default=os.path.join(REPO_ROOT, "Features/hateful_memes_features_id_aligned.pt"))
    parser.add_argument("--estimator_ckpt", default=os.path.join(REPO_ROOT, "Estimator/saved_estimators/mi_estimators_latest_state_dict.pt"))
    parser.add_argument("--siglip_model_id", default="google/siglip2-giant-opt-patch16-384")
    parser.add_argument("--text_field", default="original_text", help="data.json field to feed to the text encoder.")
    parser.add_argument("--caption_field", default="generated_caption_smolvlm")
    parser.add_argument("--label_map", default="No.:0,Yes.:1", help="Comma list mapping correct_answer → int.")
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--skip_feature_extraction", action="store_true",
                        help="Reuse --features_out if it already exists.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, only operate on the first N rows of data.json (testing).")
    parser.add_argument("--data_path_out", default=None,
                        help="Optional separate write path for the augmented data.json (defaults to --data_path).")
    args = parser.parse_args()

    setup_seed(args.seed)
    label_map = dict(part.split(":") for part in args.label_map.split(","))
    label_map = {k: int(v) for k, v in label_map.items()}
    print(f"[setup] label_map = {label_map}")

    # ---------------- Load data + targets ----------------
    with open(args.data_path) as fh:
        data = json.load(fh)
    if args.max_samples is not None:
        data = data[: int(args.max_samples)]
        print(f"[setup] truncated data to {len(data)} rows for testing")
    targets_list = []
    for s in data:
        ans = s["correct_answer"]
        if ans not in label_map:
            raise ValueError(f"Unknown correct_answer={ans!r} for id={s['id']}.")
        targets_list.append(label_map[ans])
    targets_np = np.asarray(targets_list, dtype=np.int64)

    # ---------------- Feature extraction ----------------
    if args.skip_feature_extraction and os.path.exists(args.features_out):
        print(f"[features] reusing {args.features_out}")
        feat_blob = torch.load(args.features_out, map_location="cpu")
        img_pca = feat_blob["modal_1_features"]
        txt_pca = feat_blob["modal_2_features"]
        valid = feat_blob["valid_mask"].numpy().astype(bool)
    else:
        img_raw, txt_raw, valid = extract_id_aligned_features(
            data, args.image_dir, args.siglip_model_id, args.device,
            image_batch_size=args.image_batch_size,
            text_batch_size=args.text_batch_size,
            text_field=args.text_field,
        )
        # Apply legacy PCA so features match the estimator training space.
        legacy = torch.load(args.legacy_features, map_location="cpu")
        img_pca = apply_pca(img_raw, legacy["pca_img_mean"], legacy["pca_img_components"])
        txt_pca = apply_pca(txt_raw, legacy["pca_txt_mean"], legacy["pca_txt_components"])
        os.makedirs(os.path.dirname(args.features_out), exist_ok=True)
        torch.save(
            {
                "modal_1_features": img_pca,
                "modal_2_features": txt_pca,
                "ids": torch.tensor([s["id"] for s, v in zip(data, valid) if v], dtype=torch.long),
                "targets": torch.tensor([t for t, v in zip(targets_list, valid) if v], dtype=torch.long),
                "valid_mask": torch.tensor(valid),
            },
            args.features_out,
        )
        print(f"[features] saved id-aligned features → {args.features_out}")

    # img_pca/txt_pca have shape [M, 512]; we work in the M-row valid space.
    valid_idx_abs = np.where(valid)[0]              # absolute index in data[]
    if len(valid_idx_abs) == 0:
        raise RuntimeError(
            f"No samples with a usable image found under {args.image_dir}. "
            "Run Features/download_images.py or pass --image_dir to point at the right directory."
        )
    targets_valid = targets_np[valid_idx_abs]
    targets_t = torch.tensor(targets_valid, dtype=torch.long)

    # ---------------- Splits ----------------
    splits = assign_splits(targets_np, valid, args.train_ratio, args.val_ratio, args.seed)
    print(f"[split] train={(splits=='train').sum()} val={(splits=='val').sum()} "
          f"test={(splits=='test').sum()} excluded={(splits=='excluded').sum()}")

    # Indices into the *valid* tensor (used for scoring) for each split:
    valid_to_local = {abs_i: local_i for local_i, abs_i in enumerate(valid_idx_abs)}
    def _local(name):
        return np.array(
            [valid_to_local[i] for i in np.where(splits == name)[0]],
            dtype=np.int64,
        )
    train_local = _local("train")
    val_local = _local("val")
    test_local = _local("test")

    # ---------------- Scoring ----------------
    discriminator, entropy_estimator, meta = load_trained_estimators(args.estimator_ckpt, args.device)
    if img_pca.shape[1] != int(meta["input_size_1"]) or txt_pca.shape[1] != int(meta["input_size_2"]):
        raise RuntimeError(
            f"Feature dim mismatch: features {img_pca.shape[1]}/{txt_pca.shape[1]} vs "
            f"estimator {meta['input_size_1']}/{meta['input_size_2']}"
        )

    per_sample = score_per_sample(
        discriminator, entropy_estimator,
        img_pca.to(args.device), txt_pca.to(args.device),
        targets_t.to(args.device),
        device=args.device, n_classes=int(meta["n_classes"]),
        batch_size=args.score_batch_size,
    )
    # per_sample arrays are length M (valid samples), aligned with valid_idx_abs.

    # ---------------- Interaction summaries ----------------
    summarize_interactions(per_sample, "all valid")
    summarize_interactions(per_sample, "train", train_local)
    summarize_interactions(per_sample, "val", val_local)
    summarize_interactions(per_sample, "test", test_local)

    # ---------------- MI Gate ----------------
    chosen_local = mi_gate_select(per_sample, train_local, args.tau)
    chosen_abs = set(int(valid_idx_abs[i]) for i in chosen_local)

    # ---------------- Write back to data.json ----------------
    nan = float("nan")
    for i, sample in enumerate(data):
        sample["split"] = str(splits[i])
        if valid[i]:
            local = valid_to_local[i]
            sample["mi_r"] = float(per_sample["r"][local])
            sample["mi_u1"] = float(per_sample["u1"][local])
            sample["mi_u2"] = float(per_sample["u2"][local])
            sample["mi_s"] = float(per_sample["s"][local])
        else:
            sample["mi_r"] = nan
            sample["mi_u1"] = nan
            sample["mi_u2"] = nan
            sample["mi_s"] = nan
        in_subset = i in chosen_abs
        sample["mi_gate_in_subset"] = bool(in_subset)
        original = sample.get(args.text_field, "")
        if in_subset:
            caption = (sample.get(args.caption_field) or "").strip()
            sample["mi_gate_text"] = (original + " " + caption).strip() if caption else original
        else:
            sample["mi_gate_text"] = original

    out_path = args.data_path_out or args.data_path
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[write] updated {out_path}")
    print(f"[done] tau={args.tau}: captioned {len(chosen_abs)} train samples "
          f"out of {(splits=='train').sum()}.")


if __name__ == "__main__":
    main()
