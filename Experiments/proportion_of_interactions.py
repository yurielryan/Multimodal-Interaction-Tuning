"""Proportion of dominant interactions across the dataset.

Computes per-sample R / U1 / U2 / S using the trained estimators and reports
the proportion of samples whose *dominant* interaction term (argmax over the
four point-wise terms) is each of R, U1, U2, S — both for the dataset as a
whole and per train/val/test split when those fields are present in
``data.json``.

Two passes are produced:

1. **Part 1 — Frozen estimator.** Scores every sample with the existing
   pre-trained estimator at
   ``Estimator/saved_estimators/mi_estimators_latest_state_dict.pt`` in the
   legacy PCA space. The text fed to the SigLIP text encoder comes from
   ``--text_field`` (default ``original_text``).

2. **Part 2 — All-captioned retraining.** Re-encodes the text with the
   caption appended for *every* sample (``original_text + " " + caption``),
   fits a fresh PCA on a stratified train split, saves the features as a
   standard .pt at
   ``Features/features_all_augmented_siglip2_pca512.pt`` (same schema as
   ``Features/hateful_memes_features_siglip2_pca512.pt``), retrains the
   discriminator + entropy estimators from scratch, saves them at
   ``Estimator/saved_estimators/mi_estimators_all_augmented_state_dict.pt``,
   and rescores the full dataset in the new PCA space. Pass
   ``--skip_retrain`` to disable Part 2.

τ is *not* a variable in this experiment. τ is the MI Gate's threshold (the
fraction of U1-dominant samples that receive a caption during SFT) and lives
in ``SFT/preprocess_mi_gate.py``; here we only characterize the raw dominance
distribution of two specific dataset views (no caption / all captions).

Run from repo root:

    python Experiments/proportion_of_interactions.py
    python Experiments/proportion_of_interactions.py --skip_retrain      # Part 1 only
    python Experiments/proportion_of_interactions.py --max_samples 200    # smoke run
    python Experiments/proportion_of_interactions.py --plot_only          # re-render figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTIMATOR_DIR = os.path.join(REPO_ROOT, "Estimator")
for p in (REPO_ROOT, ESTIMATOR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Features.utils_features import (  # noqa: E402
    build_siglip2,
    extract_image_features_siglip2,
    extract_text_features_siglip2,
    pca_fit_on_train_and_transform,
    stratified_split_indices,
)
from SFT.preprocess_mi_gate import (  # noqa: E402
    apply_pca,
    load_trained_estimators,
    score_per_sample,
)
from Estimator.utils import feature_dataset, setup_seed  # noqa: E402
from Estimator.mi_estimator import (  # noqa: E402
    obtain_discriminator,
    obtain_entropy_estimator,
    save_trained_estimators,
)

DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "data", "data.json")
DEFAULT_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "images")
DEFAULT_LEGACY_FEATURES = os.path.join(
    REPO_ROOT, "Features/hateful_memes_features_siglip2_pca512.pt"
)
DEFAULT_ESTIMATOR_CKPT = os.path.join(
    REPO_ROOT, "Estimator/saved_estimators/mi_estimators_latest_state_dict.pt"
)
DEFAULT_ESTIMATOR_OUT_DIR = os.path.join(REPO_ROOT, "Estimator/saved_estimators")
DEFAULT_FEATURES_OUT = os.path.join(
    REPO_ROOT, "Features/features_all_augmented_siglip2_pca512.pt"
)
DEFAULT_SIGLIP = "google/siglip2-giant-opt-patch16-384"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "Experiments/outputs")

LABEL_MAP = {"No.": 0, "Yes.": 1}
TERM_NAMES = ("R", "U1", "U2", "S")
PALETTE = {
    "R":  "#4C78A8",
    "U1": "#F58518",
    "U2": "#54A24B",
    "S":  "#E45756",
}


# ---------------------------------------------------------------------------
# Subset collection
# ---------------------------------------------------------------------------
def collect_subset(
    data: List[dict],
    image_dir: str,
    text_field: str,
    caption_field: str = "generated_caption_smolvlm",
    max_samples: int = None,
) -> Tuple[List[Image.Image], List[str], List[str], List[int], List[int]]:
    """Return (images, baseline_texts, augmented_texts, targets, ids).

    - baseline_texts come from ``text_field`` (default ``original_text``).
    - augmented_texts are ``original_text + " " + caption`` for every row that
      has a non-empty caption (otherwise the baseline string is reused).
    """
    images: List[Image.Image] = []
    baseline: List[str] = []
    augmented: List[str] = []
    targets: List[int] = []
    ids: List[int] = []
    for s in data:
        img_path = os.path.join(image_dir, f"{s['id']}.png")
        if not os.path.exists(img_path):
            continue
        ans = s.get("correct_answer")
        if ans not in LABEL_MAP:
            continue
        original = str(s.get("original_text", "") or "")
        caption = str(s.get(caption_field) or "").strip()
        base_text = str(s.get(text_field, "") or original)
        aug_text = (original + " " + caption).strip() if caption else original
        images.append(Image.open(img_path).convert("RGB"))
        baseline.append(base_text)
        augmented.append(aug_text)
        targets.append(LABEL_MAP[ans])
        ids.append(int(s["id"]))
        if max_samples is not None and len(images) >= max_samples:
            break
    return images, baseline, augmented, targets, ids


# ---------------------------------------------------------------------------
# MI helpers
# ---------------------------------------------------------------------------
def score(img_pca, txt_pca, targets_t, discriminator, entropy_estimator,
          n_classes, device, score_batch_size):
    return score_per_sample(
        discriminator, entropy_estimator,
        img_pca.to(device), txt_pca.to(device), targets_t,
        device=device, n_classes=n_classes, batch_size=score_batch_size,
    )


def dominance_proportions(per_sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return {term: proportion} for the four interaction terms."""
    keys = ("r", "u1", "u2", "s")
    n = len(per_sample[keys[0]])
    if n == 0:
        return {name: float("nan") for name in TERM_NAMES}
    stack = np.stack([per_sample[k] for k in keys], axis=1)
    counts = np.bincount(stack.argmax(axis=1), minlength=4)
    return {name: float(c) / n for name, c in zip(TERM_NAMES, counts)}


def expectations(per_sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    keys = ("r", "u1", "u2", "s")
    n = len(per_sample[keys[0]])
    return {f"E[{name}]": float(per_sample[k].mean()) if n else float("nan")
            for k, name in zip(keys, TERM_NAMES)}


def per_split_breakdown(
    per_sample: Dict[str, np.ndarray],
    ids: List[int],
    data: List[dict],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, int]]:
    """Slice per-sample MI tensors by data.json's `split` field if present."""
    id_to_local = {i: k for k, i in enumerate(ids)}
    split_to_local: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for s in data:
        split = s.get("split")
        sid = int(s["id"])
        if split in split_to_local and sid in id_to_local:
            split_to_local[split].append(id_to_local[sid])

    per_split: Dict[str, Dict[str, float]] = {}
    per_split_means: Dict[str, Dict[str, float]] = {}
    n_per_split: Dict[str, int] = {}
    for split, locs in split_to_local.items():
        if not locs:
            continue
        locs_np = np.asarray(locs, dtype=np.int64)
        sub = {k: per_sample[k][locs_np] for k in ("r", "u1", "u2", "s")}
        per_split[split] = dominance_proportions(sub)
        per_split_means[split] = expectations(sub)
        n_per_split[split] = int(locs_np.size)
    return per_split, per_split_means, n_per_split


def print_summary(label: str, n: int, overall: Dict[str, float],
                  means: Dict[str, float],
                  per_split: Dict[str, Dict[str, float]] = None,
                  per_split_means: Dict[str, Dict[str, float]] = None,
                  n_per_split: Dict[str, int] = None) -> None:
    print(f"\n[{label}] N={n}")
    print(f"  E[R]={means['E[R]']:+.4f}  E[U1]={means['E[U1]']:+.4f}  "
          f"E[U2]={means['E[U2]']:+.4f}  E[S]={means['E[S]']:+.4f}")
    print("  dominance proportions:")
    for name in TERM_NAMES:
        print(f"    {name}-dominant: {100.0 * overall[name]:5.1f}%")
    if not per_split:
        return
    for split in ("train", "val", "test"):
        if split not in per_split:
            continue
        print(f"  [{label} | split={split}] N={n_per_split[split]}")
        m = per_split_means[split]
        print(f"    E[R]={m['E[R]']:+.4f}  E[U1]={m['E[U1]']:+.4f}  "
              f"E[U2]={m['E[U2]']:+.4f}  E[S]={m['E[S]']:+.4f}")
        for name in TERM_NAMES:
            print(f"    {name}-dominant: {100.0 * per_split[split][name]:5.1f}%")


# ---------------------------------------------------------------------------
# Part 2: build features, retrain, score
# ---------------------------------------------------------------------------
class _RetrainCfg:
    """Lightweight cfg for Estimator/mi_estimator.py's obtain_*/save_* APIs."""
    pass


def make_retrain_cfg(
    device: str,
    n_classes: int,
    embed_size: int = 512,
    input_size_1: int = 512,
    input_size_2: int = 512,
    batch_size: int = 64,
    num_epochs_disc: int = 30,
    num_epochs_ent: int = 80,
    es_patience: int = 10,
    es_min_delta: float = 1e-4,
):
    cfg = _RetrainCfg()
    cfg.device = device
    cfg.n_classes = n_classes
    cfg.input_size_1 = input_size_1
    cfg.input_size_2 = input_size_2
    cfg.embed_size = embed_size
    cfg.batch_size = batch_size
    cfg.num_workers = 0
    cfg.num_epochs_discriminator = num_epochs_disc
    cfg.num_epochs_entropy_estimator = num_epochs_ent
    cfg.early_stopping = True
    cfg.es_patience = es_patience
    cfg.es_min_delta = es_min_delta
    cfg.save_full_models = False
    return cfg


def build_and_save_all_augmented_features(
    img_raw: torch.Tensor,
    txt_augmented_raw: torch.Tensor,
    targets_np: np.ndarray,
    ids: List[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    save_path: str,
    target_dim: int = 512,
):
    """Fit fresh PCA on the train split (img + all-augmented text), then save
    a standard-schema feature .pt covering every row."""
    img_pca_d = pca_fit_on_train_and_transform(img_raw, train_idx, target_dim)
    txt_pca_d = pca_fit_on_train_and_transform(txt_augmented_raw, train_idx, target_dim)

    img_red = img_pca_d["reduced"]
    txt_red = txt_pca_d["reduced"]
    targets_t = torch.tensor(targets_np, dtype=torch.long)

    feat = {
        "train_modal_1_features": img_red[train_idx],
        "train_modal_2_features": txt_red[train_idx],
        "train_targets": targets_t[train_idx],
        "val_modal_1_features": img_red[val_idx],
        "val_modal_2_features": txt_red[val_idx],
        "val_targets": targets_t[val_idx],
        "test_modal_1_features": img_red[test_idx],
        "test_modal_2_features": txt_red[test_idx],
        "test_targets": targets_t[test_idx],
        "pca_img_mean": img_pca_d["mean"],
        "pca_img_components": img_pca_d["components"],
        "pca_img_explained_variance": img_pca_d["explained_variance"],
        "pca_img_explained_variance_ratio": img_pca_d["explained_variance_ratio"],
        "pca_txt_mean": txt_pca_d["mean"],
        "pca_txt_components": txt_pca_d["components"],
        "pca_txt_explained_variance": txt_pca_d["explained_variance"],
        "pca_txt_explained_variance_ratio": txt_pca_d["explained_variance_ratio"],
        "pca_target_dim": target_dim,
        "pca_img_effective_dim": img_pca_d["effective_dim"],
        "pca_txt_effective_dim": txt_pca_d["effective_dim"],
        "pca_enabled": True,
        # Provenance
        "experiment": "all_augmented",
        "n_total": int(img_raw.shape[0]),
        "ids": np.asarray(ids, dtype=np.int64),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(feat, save_path)
    return feat, img_red, txt_red


def retrain_from_features(feat: dict, cfg: _RetrainCfg):
    train_ds = feature_dataset(
        feat["train_modal_1_features"],
        feat["train_modal_2_features"],
        feat["train_targets"],
    )
    val_ds = feature_dataset(
        feat["val_modal_1_features"],
        feat["val_modal_2_features"],
        feat["val_targets"],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    discriminator = obtain_discriminator(cfg, train_loader, val_loader)
    entropy_estimator = obtain_entropy_estimator(cfg, train_loader, val_loader)
    return discriminator, entropy_estimator


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _draw_donut(ax, sizes, labels, colors, center_text, wedge_label_min=2.0):
    """Render a donut on ``ax``."""
    total = float(sum(sizes)) if sum(sizes) else 1.0
    pcts = [100.0 * s / total for s in sizes]
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
    )
    for w, lbl, p in zip(wedges, labels, pcts):
        if p < wedge_label_min:
            continue
        ang = (w.theta1 + w.theta2) / 2.0
        rad = 0.79
        x = rad * np.cos(np.deg2rad(ang))
        y = rad * np.sin(np.deg2rad(ang))
        ax.text(
            x, y, f"{lbl}\n{p:.1f}%",
            ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold",
        )
    ax.text(0, 0, center_text, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")


def plot_proportions(
    overall: Dict[str, float],
    per_split: Dict[str, Dict[str, float]],
    n_overall: int,
    n_per_split: Dict[str, int],
    out_path: str,
    title: str = "Dominant interaction proportions",
) -> None:
    """One big donut + optional per-split mini-donuts side-by-side."""
    has_splits = bool(per_split)
    split_order = [s for s in ("train", "val", "test") if s in per_split] if has_splits else []
    n_splits = len(split_order)

    if has_splits and n_splits > 0:
        fig = plt.figure(figsize=(4.0 + 2.6 * n_splits, 5.4))
        gs = fig.add_gridspec(1, 1 + n_splits, width_ratios=[1.4] + [1.0] * n_splits, wspace=0.05)
        ax_g = fig.add_subplot(gs[0, 0])
        split_axes = [fig.add_subplot(gs[0, 1 + i]) for i in range(n_splits)]
    else:
        fig, ax_g = plt.subplots(figsize=(6.4, 5.6))
        split_axes = []

    sizes = [overall[name] for name in TERM_NAMES]
    colors = [PALETTE[n] for n in TERM_NAMES]
    _draw_donut(ax_g, sizes, TERM_NAMES, colors, center_text=f"N={n_overall}")
    ax_g.set_title(title, fontsize=12, pad=10)

    for ax_s, split in zip(split_axes, split_order):
        sizes = [per_split[split][name] for name in TERM_NAMES]
        _draw_donut(
            ax_s, sizes, TERM_NAMES, colors,
            center_text=f"{split}\nN={n_per_split[split]}",
        )
        ax_s.set_title(split, fontsize=11, pad=6)

    legend_handles = [
        plt.matplotlib.patches.Patch(facecolor=PALETTE[name], edgecolor="white", label=f"{name}-dominant")
        for name in TERM_NAMES
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--legacy_features", default=DEFAULT_LEGACY_FEATURES)
    parser.add_argument("--estimator_ckpt", default=DEFAULT_ESTIMATOR_CKPT)
    parser.add_argument("--siglip_model_id", default=DEFAULT_SIGLIP)
    parser.add_argument("--text_field", default="original_text",
                        help="Which data.json field to feed the SigLIP text encoder (Part 1).")
    parser.add_argument("--caption_field", default="generated_caption_smolvlm",
                        help="data.json field used as the caption (Part 2 augmentation).")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap subset size for fast smoke runs.")
    parser.add_argument("--no_per_split", dest="per_split", action="store_false",
                        help="Skip per-split breakdown even if 'split' is in data.json.")
    parser.set_defaults(per_split=True)
    # Part 2 controls
    parser.add_argument("--skip_retrain", action="store_true",
                        help="Skip Part 2 (re-extract + retrain on all-augmented features).")
    parser.add_argument("--features_out", default=DEFAULT_FEATURES_OUT,
                        help="Where Part 2 saves the all-augmented feature .pt.")
    parser.add_argument("--estimator_out_dir", default=DEFAULT_ESTIMATOR_OUT_DIR,
                        help="Where Part 2 saves the retrained estimator state_dict.")
    parser.add_argument("--estimator_tag", default="all_augmented",
                        help="Tag suffix for the retrained estimator filename.")
    parser.add_argument("--retrain_seed", type=int, default=42)
    parser.add_argument("--retrain_epochs_disc", type=int, default=30)
    parser.add_argument("--retrain_epochs_ent", type=int, default=80)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip scoring and re-render figures from the existing JSON in --out_dir.")
    args = parser.parse_args()

    fig_path_p1 = os.path.join(args.out_dir, "proportion_of_interactions.png")
    fig_path_p2 = os.path.join(args.out_dir, "proportion_of_interactions_retrained.png")
    json_path = os.path.join(args.out_dir, "proportion_of_interactions.json")

    if args.plot_only:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"--plot_only requires {json_path}; run the full sweep first.")
        with open(json_path) as fh:
            rec = json.load(fh)
        # Backwards-compat: old top-level layout had overall_proportions at the root.
        frozen = rec.get("frozen") or {
            "overall_proportions": rec.get("overall_proportions"),
            "per_split_proportions": rec.get("per_split_proportions", {}),
            "n_per_split": rec.get("n_per_split", {}),
        }
        n_overall = int(rec.get("n", 0))
        plot_proportions(
            frozen["overall_proportions"],
            frozen.get("per_split_proportions", {}),
            n_overall,
            frozen.get("n_per_split", {}),
            fig_path_p1,
            title="Dominant interaction proportions (frozen estimator, baseline text)",
        )
        print(f"[save] figure → {fig_path_p1} (+ .pdf)")
        retrained = rec.get("retrained_all_augmented")
        if retrained:
            plot_proportions(
                retrained["overall_proportions"],
                retrained.get("per_split_proportions", {}),
                n_overall,
                retrained.get("n_per_split", {}),
                fig_path_p2,
                title="Dominant interaction proportions (retrained on all-augmented features)",
            )
            print(f"[save] figure → {fig_path_p2} (+ .pdf)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.skip_retrain:
        os.makedirs(os.path.dirname(args.features_out), exist_ok=True)
        os.makedirs(args.estimator_out_dir, exist_ok=True)

    print(f"[setup] loading data from {args.data_path}", flush=True)
    with open(args.data_path) as fh:
        data = json.load(fh)
    print(f"[setup] {len(data)} rows in data.json")

    print(f"[setup] loading SigLIP2 ({args.siglip_model_id}) on {args.device}", flush=True)
    siglip_model, siglip_tok, siglip_ip = build_siglip2(args.siglip_model_id, device=args.device)
    legacy_pca = torch.load(args.legacy_features, map_location="cpu")
    discriminator, entropy_estimator, meta = load_trained_estimators(
        args.estimator_ckpt, args.device,
    )
    n_classes = int(meta["n_classes"])

    # --------- Collect ---------
    print(f"[run] collecting samples (text_field={args.text_field}, caption_field={args.caption_field})", flush=True)
    images, baseline_texts, augmented_texts, targets, ids = collect_subset(
        data, args.image_dir, args.text_field, args.caption_field,
        max_samples=args.max_samples,
    )
    n = len(images)
    if n == 0:
        raise RuntimeError(
            f"No usable samples found under {args.image_dir}. "
            "Run Features/download_images.py or pass --image_dir to point at the right directory."
        )
    n_with_caption = sum(1 for b, a in zip(baseline_texts, augmented_texts) if a != b)
    print(f"[run] N={n} ({n_with_caption} have a caption available for Part 2)")

    # --------- Encode raw features once ---------
    print(f"[features] encoding image features", flush=True)
    img_raw = extract_image_features_siglip2(
        images, siglip_model, siglip_ip,
        batch_size=args.image_batch_size, device=args.device,
    )
    print(f"[features] encoding baseline text features", flush=True)
    txt_baseline_raw = extract_text_features_siglip2(
        baseline_texts, siglip_model, siglip_tok,
        batch_size=args.text_batch_size, device=args.device,
    )
    txt_augmented_raw = None
    if not args.skip_retrain:
        print(f"[features] encoding all-augmented text features", flush=True)
        txt_augmented_raw = extract_text_features_siglip2(
            augmented_texts, siglip_model, siglip_tok,
            batch_size=args.text_batch_size, device=args.device,
        )

    # Free SigLIP weights now that encoding is done.
    del siglip_model, siglip_tok, siglip_ip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    targets_np = np.asarray(targets, dtype=np.int64)
    targets_t = torch.tensor(targets, dtype=torch.long, device=args.device)

    # --------- Part 1: frozen estimator + legacy PCA ---------
    print("[part1] applying legacy PCA + scoring with frozen estimator", flush=True)
    img_pca_legacy = apply_pca(img_raw, legacy_pca["pca_img_mean"], legacy_pca["pca_img_components"])
    txt_pca_legacy = apply_pca(txt_baseline_raw, legacy_pca["pca_txt_mean"], legacy_pca["pca_txt_components"])
    per_sample_p1 = score(
        img_pca_legacy, txt_pca_legacy, targets_t,
        discriminator, entropy_estimator, n_classes,
        args.device, args.score_batch_size,
    )
    overall_p1 = dominance_proportions(per_sample_p1)
    means_p1 = expectations(per_sample_p1)
    per_split_p1, per_split_means_p1, n_per_split_p1 = ({}, {}, {})
    if args.per_split:
        per_split_p1, per_split_means_p1, n_per_split_p1 = per_split_breakdown(per_sample_p1, ids, data)

    print_summary(
        "Part 1 — frozen estimator", n,
        overall_p1, means_p1, per_split_p1, per_split_means_p1, n_per_split_p1,
    )
    plot_proportions(
        overall_p1, per_split_p1, n, n_per_split_p1, fig_path_p1,
        title="Dominant interaction proportions (frozen estimator, baseline text)",
    )
    print(f"[save] figure  → {fig_path_p1} (+ .pdf)", flush=True)

    record: Dict[str, Any] = {
        "n": n,
        "frozen": {
            "text_field": args.text_field,
            "overall_proportions": overall_p1,
            "overall_means": means_p1,
            "per_split_proportions": per_split_p1,
            "per_split_means": per_split_means_p1,
            "n_per_split": n_per_split_p1,
        },
    }

    # --------- Part 2: re-fit PCA on all-augmented features + retrain ---------
    if not args.skip_retrain:
        print("[part2] preparing all-augmented dataset (every sample uses captions)", flush=True)
        train_idx, val_idx, test_idx = stratified_split_indices(
            targets_np, args.train_ratio, args.val_ratio, args.retrain_seed,
        )
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        print(f"[part2] split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"(seed={args.retrain_seed})", flush=True)

        feat, img_red_p2, txt_red_p2 = build_and_save_all_augmented_features(
            img_raw=img_raw,
            txt_augmented_raw=txt_augmented_raw,
            targets_np=targets_np,
            ids=ids,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            save_path=args.features_out,
            target_dim=int(meta["input_size_1"]),
        )
        print(f"[part2] saved features → {args.features_out}", flush=True)

        retrain_cfg = make_retrain_cfg(
            device=args.device, n_classes=n_classes,
            embed_size=int(meta["embed_size"]),
            input_size_1=int(feat["train_modal_1_features"].shape[1]),
            input_size_2=int(feat["train_modal_2_features"].shape[1]),
            num_epochs_disc=args.retrain_epochs_disc,
            num_epochs_ent=args.retrain_epochs_ent,
        )
        setup_seed(args.retrain_seed)
        print(f"[part2] retraining estimator on all-augmented features "
              f"(in1={retrain_cfg.input_size_1}, in2={retrain_cfg.input_size_2})", flush=True)
        disc_new, ent_new = retrain_from_features(feat, retrain_cfg)
        save_trained_estimators(
            retrain_cfg, disc_new, ent_new,
            output_dir=args.estimator_out_dir, tag=args.estimator_tag,
        )

        # Score full set in the new PCA space.
        per_sample_p2 = score(
            img_red_p2, txt_red_p2, targets_t,
            disc_new, ent_new, n_classes,
            args.device, args.score_batch_size,
        )
        overall_p2 = dominance_proportions(per_sample_p2)
        means_p2 = expectations(per_sample_p2)
        per_split_p2, per_split_means_p2, n_per_split_p2 = ({}, {}, {})
        if args.per_split:
            per_split_p2, per_split_means_p2, n_per_split_p2 = per_split_breakdown(
                per_sample_p2, ids, data,
            )

        print_summary(
            "Part 2 — retrained estimator (all-augmented)", n,
            overall_p2, means_p2, per_split_p2, per_split_means_p2, n_per_split_p2,
        )
        plot_proportions(
            overall_p2, per_split_p2, n, n_per_split_p2, fig_path_p2,
            title="Dominant interaction proportions (retrained on all-augmented features)",
        )
        print(f"[save] figure  → {fig_path_p2} (+ .pdf)", flush=True)

        record["retrained_all_augmented"] = {
            "caption_field": args.caption_field,
            "overall_proportions": overall_p2,
            "overall_means": means_p2,
            "per_split_proportions": per_split_p2,
            "per_split_means": per_split_means_p2,
            "n_per_split": n_per_split_p2,
            "features_path": args.features_out,
            "estimator_path": os.path.join(
                args.estimator_out_dir, f"mi_estimators_{args.estimator_tag}_state_dict.pt",
            ),
            "retrain_seed": int(args.retrain_seed),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
        }

    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\n[save] records → {json_path}")


if __name__ == "__main__":
    main()
