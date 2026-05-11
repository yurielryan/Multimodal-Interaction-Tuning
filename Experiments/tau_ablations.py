"""Tau ablations as the MI Gate captions more valid samples.

For τ ∈ {0%, 10%, …, 100%} this script replays the MI Gate on the dataset
and recomputes the dominance distribution. At each τ the top
``floor(τ · |valid|)`` U1-dominant samples (ranked by u1 score) have their
text replaced with ``original_text + " " + caption``; everyone else keeps
``original_text``.

Two passes are produced over the same τ grid:

- **Part 1 (frozen estimator)**: every τ is scored with the single
  pre-trained estimator at ``Estimator/saved_estimators/mi_estimators_latest_state_dict.pt``.
  The decomposition shifts because the input distribution shifts under it.
- **Part 2 (online retraining)**: at every τ we (a) materialise the
  τ-augmented dataset — for the top ``floor(τ · |valid|)`` U1-dominant
  samples we use the *augmented* SigLIP text features (encoded over
  ``original + caption``), everyone else gets the *baseline* features —
  (b) fit a fresh PCA on this τ's train split, (c) save the resulting
  features as a standard .pt under
  ``Features/per_tau/features_tau{tau:.2f}_siglip2_pca512.pt`` (same
  schema as ``Features/hateful_memes_features_siglip2_pca512.pt``),
  (d) retrain the discriminators + entropy estimators from scratch on
  these features, and (e) score the full τ-set with the new estimator.
  Each retrained estimator is checkpointed to
  ``Estimator/saved_estimators/mi_estimators_tau{tau:.2f}_state_dict.pt``.

Hypothesis: in Part 1 we expect E[R] (and the R-dominant proportion) to
rise as τ grows because the caption injects image-derived information into
the text modality. In Part 2 the retrained estimator adapts to the new
distribution at each τ to match the distribution shift.

Part 1 reuses the same PCA features at
``Features/hateful_memes_features_siglip2_pca512.pt`` so the frozen
estimator stays in the same 512-dim space it was trained on. Part 2 fits
a *fresh* PCA per τ — each τ's saved .pt is therefore a self-contained,
standard-schema feature file. Image features and both text variants are
encoded by SigLIP once and reused (text mixing is exact: SigLIP encodes
each sample independently so the per-τ raw text features are equivalent
to re-encoding).

Run from repo root:

    python Experiments/tau_ablations.py                  # full sweep, both parts
    python Experiments/tau_ablations.py --skip_retrain   # frozen only
    python Experiments/tau_ablations.py --max_samples 200  # quick smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_FEATURES_PER_TAU_DIR = os.path.join(REPO_ROOT, "Features/per_tau")
DEFAULT_SIGLIP = "google/siglip2-giant-opt-patch16-384"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "Experiments/outputs")

LABEL_MAP = {"No.": 0, "Yes.": 1}
TERM_NAMES = ("R", "U1", "U2", "S")
PALETTE = {"R": "#4C78A8", "U1": "#F58518", "U2": "#54A24B", "S": "#E45756"}


# ---------------------------------------------------------------------------
# Subset collection
# ---------------------------------------------------------------------------
def collect_subset(
    data: List[dict],
    image_dir: str,
    caption_field: str = "generated_caption_smolvlm",
    max_samples: int = None,
):
    """Pick rows with usable image + label, returning images, baseline text,
    augmented text (original + caption), targets, and ids."""
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
        aug = (original + " " + caption).strip() if caption else original
        images.append(Image.open(img_path).convert("RGB"))
        baseline.append(original)
        augmented.append(aug)
        targets.append(LABEL_MAP[ans])
        ids.append(int(s["id"]))
        if max_samples is not None and len(images) >= max_samples:
            break
    return images, baseline, augmented, targets, ids


# ---------------------------------------------------------------------------
# MI helpers
# ---------------------------------------------------------------------------
def score(
    img_pca, txt_pca, targets_t,
    discriminator, entropy_estimator, n_classes,
    device, score_batch_size,
):
    return score_per_sample(
        discriminator, entropy_estimator,
        img_pca.to(device), txt_pca.to(device), targets_t,
        device=device, n_classes=n_classes, batch_size=score_batch_size,
    )


def dominance_proportions(per_sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    keys = ("r", "u1", "u2", "s")
    n = len(per_sample[keys[0]])
    if n == 0:
        return {name: float("nan") for name in TERM_NAMES}
    stack = np.stack([per_sample[k] for k in keys], axis=1)
    counts = np.bincount(stack.argmax(axis=1), minlength=4)
    return {name: float(c) / n for name, c in zip(TERM_NAMES, counts)}


def expectations(per_sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    keys = ("r", "u1", "u2", "s")
    return {f"E[{name}]": float(per_sample[k].mean())
            for k, name in zip(keys, TERM_NAMES)}


def build_record(tau, n_chosen, n_valid, n_total, per_sample) -> Dict[str, Any]:
    rec = {"tau": float(tau), "n_chosen": int(n_chosen),
           "n_valid_baseline": int(n_valid), "n_total": int(n_total)}
    rec.update(expectations(per_sample))
    rec.update({f"prop_{name}": p for name, p in dominance_proportions(per_sample).items()})
    return rec


# ---------------------------------------------------------------------------
# Online retraining (Part 2)
# ---------------------------------------------------------------------------
class _RetrainCfg:
    """Lightweight cfg object satisfying the obtain_*/save_* APIs from
    Estimator/mi_estimator.py without pulling in Hydra."""
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
    cfg.save_full_models = False  # state_dict only at each τ to avoid bloat
    return cfg


def make_loader(img_pca, txt_pca, targets_cpu, indices, batch_size, shuffle):
    ds = feature_dataset(img_pca[indices], txt_pca[indices], targets_cpu[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_and_save_per_tau_features(
    tau: float,
    img_raw: torch.Tensor,
    txt_baseline_raw: torch.Tensor,
    txt_augmented_raw: torch.Tensor,
    targets_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ranked_valid: np.ndarray,
    n_valid: int,
    save_path: str,
    target_dim: int = 512,
):
    """Materialize the τ-augmented dataset and save it as a standard `.pt`.

    For the top ``floor(tau * |valid|)`` U1-dominant samples we use the
    augmented text (raw SigLIP features over ``original + caption``); for
    everyone else we use the baseline text. PCA is then fit on the τ-specific
    train split (so it reflects this τ's distribution) and applied to all
    rows. The saved blob matches the schema of
    ``Features/hateful_memes_features_siglip2_pca512.pt``, plus ``tau`` /
    ``n_chosen`` / ``chosen_indices`` for traceability.

    Returns: (feat_dict, img_reduced_full, txt_reduced_full, n_chosen, chosen_indices_sorted)
    """
    n_total = img_raw.shape[0]
    n_chosen = int(np.floor(float(tau) * n_valid))
    chosen_indices = ranked_valid[:n_chosen].astype(np.int64).tolist() if n_chosen > 0 else []

    # Build per-sample raw text features: augmented for chosen, baseline otherwise.
    mask = torch.zeros(n_total, dtype=torch.bool)
    if chosen_indices:
        mask[torch.tensor(sorted(chosen_indices), dtype=torch.long)] = True
    txt_mixed_raw = torch.where(mask.unsqueeze(1), txt_augmented_raw, txt_baseline_raw)

    # Per-τ PCA fit on the τ-specific train split.
    img_pca_d = pca_fit_on_train_and_transform(img_raw, train_idx, target_dim)
    txt_pca_d = pca_fit_on_train_and_transform(txt_mixed_raw, train_idx, target_dim) # includes the samples with captions.

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
        # Per-τ traceability
        "tau": float(tau),
        "n_chosen": int(n_chosen),
        "n_valid_baseline": int(n_valid),
        "n_total": int(n_total),
        "chosen_indices": list(map(int, chosen_indices)),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(feat, save_path)
    return feat, img_red, txt_red, n_chosen, sorted(chosen_indices)


def retrain_from_features(feat: dict, cfg: _RetrainCfg):
    """Train a fresh discriminator + entropy estimator from a per-τ features
    dict (already split into train/val tensors)."""
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
def _draw_dominance_bar(ax, records, title):
    taus = [r["tau"] for r in records]
    n_chosen = [r["n_chosen"] for r in records]
    x = np.arange(len(taus))
    bottoms = np.zeros(len(taus))
    for name in TERM_NAMES:
        vals = np.array([100.0 * r[f"prop_{name}"] for r in records])
        ax.bar(x, vals, 0.78, bottom=bottoms,
               color=PALETTE[name], edgecolor="white", linewidth=0.5,
               label=f"{name}-dominant")
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 4.0:
                ax.text(xi, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if name in ("R", "S") else "black")
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Dominance proportion (% of samples)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}\nk={k}" for t, k in zip(taus, n_chosen)])


def _draw_expectation_lines(ax, records, title):
    taus = [r["tau"] for r in records]
    x = np.arange(len(taus))
    for name in TERM_NAMES:
        ys = np.array([r[f"E[{name}]"] for r in records])
        ax.plot(x, ys, marker="o", linewidth=1.8, markersize=5,
                color=PALETTE[name], label=f"E[{name}]")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("Expectation (nats)")
    ax.set_xlabel("τ (fraction of U1-dominant samples captioned)")
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in taus])


def plot_transfer(payload: Dict[str, List[dict]], out_path: str) -> None:
    """Render Part 1 (and optionally Part 2) as a 2×2 (or 1×2) figure.

    Layout when both parts are present:

        [Frozen — dominance bar]   [Frozen — expectations line]
        [Retrained — dominance bar][Retrained — expectations line]
    """
    parts: List[Tuple[str, List[dict]]] = []
    if "frozen" in payload and payload["frozen"]:
        parts.append(("Frozen estimator (single, pre-trained)", payload["frozen"]))
    if "retrained" in payload and payload["retrained"]:
        parts.append(("Retrained estimator (fresh per τ)", payload["retrained"]))
    if not parts:
        raise ValueError("No records to plot.")

    n_rows = len(parts)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(14.0, 4.6 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0], "hspace": 0.45, "wspace": 0.25},
    )

    fig.suptitle(
        "Tau ablation: MI Gate captions top floor(τ·|valid|) U1-dominant samples",
        fontsize=13, y=0.995,
    )

    # First pass to render each row
    for row_idx, (label, records) in enumerate(parts):
        _draw_dominance_bar(axes[row_idx][0], records, f"{label} — dominance")
        _draw_expectation_lines(axes[row_idx][1], records, f"{label} — expectations")

    # Sync expectation y-limits across rows so trends are visually comparable.
    if n_rows > 1:
        all_lo, all_hi = [], []
        for ax in [axes[row][1] for row in range(n_rows)]:
            lo, hi = ax.get_ylim()
            all_lo.append(lo); all_hi.append(hi)
        lo, hi = min(all_lo), max(all_hi)
        for ax in [axes[row][1] for row in range(n_rows)]:
            ax.set_ylim(lo, hi)

    # Single shared legend at the bottom.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
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
    parser.add_argument("--estimator_out_dir", default=DEFAULT_ESTIMATOR_OUT_DIR,
                        help="Where per-τ retrained estimators get checkpointed.")
    parser.add_argument("--features_per_tau_dir", default=DEFAULT_FEATURES_PER_TAU_DIR,
                        help="Where per-τ feature .pt files get saved (Part 2).")
    parser.add_argument("--siglip_model_id", default=DEFAULT_SIGLIP)
    parser.add_argument("--caption_field", default="generated_caption_smolvlm")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap subset size for fast smoke runs.")
    parser.add_argument("--tau_step", type=float, default=0.1,
                        help="τ step size (default 0.1 → 11 points 0..1).")
    parser.add_argument("--skip_retrain", action="store_true",
                        help="Skip Part 2 (online retraining at each τ).")
    parser.add_argument("--retrain_epochs_disc", type=int, default=30)
    parser.add_argument("--retrain_epochs_ent", type=int, default=80)
    parser.add_argument("--retrain_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.estimator_out_dir, exist_ok=True)
    if not args.skip_retrain:
        os.makedirs(args.features_per_tau_dir, exist_ok=True)

    print(f"[setup] loading data from {args.data_path}", flush=True)
    with open(args.data_path) as fh:
        data = json.load(fh)
    print(f"[setup] {len(data)} rows in data.json")

    print(f"[setup] loading SigLIP2 ({args.siglip_model_id}) on {args.device}", flush=True)
    siglip_model, siglip_tok, siglip_ip = build_siglip2(args.siglip_model_id, device=args.device)
    legacy_pca = torch.load(args.legacy_features, map_location="cpu")
    frozen_disc, frozen_ent, meta = load_trained_estimators(
        args.estimator_ckpt, args.device,
    )
    n_classes = int(meta["n_classes"])

    # --------- Collect samples ---------
    print(f"[run] collecting samples (caption_field={args.caption_field})", flush=True)
    images, baseline_texts, augmented_texts, targets, ids = collect_subset(
        data, args.image_dir,
        caption_field=args.caption_field,
        max_samples=args.max_samples,
    )
    n_total = len(images)
    if n_total == 0:
        raise RuntimeError(
            f"No usable samples found under {args.image_dir}. "
            "Run Features/download_images.py or pass --image_dir."
        )
    n_with_caption = sum(1 for b, a in zip(baseline_texts, augmented_texts) if a != b)
    print(f"[run] N={n_total} ({n_with_caption} with caption available)")

    # --------- Encode features once ---------
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
    print(f"[features] encoding augmented text features", flush=True)
    txt_augmented_raw = extract_text_features_siglip2(
        augmented_texts, siglip_model, siglip_tok,
        batch_size=args.text_batch_size, device=args.device,
    )

    # Free SigLIP weights now that we're done encoding.
    del siglip_model, siglip_tok, siglip_ip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    img_pca = apply_pca(img_raw, legacy_pca["pca_img_mean"], legacy_pca["pca_img_components"])
    txt_baseline_pca = apply_pca(txt_baseline_raw, legacy_pca["pca_txt_mean"], legacy_pca["pca_txt_components"])
    txt_augmented_pca = apply_pca(txt_augmented_raw, legacy_pca["pca_txt_mean"], legacy_pca["pca_txt_components"])
    targets_cpu = torch.tensor(targets, dtype=torch.long)
    targets_t = targets_cpu.to(args.device)
    targets_np = np.asarray(targets, dtype=np.int64)

    # --------- Initial scoring with baseline text → identify U1-dominant ranking ---------
    print("[run] initial scoring with baseline text → ranking U1-dominant samples", flush=True)
    baseline_scores = score(
        img_pca, txt_baseline_pca, targets_t,
        frozen_disc, frozen_ent, n_classes,
        args.device, args.score_batch_size,
    )
    stack = np.stack([baseline_scores[k] for k in ("r", "u1", "u2", "s")], axis=1)
    is_u1_dominant = stack.argmax(axis=1) == 1
    valid_idx = np.where(is_u1_dominant)[0]
    valid_u1 = baseline_scores["u1"][valid_idx]
    ranked_valid = valid_idx[np.argsort(-valid_u1)]
    n_valid = len(ranked_valid)
    print(f"[run] U1-dominant under baseline: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # --------- Stratified train/val split (shared across τ for Part 2) ---------
    if not args.skip_retrain:
        train_idx, val_idx, test_idx = stratified_split_indices(
            targets_np, args.train_ratio, args.val_ratio, args.retrain_seed,
        )
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        print(f"[setup] retrain split: train={len(train_idx)} val={len(val_idx)} "
              f"test={len(test_idx)} (seed={args.retrain_seed})")

    retrain_cfg = make_retrain_cfg(
        device=args.device, n_classes=n_classes,
        embed_size=int(meta["embed_size"]),
        input_size_1=int(meta["input_size_1"]),
        input_size_2=int(meta["input_size_2"]),
        num_epochs_disc=args.retrain_epochs_disc,
        num_epochs_ent=args.retrain_epochs_ent,
    )

    # --------- Sweep τ ---------
    taus = np.arange(0.0, 1.0 + 1e-9, args.tau_step)
    frozen_records: List[dict] = []
    retrained_records: List[dict] = []
    saved_feature_paths: List[str] = []

    for tau in taus:
        n_chosen = int(np.floor(float(tau) * n_valid))
        chosen = set(int(i) for i in ranked_valid[:n_chosen]) if n_chosen > 0 else set()

        # Mix of *legacy-PCA* features for Part 1 (frozen estimator was trained
        # on legacy PCA, so to be in-distribution we score in the same space).
        mask = torch.zeros(n_total, dtype=torch.bool)
        if chosen:
            idx = torch.tensor(sorted(chosen), dtype=torch.long)
            mask[idx] = True
        txt_mixed_legacy = torch.where(mask.unsqueeze(1), txt_augmented_pca, txt_baseline_pca)

        # ---- Part 1: frozen estimator ----
        per_sample = score(
            img_pca, txt_mixed_legacy, targets_t,
            frozen_disc, frozen_ent, n_classes,
            args.device, args.score_batch_size,
        )
        rec = build_record(tau, n_chosen, n_valid, n_total, per_sample)
        frozen_records.append(rec)
        print(
            f"[frozen τ={tau:.1f}] k={n_chosen:>4d}/{n_valid}  "
            f"E[R]={rec['E[R]']:+.4f}  E[U1]={rec['E[U1]']:+.4f}  "
            f"E[U2]={rec['E[U2]']:+.4f}  E[S]={rec['E[S]']:+.4f}  | "
            f"R={100*rec['prop_R']:5.1f}%  U1={100*rec['prop_U1']:5.1f}%  "
            f"U2={100*rec['prop_U2']:5.1f}%  S={100*rec['prop_S']:5.1f}%",
            flush=True,
        )

        # ---- Part 2: build per-τ dataset → save .pt → retrain → score ----
        if not args.skip_retrain:
            features_path = os.path.join(
                args.features_per_tau_dir,
                f"features_tau{tau:.2f}_siglip2_pca512.pt",
            )
            print(
                f"[features τ={tau:.1f}] building per-τ dataset "
                f"(k={n_chosen}/{n_valid}) → {features_path}",
                flush=True,
            )
            feat, img_red_tau, txt_red_tau, _, _ = build_and_save_per_tau_features(
                tau=float(tau),
                img_raw=img_raw,
                txt_baseline_raw=txt_baseline_raw,
                txt_augmented_raw=txt_augmented_raw,
                targets_np=targets_np,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                ranked_valid=ranked_valid,
                n_valid=n_valid,
                save_path=features_path,
                target_dim=int(meta["input_size_1"]),
            )
            saved_feature_paths.append(features_path)

            # Effective PCA dim may be < target_dim when N_train is small (e.g., smoke
            # runs). Mirror the actual shape into retrain_cfg so cls_network is sized
            # to match the saved features.
            retrain_cfg.input_size_1 = int(feat["train_modal_1_features"].shape[1])
            retrain_cfg.input_size_2 = int(feat["train_modal_2_features"].shape[1])

            setup_seed(args.retrain_seed)  # deterministic across τ for fair comparison
            print(f"[retrain τ={tau:.1f}] training fresh estimator on per-τ features "
                  f"(in1={retrain_cfg.input_size_1}, in2={retrain_cfg.input_size_2})", flush=True)
            disc_new, ent_new = retrain_from_features(feat, retrain_cfg)
            tag = f"tau{tau:.2f}"
            save_trained_estimators(
                retrain_cfg, disc_new, ent_new,
                output_dir=args.estimator_out_dir, tag=tag,
            )

            # Score on the full set in this τ's PCA space.
            per_sample_rt = score(
                img_red_tau, txt_red_tau, targets_t,
                disc_new, ent_new, n_classes,
                args.device, args.score_batch_size,
            )
            rec_rt = build_record(tau, n_chosen, n_valid, n_total, per_sample_rt)
            rec_rt["features_path"] = features_path
            retrained_records.append(rec_rt)
            print(
                f"[retrained τ={tau:.1f}] k={n_chosen:>4d}/{n_valid}  "
                f"E[R]={rec_rt['E[R]']:+.4f}  E[U1]={rec_rt['E[U1]']:+.4f}  "
                f"E[U2]={rec_rt['E[U2]']:+.4f}  E[S]={rec_rt['E[S]']:+.4f}  | "
                f"R={100*rec_rt['prop_R']:5.1f}%  U1={100*rec_rt['prop_U1']:5.1f}%  "
                f"U2={100*rec_rt['prop_U2']:5.1f}%  S={100*rec_rt['prop_S']:5.1f}%",
                flush=True,
            )
            # Free retrained models + intermediate tensors before next iteration.
            del disc_new, ent_new, feat, img_red_tau, txt_red_tau
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------- Save ---------
    payload: Dict[str, Any] = {
        "frozen": frozen_records,
        "n_total": int(n_total),
        "n_valid_baseline": int(n_valid),
        "tau_step": float(args.tau_step),
    }
    if not args.skip_retrain:
        payload["retrained"] = retrained_records
        payload["retrain_seed"] = int(args.retrain_seed)
        payload["train_size"] = int(len(train_idx))
        payload["val_size"] = int(len(val_idx))
        payload["features_per_tau_dir"] = args.features_per_tau_dir
        payload["features_per_tau_paths"] = saved_feature_paths

    json_path = os.path.join(args.out_dir, "tau_ablations.json")
    fig_path = os.path.join(args.out_dir, "tau_ablations.png")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[save] records → {json_path}")
    plot_transfer(payload, fig_path)
    print(f"[save] figure  → {fig_path} (+ .pdf)")
    if not args.skip_retrain:
        print(f"[save] retrained estimators → {args.estimator_out_dir}/mi_estimators_tau*.pt")
        print(f"[save] per-τ feature .pt    → {args.features_per_tau_dir}/features_tau*.pt")


if __name__ == "__main__":
    main()
