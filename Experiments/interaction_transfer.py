"""Compare interaction decompositions across text-modality variants.

Iterates every ``.pt`` feature file in ``Features/interaction_transfers/``,
trains a fresh discriminator + entropy estimator on each, evaluates the
expectations E[R] / E[U1] / E[U2] / E[S] on the train, val, and test splits,
and prints a summary table grouped by embedding size (1536/raw, 1024, 512).

Each file's name encodes its modality variant and PCA dimension:

- ``hateful_memes_features_siglip2_{1536,pca1024,pca512}.pt`` — Baseline
  (image + original meme text only, no caption).
- ``hateful_memes_features_with_random_text_siglip2_{raw,pca1024,pca512}.pt``
  — Random Text (text replaced with random English / lorem-ipsum-style
  strings; isolates the effect of blindly enlarging the text input with
  no label-relevant signal).
- ``combined_smolvlm_features_siglip2_{raw,pca1024,pca512}.pt`` — SmolVLM 2B
  (text augmented with SmolVLM captions).
- ``hateful_memes_features_with_captions_siglip2_{1536,pca1024,pca512}.pt``
  — Qwen2.5 32B (text augmented with Qwen2.5-VL 32B captions).

For every PCA size the table reports each variant's absolute R / U1 / U2 / S
and the relative change vs. the Baseline at that same size, mirroring the
"interaction transfer" comparison in the paper.

Run from repo root:

    python Experiments/interaction_transfer.py
    python Experiments/interaction_transfer.py --features_dir Features/interaction_transfers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTIMATOR_DIR = os.path.join(REPO_ROOT, "Estimator")
for p in (REPO_ROOT, ESTIMATOR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from Estimator.utils import feature_dataset, setup_seed  # noqa: E402
from Estimator.mi_estimator import (  # noqa: E402
    LSMI_estimation,
    obtain_discriminator,
    obtain_entropy_estimator,
)

DEFAULT_FEATURES_DIR = os.path.join(REPO_ROOT, "Features", "interaction_transfers")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "Experiments", "outputs")

VARIANT_BASELINE = "Baseline"
VARIANT_RANDOM = "Random Text"
VARIANT_SMOLVLM = "SmolVLM 2B"
VARIANT_QWEN = "Qwen2.5 32B"
VARIANT_ORDER = (VARIANT_BASELINE, VARIANT_RANDOM, VARIANT_SMOLVLM, VARIANT_QWEN)

SPLIT_ORDER = ("train", "val", "test")
SPLIT_LABELS = {"train": "Training Set", "val": "Validation Set", "test": "Test Set"}
TERM_KEYS = ("R", "U1", "U2", "S")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def classify_filename(fname: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (variant, embed_size) for a feature filename, or (None, None).

    embed_size is the PCA target dim, or the raw SigLIP dim (1536) when the
    file is unreduced (``_raw`` or ``_1536`` suffix).
    """
    base = os.path.basename(fname)
    # Variant — order matters: check the more specific names first.
    if "with_random_text" in base:
        variant = VARIANT_RANDOM
    elif base.startswith("combined_smolvlm_features"):
        variant = VARIANT_SMOLVLM
    elif "with_captions" in base:
        variant = VARIANT_QWEN
    elif base.startswith("hateful_memes_features_siglip2"):
        variant = VARIANT_BASELINE
    else:
        return None, None

    m = re.search(r"(?:_pca(\d+)|_(\d+)|_raw)\.pt$", base)
    if m is None:
        return variant, None
    if m.group(0) == "_raw.pt":
        return variant, 1536
    size = m.group(1) or m.group(2)
    return variant, int(size)


# ---------------------------------------------------------------------------
# Lightweight cfg (mirrors train.yaml fields the Estimator helpers read)
# ---------------------------------------------------------------------------
class _Cfg:
    pass


def make_cfg(
    device: str,
    n_classes: int,
    input_size_1: int,
    input_size_2: int,
    embed_size: int,
    batch_size: int,
    num_epochs_disc: int,
    num_epochs_ent: int,
    es_patience: int,
    es_min_delta: float,
) -> _Cfg:
    cfg = _Cfg()
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


# ---------------------------------------------------------------------------
# Per-file train + evaluate
# ---------------------------------------------------------------------------
def loaders_from_pt(data: dict, batch_size: int) -> Dict[str, DataLoader]:
    splits = {
        "train": ("train_modal_1_features", "train_modal_2_features", "train_targets"),
        "val":   ("val_modal_1_features",   "val_modal_2_features",   "val_targets"),
        "test":  ("test_modal_1_features",  "test_modal_2_features",  "test_targets"),
    }
    out: Dict[str, DataLoader] = {}
    for split, (k1, k2, ky) in splits.items():
        if not all(k in data for k in (k1, k2, ky)):
            continue
        ds = feature_dataset(data[k1], data[k2], data[ky])
        out[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
    return out


def train_and_eval(
    pt_path: str,
    args,
) -> Dict[str, Dict[str, float]]:
    """Train fresh estimators on one feature .pt and return per-split RUS means."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    in1 = int(data["train_modal_1_features"].shape[1])
    in2 = int(data["train_modal_2_features"].shape[1])
    cfg = make_cfg(
        device=args.device,
        n_classes=args.n_classes,
        input_size_1=in1,
        input_size_2=in2,
        embed_size=args.embed_size,
        batch_size=args.batch_size,
        num_epochs_disc=args.epochs_disc,
        num_epochs_ent=args.epochs_ent,
        es_patience=args.es_patience,
        es_min_delta=args.es_min_delta,
    )

    loaders = loaders_from_pt(data, cfg.batch_size)
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError(f"{pt_path} is missing train/val splits")

    setup_seed(args.seed)
    disc = obtain_discriminator(cfg, loaders["train"], loaders["val"])
    ent = obtain_entropy_estimator(cfg, loaders["train"], loaders["val"])

    results: Dict[str, Dict[str, float]] = {}
    for split in SPLIT_ORDER:
        if split not in loaders:
            continue
        print(f"  [Eval] {split} split:")
        # LSMI_estimation prints "R: .. U1: .. U2: .. S: .." internally.
        R, U1, U2, S = LSMI_estimation(loaders[split], disc, ent, cfg)
        results[split] = {
            "R":  float(R.detach().cpu().item()),
            "U1": float(U1.detach().cpu().item()),
            "U2": float(U2.detach().cpu().item()),
            "S":  float(S.detach().cpu().item()),
        }

    # Free per-file models before next iteration.
    del disc, ent, data, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _fmt_pct(value: float, baseline: float) -> str:
    """Return the %-change suffix used in the paper's interaction-transfer table."""
    if baseline is None:
        return ""
    if abs(baseline) < 1e-9:
        return ""  # baseline is exactly zero — % change is undefined
    pct = 100.0 * (value - baseline) / abs(baseline)
    sign = "+" if pct >= 0 else ""
    return f" ({sign}{pct:.0f}%)"


def _fmt_cell(value: float, baseline: Optional[float], is_baseline: bool) -> str:
    cell = f"{value:.4f}"
    if not is_baseline and baseline is not None:
        cell += _fmt_pct(value, baseline)
    return cell


def print_summary_table(
    # results[size][variant][split] = {R, U1, U2, S}
    results: Dict[int, Dict[str, Dict[str, Dict[str, float]]]],
) -> None:
    print()
    print("=" * 78)
    print("Interaction transfer summary")
    print("(absolute scores; parenthesised values are % change vs. Baseline at the same size)")
    print("=" * 78)

    sizes = sorted(results.keys(), reverse=True)  # 1536, 1024, 512
    col_w = 22
    name_w = 16
    header_terms = ["R", "U_V", "U_T", "S"]

    for size in sizes:
        print()
        print(f"--- Embedding size: {size} ---")
        per_variant = results[size]

        # Header
        head = f"{'Model':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in header_terms)
        print(head)

        for split in SPLIT_ORDER:
            # Skip split if no variant has it.
            if not any(split in per_variant.get(v, {}) for v in VARIANT_ORDER):
                continue
            print()
            print(f"  {SPLIT_LABELS[split]}")
            base_vals = per_variant.get(VARIANT_BASELINE, {}).get(split)
            for variant in VARIANT_ORDER:
                if variant not in per_variant or split not in per_variant[variant]:
                    continue
                vals = per_variant[variant][split]
                cells = [
                    _fmt_cell(
                        vals[k],
                        base_vals[k] if base_vals is not None else None,
                        is_baseline=(variant == VARIANT_BASELINE),
                    )
                    for k in TERM_KEYS
                ]
                row = f"  {variant:<{name_w-2}}" + "".join(f"{c:>{col_w}}" for c in cells)
                print(row)
        print()
    print("=" * 78)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_disc", type=int, default=30)
    parser.add_argument("--epochs_ent", type=int, default=80)
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only_size", type=int, default=None,
        help="Restrict to a single embedding size (e.g., 512) for quick runs.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(args.features_dir):
        raise FileNotFoundError(f"Features dir not found: {args.features_dir}")

    pt_files = sorted(
        os.path.join(args.features_dir, f)
        for f in os.listdir(args.features_dir)
        if f.endswith(".pt")
    )
    if not pt_files:
        raise RuntimeError(f"No .pt files found under {args.features_dir}")

    # results[size][variant][split] = {R, U1, U2, S}
    results: "OrderedDict[int, Dict[str, Dict[str, Dict[str, float]]]]" = OrderedDict()
    json_path = os.path.join(args.out_dir, "interaction_transfer.json")

    def _save_results() -> None:
        serializable = {
            str(size): {
                variant: {split: vals for split, vals in per_split.items()}
                for variant, per_split in per_variant.items()
            }
            for size, per_variant in results.items()
        }
        # Atomic write so a crash mid-dump can't truncate the file.
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(serializable, fh, indent=2)
        os.replace(tmp_path, json_path)

    for pt_path in pt_files:
        variant, size = classify_filename(pt_path)
        if variant is None or size is None:
            print(f"[skip] could not classify: {os.path.basename(pt_path)}")
            continue
        if args.only_size is not None and size != args.only_size:
            continue

        print()
        print("=" * 78)
        print(f"[run] {variant}  |  size={size}  |  {os.path.basename(pt_path)}")
        print("=" * 78)
        per_split = train_and_eval(pt_path, args)
        results.setdefault(size, {})[variant] = per_split

        # Save after every file so partial sweeps still produce usable output.
        _save_results()
        print(f"[save] records → {json_path}")

    if not results:
        raise RuntimeError("No results produced — check --features_dir and filenames.")

    print_summary_table(results)
    _save_results()
    print(f"[save] final records → {json_path}")


if __name__ == "__main__":
    main()
