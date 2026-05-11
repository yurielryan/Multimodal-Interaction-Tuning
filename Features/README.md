# `Features/` — feature extraction and shipped feature files

Everything related to turning raw `(image, text, label)` triples into the
fixed-shape modality tensors that the PID estimator consumes.

```
Features/
    download_images.py                              # idempotent HatefulMemes image downloader
    prepare_features.py                             # batch-level SigLIP2 extraction (used by SFT)
    utils_features.py                               # SigLIP2 loader, PCA, stratified split
    hateful_memes_features_siglip2_pca512.pt        # legacy PCA-512 feature file (estimator + MI Gate)
    features_all_augmented_siglip2_pca512.pt        # all-captioned variant (used by proportion_of_interactions.py)
    interaction_transfers/                          # shipped .pt files for Experiments/interaction_transfer.py
    per_tau/                                        # shipped .pt files for Experiments/tau_ablations.py
    README.md                                       # this file
```

## Standard feature `.pt` schema

Every feature file in this directory follows the same base schema, so the
estimator's `get_loader` (in `Estimator/utils.py`) can consume any of
them interchangeably. Counts below are for the full HatefulMemes split
(N = 8500 = 5949 train / 1274 val / 1277 test).

| Key | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `train_modal_1_features` | `(N_train, D)` | `float32` | Image features for the train split |
| `train_modal_2_features` | `(N_train, D)` | `float32` | Text features for the train split |
| `train_targets` | `(N_train,)` | `int64` | Train labels (0 / 1 for HatefulMemes) |
| `val_modal_1_features` | `(N_val, D)` | `float32` | Image features for the val split |
| `val_modal_2_features` | `(N_val, D)` | `float32` | Text features for the val split |
| `val_targets` | `(N_val,)` | `int64` | Val labels |
| `test_modal_1_features` | `(N_test, D)` | `float32` | Image features for the test split |
| `test_modal_2_features` | `(N_test, D)` | `float32` | Text features for the test split |
| `test_targets` | `(N_test,)` | `int64` | Test labels |
| `pca_img_mean` | `(1536,)` | `float32` | Mean used to centre raw image features before PCA |
| `pca_img_components` | `(1536, D)` | `float32` | Image PCA basis (right singular vectors) |
| `pca_img_explained_variance` | `(D,)` | `float32` | Per-component variance |
| `pca_img_explained_variance_ratio` | `(D,)` | `float32` | Per-component variance ratio |
| `pca_txt_mean` | `(1536,)` | `float32` | Same as above for the text modality |
| `pca_txt_components` | `(1536, D)` | `float32` | |
| `pca_txt_explained_variance` | `(D,)` | `float32` | |
| `pca_txt_explained_variance_ratio` | `(D,)` | `float32` | |
| `pca_target_dim` | scalar `int` | — | PCA target dim requested at fit time (typically `D`) |
| `pca_img_effective_dim` | scalar `int` | — | Actual PCA dim used for images (≤ `pca_target_dim`) |
| `pca_txt_effective_dim` | scalar `int` | — | Actual PCA dim used for text |
| `pca_enabled` | scalar `bool` | — | `True` if features are PCA-reduced; `False` for raw 1536-dim files |

`D` is `pca_img_effective_dim` (= `pca_txt_effective_dim` in practice) —
512, 1024, or 1536 depending on the file.

> **Raw vs. PCA files.** Files whose suffix is `_1536` or `_raw` carry
> raw 1536-dim SigLIP2 embeddings (no projection); the `pca_*` fields are
> still present (with `pca_target_dim = 1536` and `components` an
> identity-like 1536×1536 basis) so downstream code that expects them
> doesn't need to special-case anything.

### Optional extension keys

Some files add extra metadata for traceability. None of these are read
by the estimator itself — they exist so downstream scripts can verify
which τ / experiment a file belongs to.

`features_all_augmented_siglip2_pca512.pt` adds:

| Key | Type | Meaning |
| --- | --- | --- |
| `experiment` | `str` | Always `"all_augmented"` |
| `n_total` | `int` | Total rows (8500 = train + val + test) |
| `ids` | `int64` `(n_total,)` | `data.json` `id` values aligned with the row order |
| `train_idx` / `val_idx` / `test_idx` | `int64` | Row indices into `ids` for each split |

`per_tau/features_tau{tau:.2f}_siglip2_pca512.pt` adds:

| Key | Type | Meaning |
| --- | --- | --- |
| `tau` | `float` | The τ value (0.0 … 1.0) this file was built for |
| `n_chosen` | `int` | `floor(τ · n_valid_baseline)` — how many U1-dominant rows were captioned |
| `n_valid_baseline` | `int` | Number of U1-dominant rows under the baseline scoring pass |
| `n_total` | `int` | Total rows |
| `chosen_indices` | `list[int]` | Sorted row indices (length `n_chosen`) that received `original_text + " " + caption` |
| `train_idx` / `val_idx` / `test_idx` | `int64` | Stratified split indices used for this τ |


## Shipped `.pt` files

### `hateful_memes_features_siglip2_pca512.pt`

The legacy PCA-512 SigLIP2 feature file. This is what:

- the estimator config (`Estimator/train.yaml`) points to by default for
  PID estimator training, and
- `SFT/preprocess_mi_gate.py` uses as the source of the PCA fit when
  re-encoding new images (so the new features land in the same 512-dim
  space the trained estimators were trained on).

Modality 1 = SigLIP2 image embedding of `{id}.png`. Modality 2 = SigLIP2
text embedding of `original_text`.

### `features_all_augmented_siglip2_pca512.pt`

Same schema as above, but every row's text features come from
`original_text + " " + generated_caption_smolvlm` (the all-captioned
variant). Produced by `Experiments/proportion_of_interactions.py`'s
Part 2 and consumed back by it on subsequent runs.

### `interaction_transfers/` — Quick Start feature files

12 files (4 text-modality variants × 3 PCA sizes) used by
[`Experiments/interaction_transfer.py`](../Experiments/interaction_transfer.py)
to reproduce the paper's interaction-transfer table. **All 12 are
shipped with the repo** — no preprocessing needed to run the Quick Start.

Modality 1 (image) is always the SigLIP2 image embedding of `{id}.png`.
Only the text modality (modality 2) changes between variants:

| Filename | Variant | Text fed to SigLIP2 text encoder |
| --- | --- | --- |
| `hateful_memes_features_siglip2_{1536,pca1024,pca512}.pt` | Baseline | `original_text` |
| `hateful_memes_features_with_random_text_siglip2_{1536,pca1024,pca512}.pt` | Random Text | Random English / lorem-ipsum-style strings (no label-relevant signal — control) |
| `combined_smolvlm_features_siglip2_{1536,pca1024,pca512}.pt` | SmolVLM 2B | `original_text + " " + smolvlm_caption` |
| `hateful_memes_features_with_captions_siglip2_{1536,pca1024,pca512}.pt` | Qwen2.5 32B | `original_text + " " + qwen_caption` |

PCA sizes available per variant: `512`, `1024`, and raw 1536 (suffix
varies — `_1536` for the Baseline / Qwen / SmolVLM / Random Text trees;
the `_raw` suffix is also accepted by `interaction_transfer.py`'s filename
parser when present).

These files are **inputs only** — the experiment script does not
overwrite them.

### `per_tau/` — τ-ablation feature files (generated on demand)

Used by [`Experiments/tau_ablations.py`](../Experiments/tau_ablations.py)'s
Part 2 (online retraining at each τ). Each file is a self-contained
PCA-512 dataset where the top `floor(τ · n_valid)` U1-dominant rows have
their text features built from `original_text + " " + caption` and the
rest from `original_text` only. PCA is fit per-τ on that τ's train split.

**Not shipped** — these files are written by `tau_ablations.py` itself the
first time you run it (one file per τ in `Experiments/tau_ablations.py`'s
default grid `{0.00, 0.10, …, 1.00}`, ~41 MB each). They land here so
subsequent runs can iterate on the figure / JSON without redoing the
SigLIP2 encoding.


## Scripts

### `download_images.py`

Idempotent downloader for the HatefulMemes images. Pulls
`HuggingFaceM4/the_cauldron`, `hateful_memes` config, indexed
positionally so Cauldron row N maps to `{N}.png` (verified
byte-identical against pre-existing local images at ids 3, 100, 1018,
8497).

```
python Features/download_images.py                                # → <repo>/data/images
python Features/download_images.py --out_dir /scratch/HM/images   # custom location
python Features/download_images.py --overwrite                    # force re-download
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out_dir` | `<repo>/data/images` | Where `{id}.png` files land |
| `--overwrite` | off | Re-save even if the file already exists |
| `--config` | `hateful_memes` | Cauldron config name |
| `--repo` | `HuggingFaceM4/the_cauldron` | HF dataset repo |

### `prepare_features.py`

Batch-level SigLIP2 feature extraction designed to slot into a
dataloader. Used by `Estimator/estimate_batch.py` and the SFT pipeline,
not for offline `.pt` materialisation.

- `load_siglip2_extractor(model_id, device)` → `(model, tokenizer,
  image_processor, device_str)`
- `extract_features_from_batch(batch, model, tokenizer, image_processor,
  device, pca_512=True, image_batch_size=32, text_batch_size=128)` →
  `{"modal_1_features": [N, D], "modal_2_features": [N, D]}`

Batch formats supported:

- dict batch: `{"modal_1": <images>, "modal_2": <texts>}`
- tuple/list batch: `(images, texts, …)`

`pca_512=True` fits a fresh PCA on the current batch and projects to
512-dim. This is **only useful for ad-hoc batch-level scoring** — for
estimator training or evaluation you want a globally-fit PCA, which is
what the shipped `.pt` files contain.

### `utils_features.py`

Shared helpers used by the SFT pipeline and by the
`Experiments/proportion_of_interactions.py` /
`Experiments/tau_ablations.py` retraining paths.

Highlights:

- `build_siglip2(model_id, device)` — loads SigLIP2 model, tokenizer,
  image processor (`google/siglip2-giant-opt-patch16-384` is the default
  used elsewhere in the repo).
- `extract_image_features_siglip2(pil_images, model, image_processor,
  batch_size, device)` — returns L2-normalised SigLIP2 image embeddings.
- `extract_text_features_siglip2(texts, model, tokenizer, batch_size,
  device)` — same for text. Texts are lower-cased before tokenisation.
- `pca_fit_on_train_and_transform(x_all, idx_train, target_dim)` — fits
  `torch.pca_lowrank` on the train subset and projects all rows.
  Returns `{reduced, mean, components, explained_variance,
  explained_variance_ratio, effective_dim}`.
- `stratified_split_indices(labels, train_ratio=0.7, val_ratio=0.15,
  seed=42)` — returns class-balanced `(train_idx, val_idx, test_idx)`.
- `save_feature_tensors(features, out_path)` — convenience `torch.save`
  with directory creation.
- `build_qwen3_vl(model_id, device)` — loads Qwen3-VL for open-source
  captioning. Not used by the default pipeline (which captions with
  SmolVLM2 via `SFT/caption.py`); included here for ad-hoc use.
