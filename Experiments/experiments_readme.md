# Experiments

Standalone analyses that build on the trained PID estimators in
`Estimator/saved_estimators/`. Each script is intended to be runnable from
the repo root after the regular pipeline has been set up (data downloaded,
estimators trained).

```
Experiments/
    experiments_readme.md           # this file
    proportion_of_interactions.py   # dominance distribution over the dataset
    tau_ablations.py                # how dominance shifts as MI Gate captions more samples
    interaction_transfer.py         # how text-modality variants reshape R / U1 / U2 / S
    outputs/                        # generated figures + JSON dumps land here
```

## Note on τ

τ is **not** an axis in this directory. τ is the MI Gate's threshold —
the fraction of *valid* (U1-dominant) train samples that receive a caption
during SFT — and it lives in `SFT/preprocess_mi_gate.py`. The experiments
here characterize the dominance distribution of the dataset itself, i.e., the
pool the gate selects from.

## `tau_ablations.py` — dominance shift as the MI Gate captions more samples

Replays the MI Gate at τ ∈ {0%, 10%, …, 100%} and records, at each τ, the
expected R/U1/U2/S and the dominance proportions over the entire scored set.

At each τ the gate's selection is identical to what `SFT/preprocess_mi_gate.py`
would do: rank U1-dominant samples by `u1` (descending), take the top
`floor(τ · |valid|)`, and replace their text with
`original_text + " " + caption`. The remaining samples keep `original_text`.

Two passes over the same τ grid are produced:

1. **Part 1 — Frozen estimator.** Every τ is scored with the single
   pre-trained estimator at `Estimator/saved_estimators/mi_estimators_latest_state_dict.pt`.
   The decomposition shifts because the input distribution shifts under
   it; the estimator's parameters are held fixed.

2. **Part 2 — Online retraining.** At every τ a fresh, self-contained
   feature .pt is materialised for the τ-augmented dataset and used to
   retrain the estimator from scratch:

   - For the top `floor(τ · |valid|)` U1-dominant samples we use the
     SigLIP text features encoded over `original_text + " " + caption`;
     everyone else uses the SigLIP features over `original_text` only.
   - A new PCA (target dim 512, fit on this τ's stratified train split)
     is applied to both modalities — every τ therefore has its own PCA
     basis reflecting that τ's distribution.
   - The result is saved under
     `Features/per_tau/features_tau{tau:.2f}_siglip2_pca512.pt` in the
     same schema as `Features/hateful_memes_features_siglip2_pca512.pt`,
     plus per-τ traceability fields (`tau`, `n_chosen`, `chosen_indices`,
     `train_idx` / `val_idx` / `test_idx`).
   - The discriminators + entropy estimators are then retrained from
     scratch on this .pt and used to score the full τ-set in this τ's
     PCA space. Each retrained estimator is checkpointed to
     `Estimator/saved_estimators/mi_estimators_tau{tau:.2f}_state_dict.pt`
     (e.g., `mi_estimators_tau0.30_state_dict.pt` for τ=0.3). Pass
     `--skip_retrain` to disable Part 2.

   SigLIP encodes each sample independently, so the per-τ raw text
   features are obtained by selecting rows from the precomputed baseline
   and augmented encodings — equivalent to re-encoding from scratch but
   ~11× faster.

Hypothesis (Part 1): as τ grows we expect E[R] (and the R-dominant
proportion) to rise because the caption injects image-derived information
into the text modality, moving image-only content into the redundant
channel; E[U1] and the U1-dominant proportion should fall accordingly.

Part 2 controls for distribution shift: the retrained estimator is
in-distribution at every τ, so the dominance trend reflects the change in
the underlying joint p(x_V, x_T, y) rather than the original estimator's
extrapolation behavior.

### Efficiency

Image features and *both* text variants (baseline + augmented) are encoded
once. Each τ point is then a single O(N) frozen score pass plus, when
Part 2 is enabled, a fresh discriminator + entropy estimator training run
on the τ-augmented features. Full sweep on 8500 samples with default
training budgets (`--retrain_epochs_disc 30 --retrain_epochs_ent 80`,
both with early stopping) runs in roughly 30–45 minutes on an H200; the
frozen-only path is ~10 minutes.

### Output

- `outputs/tau_ablations.png` (and `.pdf`): a 2 × 2 figure when
  both parts are run (top row = frozen, bottom row = retrained), or 1 × 2
  when `--skip_retrain` is set. Each row is `[stacked-bar dominance | line
  plot of expectations]`. Y-limits on the expectation panels are
  synchronized across rows for direct visual comparison. Bar x-ticks show
  τ and the number of captioned samples k for each point.
- `outputs/tau_ablations.json`: a dict with `frozen` and (if
  enabled) `retrained` keys, each holding the per-τ records, plus
  metadata (`n_total`, `n_valid_baseline`, `tau_step`, `retrain_seed`,
  `train_size`, `val_size`).
- `Estimator/saved_estimators/mi_estimators_tau{tau:.2f}_state_dict.pt`:
  one state-dict checkpoint per τ when Part 2 runs.
- `Features/per_tau/features_tau{tau:.2f}_siglip2_pca512.pt`: one
  self-contained feature file per τ (Part 2 only). Same schema as the
  legacy `Features/hateful_memes_features_siglip2_pca512.pt`, plus
  τ-specific metadata.

### Running it

```
# Full sweep, both parts
python Experiments/tau_ablations.py

# Frozen-only (Part 1 only)
python Experiments/tau_ablations.py --skip_retrain

# Quick smoke (≈200 samples, 3 τ points, abbreviated training budgets)
python Experiments/tau_ablations.py \
    --max_samples 200 --tau_step 0.5 \
    --retrain_epochs_disc 3 --retrain_epochs_ent 4

# Use a different caption field (e.g., the qwen-generated caption)
python Experiments/tau_ablations.py --caption_field generated_caption_qwen

# Custom τ grid (e.g., 21 points instead of 11)
python Experiments/tau_ablations.py --tau_step 0.05
```

### Defaults and overrides

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data_path` | `<repo>/data/data.json` | Annotation file |
| `--image_dir` | `<repo>/data/images` | Where `{id}.png` files live |
| `--legacy_features` | `Features/hateful_memes_features_siglip2_pca512.pt` | Source of the PCA fit |
| `--estimator_ckpt` | `Estimator/saved_estimators/mi_estimators_latest_state_dict.pt` | Frozen estimator (Part 1) |
| `--estimator_out_dir` | `Estimator/saved_estimators` | Where per-τ retrained checkpoints land |
| `--features_per_tau_dir` | `Features/per_tau` | Where per-τ feature .pt files land (Part 2) |
| `--siglip_model_id` | `google/siglip2-giant-opt-patch16-384` | Feature encoder |
| `--caption_field` | `generated_caption_smolvlm` | `data.json` field used as the caption |
| `--out_dir` | `Experiments/outputs` | Where figure + JSON land |
| `--max_samples` | unset | Cap subset size for fast smoke runs |
| `--tau_step` | `0.1` | τ grid spacing |
| `--skip_retrain` | flag | Skip Part 2 (online retraining at each τ) |
| `--retrain_epochs_disc` / `--retrain_epochs_ent` | 30 / 80 | Per-τ training budgets (early stopping is on) |
| `--retrain_seed` | `42` | Seed for both the train/val split and per-τ training reset |
| `--train_ratio` / `--val_ratio` | 0.7 / 0.15 | Stratified split for retraining |
| `--device` | auto | `cuda` if available else `cpu` |
| `--image_batch_size` / `--text_batch_size` / `--score_batch_size` | 32 / 128 / 256 | Throughput knobs |


## `proportion_of_interactions.py` — dominance distribution

The script runs two passes back-to-back, each producing its own donut:

1. **Part 1 — Frozen estimator.** Scores every sample with the
   pre-trained estimator at
   `Estimator/saved_estimators/mi_estimators_latest_state_dict.pt` in the
   legacy PCA space. Text fed to SigLIP comes from `--text_field`
   (default `original_text`). Output:
   `outputs/proportion_of_interactions.png` (and `.pdf`).
2. **Part 2 — All-captioned retrain** (skip with `--skip_retrain`).
   Re-encodes the text with the caption appended for *every* sample
   (`original_text + " " + caption`), fits a fresh PCA on a stratified
   train split, saves the result as a standard-schema feature file at
   `Features/features_all_augmented_siglip2_pca512.pt`, retrains the
   discriminator + entropy estimators from scratch, saves them at
   `Estimator/saved_estimators/mi_estimators_all_augmented_state_dict.pt`,
   and rescores the full dataset in the new PCA space. Output:
   `outputs/proportion_of_interactions_retrained.png` (and `.pdf`).

For each pass, the script reports:

- **Expectations**: E[R], E[U1], E[U2], E[S].
- **Dominance proportions**: percentage of samples whose argmax over the
  four point-wise terms is R, U1, U2, S.

If `data.json` rows carry a `split` field (populated by
`SFT/preprocess_mi_gate.py`), the same numbers are reported per
train/val/test split. All numbers are printed to stdout for both passes.

### What the donut shows

Both donuts share the same layout:

- **Main donut**: dominance proportions over the whole scored set, with
  percentage and term labels overlaid on each wedge (wedges < 2% have
  their in-wedge label suppressed but still appear in the legend). The
  total N is shown in the center.
- **Per-split donuts** (only if `split` is present in `data.json` and
  `--no_per_split` is not set): one small donut per split (train / val
  / test) shown side-by-side, each with its own N in the center.

The raw numbers — overall + per-split proportions and means, plus the N
counts and (for Part 2) the saved features and estimator paths — are
written to `outputs/proportion_of_interactions.json` under the top-level
keys `frozen` and `retrained_all_augmented`.

You can re-render both figures from the existing JSON without rescoring
by passing `--plot_only`:

```
python Experiments/proportion_of_interactions.py --plot_only
```

### Running it

From the repo root:

```
# Full run, both parts (default)
python Experiments/proportion_of_interactions.py

# Part 1 only (no retraining)
python Experiments/proportion_of_interactions.py --skip_retrain

# Quick smoke run (≈200 samples; useful for verifying the pipeline end-to-end)
python Experiments/proportion_of_interactions.py --max_samples 200

# Skip per-split breakdown even if data.json has 'split' fields
python Experiments/proportion_of_interactions.py --no_per_split

# Score Part 1 with the captioned text instead of the prompt-only baseline
python Experiments/proportion_of_interactions.py --text_field combined_text

# Point at data living outside the repo root
python Experiments/proportion_of_interactions.py \
    --data_path /scratch/HatefulMemes/data.json \
    --image_dir /scratch/HatefulMemes/images
```

Part 1 reuses `SFT.preprocess_mi_gate.load_trained_estimators` and
`score_per_sample`, plus the legacy PCA fit at
`Features/hateful_memes_features_siglip2_pca512.pt`. Part 2 trains a
fresh estimator with `Estimator.mi_estimator.obtain_discriminator` /
`obtain_entropy_estimator` on the all-augmented feature .pt it just
saved.

### Defaults and overrides

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data_path` | `<repo>/data/data.json` | Annotation file |
| `--image_dir` | `<repo>/data/images` | Where `{id}.png` files live |
| `--legacy_features` | `Features/hateful_memes_features_siglip2_pca512.pt` | Source of the PCA fit (Part 1) |
| `--estimator_ckpt` | `Estimator/saved_estimators/mi_estimators_latest_state_dict.pt` | Frozen estimator (Part 1) |
| `--siglip_model_id` | `google/siglip2-giant-opt-patch16-384` | Feature encoder |
| `--text_field` | `original_text` | data.json field fed to SigLIP for Part 1 |
| `--caption_field` | `generated_caption_smolvlm` | data.json field appended for Part 2 |
| `--out_dir` | `Experiments/outputs` | Where figures + JSON land |
| `--max_samples` | unset | Cap subset size for fast smoke runs |
| `--no_per_split` | flag | Skip per-split breakdown |
| `--skip_retrain` | flag | Skip Part 2 entirely |
| `--features_out` | `Features/features_all_augmented_siglip2_pca512.pt` | Where Part 2 saves its features |
| `--estimator_out_dir` | `Estimator/saved_estimators` | Where Part 2 saves the retrained estimator |
| `--estimator_tag` | `all_augmented` | Suffix in the saved estimator filename |
| `--retrain_epochs_disc` / `--retrain_epochs_ent` | 30 / 80 | Per-Part-2 training budgets (early stopping is on) |
| `--retrain_seed` | `42` | Seed for both the train/val split and per-Part-2 training |
| `--train_ratio` / `--val_ratio` | 0.7 / 0.15 | Stratified split for retraining |
| `--device` | auto | `cuda` if available else `cpu` |
| `--image_batch_size` / `--text_batch_size` / `--score_batch_size` | 32 / 128 / 256 | Throughput knobs |


## `interaction_transfer.py` — text-modality variants vs. the PID decomposition

Compares how four text-modality variants reshape the redundant /
unique / synergistic decomposition relative to a baseline at three PCA
embedding sizes (1536/raw, 1024, 512). The script iterates every `.pt`
file in `Features/interaction_transfers/`, trains a fresh discriminator +
entropy estimator on each, evaluates `LSMI_estimation` on the train, val,
and test splits, and prints a grouped summary table where every
non-baseline cell is annotated with its % change vs. the Baseline at the
same size.

The four variants are inferred from the filename:

| Variant | Filename pattern | Text content |
| --- | --- | --- |
| Baseline | `hateful_memes_features_siglip2_{1536,pca1024,pca512}.pt` | Original meme text only |
| Random Text | `hateful_memes_features_with_random_text_siglip2_{raw,pca1024,pca512}.pt` | Text replaced with random English / lorem-ipsum-style strings (control for blindly enlarging the text input with no label-relevant signal) |
| SmolVLM 2B | `combined_smolvlm_features_siglip2_{raw,pca1024,pca512}.pt` | Original + SmolVLM 2B caption |
| Qwen2.5 32B | `hateful_memes_features_with_captions_siglip2_{1536,pca1024,pca512}.pt` | Original + Qwen2.5-VL 32B caption |

The PCA size is parsed from the same filename suffix (`_pca512`,
`_pca1024`, `_1536`, or `_raw` — the last two map to the raw 1536-dim
SigLIP space).

All caption-bearing variants concatenate as ``original_text + " " +
caption``, the same format the MI Gate writes to ``mi_gate_text``.

Hypothesis: as the caption source improves (Random Text → SmolVLM 2B →
Qwen2.5 32B), the text modality gains genuine image-derived signal,
which should push mass into R (and possibly S) and reduce the U1
contribution relative to the Baseline. Random Text serves as the
control: lorem-ipsum-style strings inflate marginal entropy without
adding label-relevant information, so its R / U1 / U2 / S deltas should
be small / noisy.

### Output

- Stdout: a `Model | R | U_V | U_T | S` table per embedding size, with
  one block per split (Training / Validation / Test). Each non-baseline
  cell is annotated as `value (+xx%)` relative to the Baseline at the
  same size, mirroring the interaction-transfer table in the paper.
- `outputs/interaction_transfer.json`: a nested dict
  `{size: {variant: {split: {R, U1, U2, S}}}}` written **after every
  feature file** (atomic tmp + rename) so a partial sweep still leaves
  usable results on disk, plus a final write after the summary table
  prints.

### Running it

```
# Full sweep over all .pt files in Features/interaction_transfers/
python Experiments/interaction_transfer.py

# Restrict to a single embedding size (quick smoke run)
python Experiments/interaction_transfer.py --only_size 512

# Point at a different feature directory
python Experiments/interaction_transfer.py \
    --features_dir /scratch/HatefulMemes/interaction_transfers
```

### Defaults and overrides

| Flag | Default | Meaning |
| --- | --- | --- |
| `--features_dir` | `Features/interaction_transfers` | Directory of `.pt` files to sweep |
| `--out_dir` | `Experiments/outputs` | Where the JSON dump lands |
| `--device` | auto | `cuda` if available else `cpu` |
| `--n_classes` | `2` | Label cardinality (HatefulMemes is binary) |
| `--embed_size` | `512` | Hidden width of the discriminator MLPs |
| `--batch_size` | `64` | Train / eval batch size |
| `--epochs_disc` / `--epochs_ent` | 30 / 80 | Per-file training budgets (early stopping is on) |
| `--es_patience` / `--es_min_delta` | 10 / 1e-4 | Early-stopping controls |
| `--seed` | `42` | Reset before each file's train run for fair comparison |
| `--only_size` | unset | Restrict to a single embedding size (e.g., `512`) |
