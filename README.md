# Multimodal Interaction Tuning (MIT)

This repository accompanies the ICML 2026 paper: 

[Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](https://arxiv.org/abs/2605.08145)

![](Poster.png)

It trains a partial-information-decomposition (PID) estimator over a multimodal dataset and uses the resulting per-sample interaction terms to gate a captioning step during supervised fine-tuning (SFT).

The four interaction terms are:

- **R**: redundant information shared by both modalities
- **U1** (a.k.a. **U_V**): unique information from modality 1 (image)
- **U2** (a.k.a. **U_T**): unique information from modality 2 (text)
- **S**: synergistic information

Lower-case letters (`r, u1, u2, s`) denote the point-wise interaction terms.

The repository covers three things you can do, each independently runnable:

1. **Reproduce the paper's interaction-transfer table** from precomputed
   features shipped in this repo — no GPU-heavy preprocessing required.
   *(See [Quick Start](#quick-start-reproduce-the-interaction-transfer-table).)*
2. **Run additional PID experiments** (dominance distribution, τ ablations).
3. **Run the full SFT pipeline** end-to-end (download images, generate
   captions, score interactions, gate, fine-tune SmolVLM2).

**Note**: The might be some slight differences in the values as that depends on how your trained estimators converged. That said, you should expect the same general trends in Table 1 of the paper (ablation of interaction transfers).

*I have tried my best to ensure reproducibility in this work. Do reach out to me (the corresponding author) if there are any issues in the setup / data.* 🙇🏻

## Setup

Most stages assume a CUDA-capable GPU. The interaction-transfer Quick Start
runs on CPU but is much faster on GPU.

```
# 1) Clone and enter the repo
git clone https://github.com/yurielryan/Multimodal-Interaction-Tuning.git
cd Multimodal-Interaction-Tuning

# 2) Create and activate the environment
conda create -n MIT python=3.10 -y
conda activate MIT

# 3) Install dependencies
pip install -r requirements.txt
```


## Quick Start: reproduce the interaction-transfer table

This experiment from the paper compares how four text-modality variants reshape the PID decomposition relative to a baseline, at three
PCA embedding sizes (1536 = raw SigLIP2, 1024, 512). The required SigLIP2-encoded features are **shipped with the repo** under [`Features/interaction_transfers/`](Features/interaction_transfers/), so you can reproduce the table without downloading images, captioning, or running the full pipeline.

From the repo root:

```
python Experiments/interaction_transfer.py
```

What this does:

1. Iterates every `.pt` in `Features/interaction_transfers/` (12 files: 4
   variants × 3 sizes).
2. For each file, trains a fresh discriminator + entropy estimator
   (`Estimator/mi_estimator.py:obtain_discriminator` and
   `obtain_entropy_estimator`).
3. Evaluates `LSMI_estimation` on the train, val, and test splits.
4. Prints a grouped table per embedding size with each non-baseline cell
   annotated as `value (+xx%)` relative to the Baseline at that same size.
5. Streams results to
   [`Experiments/outputs/interaction_transfer.json`](Experiments/outputs/)
   after every file (atomic write), so a partial sweep still leaves usable
   numbers on disk.

The four variants are inferred from the filename:

| Variant | Filename pattern | Text content |
| --- | --- | --- |
| Baseline | `hateful_memes_features_siglip2_{1536,pca1024,pca512}.pt` | Original meme text only |
| Random Text | `hateful_memes_features_with_random_text_siglip2_{1536,pca1024,pca512}.pt` | gibberish strings (no label-relevant signal) |
| SmolVLM 2B | `combined_smolvlm_features_siglip2_{1536,pca1024,pca512}.pt` | `original_text + " " + smolvlm_caption` |
| Qwen2.5 32B | `hateful_memes_features_with_captions_siglip2_{1536,pca1024,pca512}.pt` | `original_text + " " + qwen_caption` |

For the full schema of these `.pt` files see
[`Features/README.md`](Features/README.md).

### Smoke run

The full sweep trains 12 estimators back-to-back. To verify the pipeline end-to-end in a few minutes, restrict to a single embedding size:

```
python Experiments/interaction_transfer.py --only_size 1024
```

### Useful flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--features_dir` | `Features/interaction_transfers` | Directory of `.pt` files to sweep |
| `--out_dir` | `Experiments/outputs` | Where the JSON dump lands |
| `--only_size` | unset | Restrict to one embedding size (e.g., `512`) for quick runs |
| `--epochs_disc` / `--epochs_ent` | 30 / 80 | Per-file training budgets (early stopping is on) |
| `--seed` | `42` | Reset before each file's training run |
| `--device` | auto | `cuda` if available else `cpu` |

See [`Experiments/experiments_readme.md`](Experiments/experiments_readme.md)
for the full flag table.


## Repository at a glance

```
MIT/
    Estimator/                       # PID estimator training + saved checkpoints
        mi_estimator.py              # train + evaluate the LSMI decomposition
        estimate_batch.py            # callable batch-level scoring (R/U1/U2/S)
        entropy_estimator.py         # KNIFE-based marginal entropy models
        utils.py                     # dataset / loader / cls_network
        train.yaml                   # Hydra config for mi_estimator.py
        saved_estimators/            # output: trained estimator checkpoints
    Features/                        # SigLIP2 feature extraction + shipped .pt files
        prepare_features.py          # batch-level feature extraction (used by SFT)
        utils_features.py            # SigLIP2 loader, PCA, stratified split helpers
        download_images.py           # idempotent HatefulMemes image downloader
        interaction_transfers/       # SHIPPED .pt files for the Quick Start
        per_tau/                     # generated on demand by tau_ablations.py (not shipped)
        README.md                    # schema + provenance for everything in here
    Experiments/                     # standalone analyses on top of the estimators
        interaction_transfer.py      # Quick-Start headline experiment
        proportion_of_interactions.py
        tau_ablations.py
        outputs/                     # generated figures + JSON dumps
        experiments_readme.md
    SFT/                             # MI-Gated supervised fine-tuning of SmolVLM2
        caption.py                   # SmolVLM2 captioner CLI + shared loader
        preprocess_mi_gate.py        # score interactions, apply gate, write data.json
        multimodal_interaction_tuning.py  # SFT entrypoint
        config.yaml                  # SFT config (key=value CLI overrides supported)
    scripts/
        run_pipeline.sh              # one-command pipeline (download → … → SFT)
        submit_slurm.sh              # SLURM wrapper around run_pipeline.sh
    logs/                            # SLURM stdout/stderr land here
    requirements.txt
    README.md
```

The high-level workflow for the MI-Gate pipeline:

1. Prepare multimodal features (image and text) — `Features/`.
2. Train PID estimators on those features — `Estimator/mi_estimator.py`.
3. Save estimator checkpoints for reuse — `Estimator/saved_estimators/`.
4. Run `SFT/preprocess_mi_gate.py` to score every sample's R/U1/U2/S, build
   a train/val/test split, apply the MI Gate, and write the augmented text
   back into `data.json`.
5. Run `SFT/multimodal_interaction_tuning.py` to fine-tune SmolVLM2 on the gated dataset.

The Quick Start above only depends on (1) and (2). The other experiments add (3); the SFT pipeline adds (4) and (5).


## Other experiments

`Experiments/` contains analyses that build on the trained estimators
without rerunning the SFT pipeline. Each script is self-contained — see
[`Experiments/experiments_readme.md`](Experiments/experiments_readme.md)
for full docs, defaults, and example invocations.

- **`Experiments/proportion_of_interactions.py`** — reports the proportion
  of samples whose dominant interaction term is R, U1, U2, or S, both
  globally and per train/val/test split. Two passes: (1) score with the
  frozen estimator on baseline text → donut at
  `Experiments/outputs/proportion_of_interactions.png`; (2) re-encode every
  sample's text with the caption appended, save standard-schema features at
  `Features/features_all_augmented_siglip2_pca512.pt`, retrain the estimator
  and checkpoint it as
  `Estimator/saved_estimators/mi_estimators_all_augmented_state_dict.pt`,
  rescore, and produce a second donut at
  `Experiments/outputs/proportion_of_interactions_retrained.png`. Pass
  `--skip_retrain` to disable Part 2.

- **`Experiments/tau_ablations.py`** — replays the MI Gate at
  τ ∈ {0%, 10%, …, 100%} and tracks how the dominance distribution shifts
  as more U1-dominant samples receive the caption augmentation. Two
  passes: (1) score with the original frozen estimator, and (2) at every
  τ, materialise a fresh feature `.pt`
  (`Features/per_tau/features_tau{tau:.2f}_siglip2_pca512.pt`) with its
  own PCA fit, retrain the estimator on it from scratch, and rescore.
  Each retrained estimator is checkpointed under
  `Estimator/saved_estimators/mi_estimators_tau{tau:.2f}_state_dict.pt`.
  Outputs a 2 × 2 (or 1 × 2 with `--skip_retrain`) figure at
  `Experiments/outputs/tau_ablations.png` plus a JSON of per-τ records.

Both of these require the data layout described under
[Data folder structure](#data-folder-structure) — they re-encode raw
images and text. The Quick Start does not.



## License

This repository is released under the [MIT License](LICENSE). It adapts code
from the [LSMI Estimator](https://github.com/GeWu-Lab/LSMI_Estimator) (MIT)
and the [KNIFE entropy estimator](https://github.com/g-pichler/knife)
(BSD 3-Clause); both upstream licenses are retained in `LICENSE` and the
file-level attributions in `Estimator/`.


## Acknowledgements

This work builds directly on:

- **The Cauldron** dataset (HatefulMemes subset) —
  <https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>
- **LSMI Estimator** (multimodal interaction estimator) —
  <https://github.com/GeWu-Lab/LSMI_Estimator>
- **KNIFE** (the entropy estimator within the LSMI estimator) —
  <https://github.com/g-pichler/knife>
- **SmolVLM** — <https://github.com/huggingface/smollm/tree/main/vision>

The downstream hallucination / robustness evaluation in the paper is run on top of two existing benchmarks:

- **Adaptation-method robustness benchmark** (Chen et al., NeurIPS 2023) —
  Chen, S., Gu, J., Han, Z., Ma, Y., Torr, P., and Tresp, V.
  *Benchmarking Robustness of Adaptation Methods on Pre-trained
  Vision-Language Models.* Advances in Neural Information Processing
  Systems, vol. 36, pp. 51758–51777, 2023.
- **HallusionBench** (Guan et al., CVPR 2024) —
  Guan, T. et al. *HallusionBench: An Advanced Diagnostic Suite for
  Entangled Language Hallucination and Visual Illusion in Large
  Vision-Language Models.* Proceedings of the IEEE/CVF Conference on
  Computer Vision and Pattern Recognition, 2024, pp. 14375–14385.
  <https://openaccess.thecvf.com/content/CVPR2024/html/Guan_HallusionBench_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_2024_paper.html>

If you do use this work, please do cite them as well! 🙇🏻

## Citation
Note: To be updated when the officil PMLR issue is made public.

```
@inproceedings{ryan2026selfcaptioning,
    title     = {Self-Captioning Multimodal Interaction Tuning: Amplifying
                 Exploitable Redundancies for Robust Vision Language Models},
    author    = {Ryan, Yuriel and Ip, Hei Man and Kuek, Adriel and Liang,
                 Paul Pu and Lee, Roy Ka-Wei},
    booktitle = {Proceedings of the 43rd International Conference on Machine
                 Learning (ICML)},
    year      = {2026},
    eprint    = {2605.08145},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url       = {https://arxiv.org/abs/2605.08145}
}
```





---

# Further details

I had some leftover documentation and thought it might be useful for those who might be interested. Thus, I cleaned it up slightly and added additional details in this section in an attempt to be more "complete".

## What gets trained

#### Stage 1 — PID estimators (`Estimator/mi_estimator.py`)

1. Three discriminators
- modality 1 classifier
- modality 2 classifier
- joint modality classifier (early fusion: concatenation of modalities 1
  and 2)

2. Two entropy estimators (based on the
   [KNIFE Differentiable Entropy Estimator](https://github.com/g-pichler/knife/tree/master))
- modality 1 entropy model
- modality 2 entropy model

Pointwise mutual information is computed with a class-prior correction:

$$i(x;y) = \log p(y|x) - \log p(y)$$

By default $p(y)$ is the **uniform prior** $1/n_{\text{classes}}$ (matches
the legacy LSMI estimator and keeps numbers reproducible across datasets
with different label distributions). To use a non-uniform prior, set
`cfg.class_priors = [...]` in `Estimator/train.yaml` (or pass
`provided_priors=` directly to `_resolve_log_class_priors`).

#### Stage 2 — SFT with MI Gate (`SFT/multimodal_interaction_tuning.py`)

Fine-tunes `HuggingFaceTB/SmolVLM2-2.2B-Instruct` on the augmented
dataset produced by `preprocess_mi_gate.py`. The user prompt for sample
n is `mi_gate_text[n]`, which equals `original_text + " " + caption` for
samples selected by the MI Gate, and `original_text` otherwise. The
assistant target is `correct_answer`.

**LoRA by default**: training uses
[PEFT LoRA](https://github.com/huggingface/peft) on the language-model
projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`) with `r=16`, `lora_alpha=32`, `lora_dropout=0.05`,
`bias="none"`, `task_type="CAUSAL_LM"`. With these defaults only ≈21 M
of 2.27 B parameters are trainable (≈0.93 %), so a full sweep fits
comfortably on a single H200 with `bf16` and gradient checkpointing. To
switch to full fine-tuning, override `use_lora=false`.

### The MI Gate

Given a dataset $\mathcal{D} = \{(x_V, x_T, y)_n\}$ and an embedding
model $\mathcal{F}$:

1. Estimate per-sample interactions $r_n, u_{V,n}, u_{T,n}, s_n$ on the
   train split.
2. Identify the *valid* set
   $\mathcal{S}_{valid} = \{n : u_{V,n} = \max(r_n, u_{V,n}, u_{T,n}, s_n)\}$
   — samples where image-uniqueness is the dominant interaction term.
3. Pick $k = \lfloor \tau \cdot |\mathcal{S}_{valid}| \rfloor$ samples by
   descending $u_{V,n}$.
4. Concatenate a caption $c_n$ to the text for those samples; leave
   others unchanged.

$\tau$ is the **fraction of valid (U1-dominant) train samples that
receive the caption**, configurable in `SFT/preprocess_mi_gate.py`
(`--tau`). For example, with τ=0.25 and 114 U1-dominant train samples,
the gate captions floor(0.25 × 114) = 28 of them. τ is independent of
any per-sample `redundancy_level` field in `data.json`.

### Saved checkpoints

#### Estimator checkpoints

Saved under `Estimator/saved_estimators/` by default:

- `mi_estimators_latest_state_dict.pt` — state-dict bundle (always saved)
- `mi_estimators_latest_full_models.pt` — optional full-model bundle

Names are configurable in `Estimator/train.yaml` via
`estimator_save_dir`, `estimator_save_tag`, and `save_full_models`.

#### MI-Gate preprocessing artifacts

`SFT/preprocess_mi_gate.py` writes:

- `Features/hateful_memes_features_id_aligned.pt` — re-extracted features
  in `data.json` id order, after applying the legacy PCA fit so they sit
  in the same 512-dim space the trained estimators were trained on.
- New keys appended to each row of `data.json`:
    - `split`: one of `train`, `val`, `test`, `excluded`
    - `mi_r`, `mi_u1`, `mi_u2`, `mi_s`: per-sample PID terms
    - `mi_gate_in_subset`: bool — whether the sample was selected by the
      gate
    - `mi_gate_text`: the user prompt to feed during SFT
      (`original_text` or `original_text + " " + caption`)

### Estimator re-use API (batch loop)

For ad-hoc per-batch interaction scoring (separate from the MI Gate
pipeline), see `Estimator/estimate_batch.py`:

- Main callable: `estimate_rus_batch`
- Inputs:
    - dataloader batch with modality 1 and modality 2 (and labels)
    - already-loaded discriminator + entropy estimators
    - already-loaded SigLIP2 feature extractor stack
- Output:
    - `rus_pointwise`: `[N, 4]` with columns `[R, U1, U2, S]`
    - `rus_mean`: `[4]`
    - the modality features used for that batch

This module is callable-only — no estimator loading happens inside it;
pass loaded estimators in.

For library usage of the per-sample LSMI decomposition during estimator
evaluation, `Estimator/mi_estimator.LSMI_estimation` accepts
`return_per_sample=True`, returning the per-sample `(r, u1, u2, s)`
tensors aligned with the dataloader's iteration order.

### Feature extraction module

`Features/prepare_features.py` is designed for dataloader batches and
returns modality feature tensors:

- `extract_features_from_batch` supports
    - dict batch with keys `modal_1` (image) and `modal_2` (text)
    - tuple/list batch where the first two elements are image and text
      batches
- Optional PCA to 512 dimensions via the `pca_512` flag.

For the schema of the precomputed feature `.pt` files shipped under
`Features/`, see [`Features/README.md`](Features/README.md).
