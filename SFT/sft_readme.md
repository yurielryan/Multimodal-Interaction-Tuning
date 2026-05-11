# Full SFT pipeline

This is a replica of the end-to-end paper pipeline: download the HatefulMemes images,
caption them with SmolVLM2, score per-sample interactions, apply the MI
Gate, and fine-tune SmolVLM2 on the gated dataset. 

This captioning process was scaled up (without running the estimators) to the entire finetuning dataset for the **general setting** in the paper.
Instead of scaling $\tau$, we caption 25\%, 50\% of the entire dataset and compare it with the default (baseline 0\% additional redundancy) dataset. See paper for full details.

**Note**: Admittedly, this pipeline is a more portable and downscaled version of the SFT in the original work. I have tested it and the pipeline works; however, the full fine-tuning process includes a cleaned version of the
[Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) SFT dataset. Due to size constraints, I am unable to include the data here. Kindly drop me (Yuriel Ryan) an [email](mailto:yurielryan@gmail.com) if you would like to have a copy of the SFT dataset.


### One-command pipeline
`scripts/run_pipeline.sh` chains the four data-and-training stages below.
Each stage is idempotent and individually skippable.

```
# Full run with defaults (tau = 0.25)
bash scripts/run_pipeline.sh

# Custom tau
TAU=0.5 bash scripts/run_pipeline.sh

# Skip specific stages
SKIP_DOWNLOAD=1 SKIP_CAPTION=1 bash scripts/run_pipeline.sh

# Run only a subset of stages
STAGES="preprocess train" bash scripts/run_pipeline.sh
```

Environment variables understood by the script:

| Var | Default | Meaning |
| --- | --- | --- |
| `DATA_PATH` | `<repo>/data/data.json` | Per-sample annotation file |
| `IMAGE_DIR` | `<repo>/data/images` | Where `{id}.png` files live |
| `TAU` | `0.25` | Fraction of *valid (U1-dominant)* train samples to caption-augment |
| `OUTPUT_DIR` | `SFT/runs/mi_gate_tau<TAU>` | Where the SFT checkpoint + eval metrics land |
| `SFT_CONFIG` | `SFT/config.yaml` | YAML config for the SFT trainer |
| `STAGES` | `download caption preprocess train` | Subset of stages to run |
| `SKIP_<STAGE>` | unset | Skip a single stage (`SKIP_DOWNLOAD=1`, …) |
| `CAPTION_BATCH_SIZE` | `8` | Batch size for SmolVLM2 captioning |
| `CAPTION_MAX_NEW_TOKENS` | `96` | Max tokens per caption |
| `PREPROCESS_EXTRA` | unset | Extra args to forward to `preprocess_mi_gate.py` |
| `TRAIN_EXTRA` | unset | Extra `key=value` overrides forwarded to the SFT script |

### Submit on SLURM

```
sbatch scripts/submit_slurm.sh
sbatch --export=ALL,TAU=0.5 scripts/submit_slurm.sh
sbatch --export=ALL,SKIP_DOWNLOAD=1,SKIP_CAPTION=1 scripts/submit_slurm.sh
```

The wrapper requests one H200, 24 h, 96 GB RAM, 8 CPUs, activates the
`MIT` conda env (override with `CONDA_ENV=…`), and runs `run_pipeline.sh`.
Adjust the `#SBATCH` directives at the top of `scripts/submit_slurm.sh`
for your cluster's partition / QoS conventions. Logs land in `logs/`.

**Note:** The original work was done on 1-4 H100 GPUs. Since then, I had more access to GPUs; hence, the slurm script you see here uses H200s. Do adjust this script accordingly to your hardware constraints. 

### End-to-end pipeline (manual)

#### 1) Train the PID estimators (one time)

Edit `Estimator/train.yaml` (`device`, `data_path`, `n_classes`, etc.)
and run from the repo root:

```
python Estimator/mi_estimator.py
```

This produces the checkpoints under `Estimator/saved_estimators/`.

#### 2) Download images for HatefulMemes (one time, if needed)

`Features/download_images.py` pulls images from
`HuggingFaceM4/the_cauldron`, `hateful_memes` config, indexed positionally
so Cauldron row N maps to `{N}.png` (verified byte-identical against
pre-existing local images).

```
python Features/download_images.py        # → <repo>/data/images by default
# or:
python Features/download_images.py --out_dir /scratch/HatefulMemes/images
```

#### 3) Generate captions (one time, only if `generated_caption_smolvlm` missing)

```
python SFT/caption.py                # uses <repo>/data/data.json by default
# or with custom paths:
python SFT/caption.py \
    --data_path /scratch/HatefulMemes/data.json \
    --image_dir /scratch/HatefulMemes/images \
    --key generated_caption_smolvlm
```

The script is idempotent: rows that already have the key populated are
skipped unless `--overwrite` is set.

#### 4) Run the MI Gate preprocessing

```
python SFT/preprocess_mi_gate.py --tau 0.25
```

Important flags:

- `--tau`: fraction of *valid (U1-dominant)* train samples to
  caption-augment.
- `--text_field`: which `data.json` field is fed to the SigLIP text
  encoder during scoring. Defaults to `original_text` (the baseline text —
  what the model would see without augmentation).
- `--skip_feature_extraction`: reuse a previously written id-aligned
  features file.

Outputs are written in place to `data.json` plus the id-aligned features
`.pt`.

#### 5) Fine-tune SmolVLM2 on the gated dataset

```
python SFT/multimodal_interaction_tuning.py --config SFT/config.yaml
```

CLI overrides take the form `key=value` after the `--config` argument:

```
python SFT/multimodal_interaction_tuning.py --config SFT/config.yaml \
    text_field=original_text \
    output_dir=SFT/runs/baseline_no_caption
```

The script trains with `transformers.Trainer` (bf16, gradient
checkpointing, right-padded chat template), saves the final model +
processor under `output_dir`, and runs a generation-based accuracy eval
on the val split — broken down by `mi_gate_in_subset` so the contribution
of the gate is observable.


## Data folder structure

By default, the bundled scripts expect data at the repo root:

```
<repo-root>/
    data/
        data.json
        images/
            {id}.png
```

- `data.json` stores per-sample metadata/annotations. The `id` field
  links each row to its corresponding `{id}.png` under `images/`.
- All bundled scripts default to `data/data.json` and `data/images`
  resolved relative to the repo root, so you can run them from any
  working directory.

Minimum required fields per `data.json` row:

- `id`: unique sample identifier (also the image filename stem)
- `original_text`: the question / prompt
- `correct_answer`: gold label

Fields populated by the SFT pipeline (you do not need to provide these):

- `generated_caption_smolvlm`: caption produced by `SFT/caption.py`
- `split`, `mi_r`, `mi_u1`, `mi_u2`, `mi_s`, `mi_gate_in_subset`,
  `mi_gate_text`: written by `SFT/preprocess_mi_gate.py`

### Putting the data somewhere else

If you want to keep the data outside the repo (e.g., on faster storage),
override the paths — every entrypoint takes them as flags or env vars
and the `run_pipeline.sh` runner takes them as env vars:

```
# 1. Per-script CLI flags
python Features/download_images.py        --out_dir   /scratch/HatefulMemes/images
python SFT/caption.py                --data_path /scratch/HatefulMemes/data.json \
                                     --image_dir /scratch/HatefulMemes/images
python SFT/preprocess_mi_gate.py     --data_path /scratch/HatefulMemes/data.json \
                                     --image_dir /scratch/HatefulMemes/images
python SFT/multimodal_interaction_tuning.py --config SFT/config.yaml \
       data_path=/scratch/HatefulMemes/data.json \
       image_dir=/scratch/HatefulMemes/images

# 2. Env vars (consumed by scripts/run_pipeline.sh and submit_slurm.sh)
DATA_PATH=/scratch/HatefulMemes/data.json \
IMAGE_DIR=/scratch/HatefulMemes/images \
    bash scripts/run_pipeline.sh

sbatch --export=ALL,DATA_PATH=/scratch/HatefulMemes/data.json,IMAGE_DIR=/scratch/HatefulMemes/images \
    scripts/submit_slurm.sh

# 3. Edit SFT/config.yaml's data_path / image_dir fields directly.
#    Relative paths in the YAML are resolved against the repo root.
```

A single `data/` symlink also works:

```
ln -s /scratch/HatefulMemes data
```
