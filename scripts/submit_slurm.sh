#!/usr/bin/env bash
#SBATCH --job-name=mi-gate-sft
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# SLURM wrapper around scripts/run_pipeline.sh.
#
# Submit:
#     sbatch scripts/submit_slurm.sh
#
# Override pipeline behavior via the same env vars as run_pipeline.sh:
#     sbatch --export=ALL,TAU=0.5 scripts/submit_slurm.sh
#     sbatch --export=ALL,SKIP_DOWNLOAD=1,SKIP_CAPTION=1 scripts/submit_slurm.sh
#
# Adjust SBATCH directives above for your cluster (partition / qos / gres
# format vary). The job assumes one H200 (~140GB); SmolVLM2-2.2B-Instruct in
# bf16 with grad checkpointing fits comfortably.
set -euo pipefail

# Resolve the repo root from this script's location, robust to either
# `sbatch scripts/submit_slurm.sh` or `bash scripts/submit_slurm.sh`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# ---------------- Environment ----------------
# Activate the conda env you set up in step (2) of the README.
CONDA_ENV="${CONDA_ENV:-MIT}"
if [[ -n "${CONDA_PREFIX_BASE:-}" ]] && [[ -f "${CONDA_PREFIX_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_PREFIX_BASE}/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV"
fi

# Helpful runtime diagnostics
echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID:-no-job} time=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Cache locations (override on the cluster if your scratch is elsewhere)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---------------- Run pipeline ----------------
bash scripts/run_pipeline.sh
