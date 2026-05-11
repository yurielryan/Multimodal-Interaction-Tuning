#!/usr/bin/env bash
# End-to-end pipeline runner for the Multimodal Interaction Tuning repo.
#
# Stages (each can be skipped via SKIP_<STAGE>=1):
#   1. download : pull HatefulMemes images from Cauldron into IMAGE_DIR
#   2. caption  : populate generated_caption_smolvlm in DATA_PATH (idempotent)
#   3. preprocess : compute per-sample MI, apply MI Gate, write augmented data.json
#   4. train    : SFT SmolVLM2 on the augmented dataset
#
# Configure via environment variables; sensible defaults are below.
#
# Usage:
#   bash scripts/run_pipeline.sh                                  # full pipeline
#   TAU=0.5 bash scripts/run_pipeline.sh                          # custom tau
#   SKIP_DOWNLOAD=1 SKIP_CAPTION=1 bash scripts/run_pipeline.sh   # skip stages
#   STAGES="preprocess train" bash scripts/run_pipeline.sh        # only some stages
set -euo pipefail

# ---------------- Defaults ----------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/data.json}"
IMAGE_DIR="${IMAGE_DIR:-${REPO_ROOT}/data/images}"
TAU="${TAU:-0.25}"
SFT_CONFIG="${SFT_CONFIG:-${REPO_ROOT}/SFT/config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/SFT/runs/mi_gate_tau${TAU}}"
ESTIMATOR_CKPT="${ESTIMATOR_CKPT:-${REPO_ROOT}/Estimator/saved_estimators/mi_estimators_latest_state_dict.pt}"
LEGACY_FEATURES="${LEGACY_FEATURES:-${REPO_ROOT}/Features/hateful_memes_features_siglip2_pca512.pt}"
ID_ALIGNED_FEATURES="${ID_ALIGNED_FEATURES:-${REPO_ROOT}/Features/hateful_memes_features_id_aligned.pt}"

# Stages selector: either run all unless SKIP_X=1, or restrict via STAGES="a b c"
ALL_STAGES=(download caption preprocess train)
STAGES_RUN="${STAGES:-${ALL_STAGES[*]}}"

is_in() {
    local needle="$1"; shift
    for s in "$@"; do [[ "$s" == "$needle" ]] && return 0; done
    return 1
}

run_stage() {
    local name="$1"; shift
    local skip_var="SKIP_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
    if ! is_in "$name" $STAGES_RUN; then
        echo "[skip] $name (not in STAGES)"
        return 0
    fi
    if [[ "${!skip_var:-0}" == "1" ]]; then
        echo "[skip] $name ($skip_var=1)"
        return 0
    fi
    echo
    echo "================================================================"
    echo "[stage] $name"
    echo "================================================================"
    "$@"
}

# ---------------- Stage implementations ----------------
stage_download() {
    python "${REPO_ROOT}/Features/download_images.py" \
        --out_dir "$IMAGE_DIR"
}

stage_caption() {
    python "${REPO_ROOT}/SFT/caption.py" \
        --data_path "$DATA_PATH" \
        --image_dir "$IMAGE_DIR" \
        --batch_size "${CAPTION_BATCH_SIZE:-8}" \
        --max_new_tokens "${CAPTION_MAX_NEW_TOKENS:-96}"
}

stage_preprocess() {
    python "${REPO_ROOT}/SFT/preprocess_mi_gate.py" \
        --data_path "$DATA_PATH" \
        --image_dir "$IMAGE_DIR" \
        --legacy_features "$LEGACY_FEATURES" \
        --features_out "$ID_ALIGNED_FEATURES" \
        --estimator_ckpt "$ESTIMATOR_CKPT" \
        --tau "$TAU" \
        ${PREPROCESS_EXTRA:-}
}

stage_train() {
    python "${REPO_ROOT}/SFT/multimodal_interaction_tuning.py" \
        --config "$SFT_CONFIG" \
        data_path="$DATA_PATH" \
        image_dir="$IMAGE_DIR" \
        output_dir="$OUTPUT_DIR" \
        tau="$TAU" \
        ${TRAIN_EXTRA:-}
}

# ---------------- Main ----------------
echo "[setup] REPO_ROOT=$REPO_ROOT"
echo "[setup] DATA_PATH=$DATA_PATH"
echo "[setup] IMAGE_DIR=$IMAGE_DIR"
echo "[setup] TAU=$TAU"
echo "[setup] OUTPUT_DIR=$OUTPUT_DIR"
echo "[setup] STAGES=$STAGES_RUN"

cd "$REPO_ROOT"

run_stage download   stage_download
run_stage caption    stage_caption
run_stage preprocess stage_preprocess
run_stage train      stage_train

echo
echo "[done] pipeline complete. output in: $OUTPUT_DIR"
