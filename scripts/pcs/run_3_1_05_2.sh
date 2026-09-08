#!/usr/bin/env bash
# 3.1.05_2 reuses the completed K67 cache and trains a new conservative scorer.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
PILOT_CACHE=${PILOT_CACHE:-$EXP_ROOT/pcs_candidates_pilot_k67_3_1_05}
GTRS_PDM=${GTRS_PDM:-/home/hndx/zyc/training_code/V2/DiffusionDriveV2/navtrain_16384.pkl}
GTRS_VOCAB=${GTRS_VOCAB:-/home/hndx/zyc/training_code/V2/DiffusionDriveV2/gtrs_traj/16384.npy}
GTRS_CACHE=${GTRS_CACHE:-$EXP_ROOT/pcs_gtrs_stratified_k67_3_1_05_2}
PILOT_GTRS_CACHE=${PILOT_GTRS_CACHE:-$EXP_ROOT/pcs_gtrs_stratified_pilot_k67_3_1_05_2}
PCS_EXP=${PCS_EXP:-$EXP_ROOT/conservative_advantage_selector_k67_3_1_05_2}
LOG_ROOT=${LOG_ROOT:-$EXP_ROOT/logs/3.1.05_2}
INIT_SCORER_CKPT=${INIT_SCORER_CKPT:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}

export NAVSIM_DEVKIT_ROOT="$CODE_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NCCL_DEBUG=WARN
cd "$CODE_ROOT"
mkdir -p "$LOG_ROOT"
MODE=${1:-help}

run_logged() {
    local name=$1
    shift
    "$@" 2>&1 | tee "$LOG_ROOT/${name}_$(date +%Y%m%d_%H%M%S)_$$.log"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing regular file (a broken/self-referential symlink also fails): $1" >&2
        exit 1
    fi
}

generator_args=(
    --baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH"
    --data-root "$DATA_ROOT" --seed 0
)

case "$MODE" in
    test)
        run_logged pcs_3_1_05_2_tests \
            python -m unittest discover -s tests -p 'test_pcs*.py'
        ;;
    prepare-gtrs-pilot)
        require_file "$GTRS_PDM"
        require_file "$GTRS_VOCAB"
        run_logged pcs_3_1_05_2_prepare_gtrs_pilot \
            python -m navsim.planning.script.run_pcs_conservative prepare-gtrs \
            --cache "$PILOT_CACHE" --gtrs-pdm "$GTRS_PDM" \
            --gtrs-vocabulary "$GTRS_VOCAB" --output "$PILOT_GTRS_CACHE" \
            --samples "${GTRS_SAMPLES:-32}" --seed 0
        ;;
    smoke)
        require_file "$INIT_SCORER_CKPT"
        run_logged pcs_3_1_05_2_smoke env CUDA_VISIBLE_DEVICES=0 \
            python -m navsim.planning.script.run_pcs_conservative train \
            --cache "$PILOT_CACHE" --gtrs-cache "$PILOT_GTRS_CACHE" \
            --output "$PCS_EXP/smoke" --devices 1 --batch-size 2 \
            --epochs 1 --workers 2 --precision 16-mixed --smoke \
            --init-scorer "$INIT_SCORER_CKPT"
        ;;
    prepare-gtrs)
        require_file "$GTRS_PDM"
        require_file "$GTRS_VOCAB"
        run_logged pcs_3_1_05_2_prepare_gtrs \
            python -m navsim.planning.script.run_pcs_conservative prepare-gtrs \
            --cache "$PCS_CACHE" --gtrs-pdm "$GTRS_PDM" \
            --gtrs-vocabulary "$GTRS_VOCAB" --output "$GTRS_CACHE" \
            --samples "${GTRS_SAMPLES:-32}" --seed 0
        ;;
    train)
        # Physical GPUs 0-3, BS32 per GPU, global BS128.
        require_file "$INIT_SCORER_CKPT"
        run_logged pcs_3_1_05_2_train env CUDA_VISIBLE_DEVICES=0,1,2,3 \
            python -m navsim.planning.script.run_pcs_conservative train \
            --cache "$PCS_CACHE" --gtrs-cache "$GTRS_CACHE" \
            --output "$PCS_EXP/train" --devices 4 --batch-size 32 \
            --epochs 20 --workers 8 --precision 16-mixed --lr 3e-4 \
            --init-scorer "$INIT_SCORER_CKPT"
        ;;
    eval|eval-smoke)
        : "${SCORER_CKPT:?Set SCORER_CKPT to the best 3.1.05_2 checkpoint}"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        if [ -n "${MIN_DELTA:-}" ]; then extra+=(--min-delta "$MIN_DELTA"); fi
        if [ -n "${MIN_WIN_PROBABILITY:-}" ]; then
            extra+=(--min-win-probability "$MIN_WIN_PROBABILITY")
        fi
        if [ -n "${MAX_CATASTROPHIC_RISK:-}" ]; then
            extra+=(--max-catastrophic-risk "$MAX_CATASTROPHIC_RISK")
        fi
        if [ -n "${RISK_PENALTY:-}" ]; then extra+=(--risk-penalty "$RISK_PENALTY"); fi
        run_logged "pcs_3_1_05_2_$MODE" env CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" \
            python -m navsim.planning.script.run_pcs_conservative evaluate \
            "${generator_args[@]}" --scorer "$SCORER_CKPT" \
            --metric-cache "$TEST_METRIC_CACHE" --output "$PCS_EXP/$MODE" \
            --workers 4 --score-workers 4 "${extra[@]}"
        ;;
    *)
        echo "Usage: bash scripts/pcs/run_3_1_05_2.sh {test|prepare-gtrs-pilot|smoke|prepare-gtrs|train|eval-smoke|eval}"
        echo "Recommended: test -> prepare-gtrs-pilot -> smoke -> prepare-gtrs -> train -> eval-smoke -> eval"
        echo "The completed K67 candidate cache is reused; do not rebuild it for this experiment."
        ;;
esac
