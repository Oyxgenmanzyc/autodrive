#!/usr/bin/env bash
# 3.1.05_2 TRV reuses the K67 cache and the proven epoch5 PCS proposer.
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
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
PAIR_CACHE=${PAIR_CACHE:-$EXP_ROOT/pcs_hard_negative_pairs_k67_3_1_05_2}
PILOT_PAIR_CACHE=${PILOT_PAIR_CACHE:-$EXP_ROOT/pcs_hard_negative_pairs_pilot_k67_3_1_05_2}
TRV_EXP=${TRV_EXP:-$EXP_ROOT/triple_risk_veto_k67_3_1_05_2}
LOG_ROOT=${LOG_ROOT:-$EXP_ROOT/logs/3.1.05_2_veto}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04,GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f,GPU-e9db5634-a044-a8fb-c294-6f372876e0e3,GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd}
EVAL_GPU=${EVAL_GPU:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04}

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
    test -f "$1" || { echo "Missing file: $1" >&2; exit 1; }
}

generator_args=(
    --baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH"
    --data-root "$DATA_ROOT" --seed 0
)

case "$MODE" in
    test)
        run_logged trv_tests python -m unittest discover -s tests -p 'test_pcs*.py'
        ;;
    prepare-pairs-pilot)
        require_file "$PCS_SCORER"
        run_logged trv_prepare_pairs_pilot env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            python -m navsim.planning.script.run_pcs_veto prepare-pairs \
            --cache "$PILOT_CACHE" --pcs-scorer "$PCS_SCORER" \
            --output "$PILOT_PAIR_CACHE" --workers 4 --batch-size 32 --smoke
        ;;
    smoke)
        require_file "$PCS_SCORER"
        run_logged trv_smoke env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            python -m navsim.planning.script.run_pcs_veto train \
            --cache "$PILOT_CACHE" --pair-cache "$PILOT_PAIR_CACHE" \
            --pcs-scorer "$PCS_SCORER" --output "$TRV_EXP/smoke" \
            --devices 1 --batch-size 2 --epochs 1 --workers 2 \
            --precision 16-mixed --lr 3e-4 --smoke
        ;;
    prepare-pairs)
        require_file "$PCS_SCORER"
        run_logged trv_prepare_pairs env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            python -m navsim.planning.script.run_pcs_veto prepare-pairs \
            --cache "$PCS_CACHE" --pcs-scorer "$PCS_SCORER" \
            --output "$PAIR_CACHE" --workers 8 --batch-size 32
        ;;
    train)
        require_file "$PCS_SCORER"
        run_logged trv_train env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
            python -m navsim.planning.script.run_pcs_veto train \
            --cache "$PCS_CACHE" --pair-cache "$PAIR_CACHE" \
            --pcs-scorer "$PCS_SCORER" --output "$TRV_EXP/train" \
            --devices 4 --batch-size 32 --epochs 10 --workers 8 \
            --precision 16-mixed --lr 3e-4
        ;;
    eval|eval-smoke)
        : "${VETO_CKPT:?Set VETO_CKPT to the best TRV checkpoint}"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        run_logged "trv_$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            python -m navsim.planning.script.run_pcs_veto evaluate \
            "${generator_args[@]}" --pcs-scorer "$PCS_SCORER" --veto "$VETO_CKPT" \
            --metric-cache "$TEST_METRIC_CACHE" --output "$TRV_EXP/$MODE" \
            --workers 4 --score-workers 4 "${extra[@]}"
        ;;
    *)
        echo "Usage: bash scripts/pcs/run_3_1_05_2_veto.sh {test|prepare-pairs-pilot|smoke|prepare-pairs|train|eval-smoke|eval}"
        echo "Recommended: test -> prepare-pairs-pilot -> smoke -> prepare-pairs -> train -> eval-smoke -> eval"
        echo "No feature, metric, K67 candidate, or GTRS cache is rebuilt."
        ;;
esac
