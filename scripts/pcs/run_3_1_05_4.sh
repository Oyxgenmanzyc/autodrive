#!/usr/bin/env bash
# Post-selection timing refiner. Invoke with bash, do not source.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
PTR_EXP=${PTR_EXP:-$EXP_ROOT/post_selection_timing_3_1_05_4}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
EDIT_CACHE=${EDIT_CACHE:-$PTR_EXP/edit_cache/navtrain}
PILOT_EDIT_CACHE=${PILOT_EDIT_CACHE:-$PTR_EXP/edit_cache/pilot}
TRAIN_METRIC_CACHE=${TRAIN_METRIC_CACHE:-$EXP_ROOT/metric_cache_navtrain_pcs_3_1_05}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
VETO_CKPT=${VETO_CKPT:-$EXP_ROOT/triple_risk_veto_k67_3_1_05_2/train/2026.09.09.14.53.12.850967/checkpoints/epoch=02.ckpt}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04,GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f,GPU-e9db5634-a044-a8fb-c294-6f372876e0e3,GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd}
EVAL_GPU=${EVAL_GPU:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04}
export NAVSIM_DEVKIT_ROOT="$CODE_ROOT" OPENSCENE_DATA_ROOT="$DATA_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PTR_RUN_DIR
cd "$CODE_ROOT"
mkdir -p "$PTR_EXP/logs"
MODE=${1:-help}
run_logged() {
    local label=$1
    shift
    "$@" 2>&1 | tee "$PTR_EXP/logs/${label}_$(date +%Y%m%d_%H%M%S)_$$.log"
}
case "$MODE" in
    check)
        for file in "$PCS_CACHE/manifest.json" "$PCS_CACHE/records.json" "$BASELINE_CKPT" "$BKB_PATH" "$ANCHOR_PATH" "$PCS_SCORER" "$VETO_CKPT"; do
            test -f "$file" || { echo "Missing: $file"; exit 1; }
            ls -lh "$file"
        done
        for directory in "$TRAIN_METRIC_CACHE" "$TEST_METRIC_CACHE"; do
            test -d "$directory" || { echo "Missing metric cache: $directory; restore/build the corresponding split"; exit 1; }
        done
        nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
        df -h "$EXP_ROOT"
        ;;
    test)
        run_logged test python -m unittest discover -s tests -p 'test_post_selection.py'
        ;;
    prepare-pilot|prepare|prepare-shard)
        output=$EDIT_CACHE; limit=0
        if [ "$MODE" = prepare-pilot ]; then output=$PILOT_EDIT_CACHE; limit=${PILOT_SCENES:-128}; fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m navsim.planning.script.run_post_selection prepare \
            --candidate-cache "$PCS_CACHE" --veto "$VETO_CKPT" --pcs-scorer "$PCS_SCORER" \
            --metric-cache "$TRAIN_METRIC_CACHE" --output "$output" --limit "$limit" \
            --score-workers "${SCORE_WORKERS:-4}" --num-shards "${NUM_SHARDS:-1}" --shard-index "${SHARD_INDEX:-0}"
        if [ "${NUM_SHARDS:-1}" -eq 1 ]; then
            run_logged oracle python -m navsim.planning.script.run_post_selection diagnose --edit-cache "$output"
        fi
        ;;
    diagnose)
        run_logged oracle python -m navsim.planning.script.run_post_selection diagnose --edit-cache "$EDIT_CACHE"
        ;;
    smoke|smoke-ddp|train)
        devices=4; batch=${BATCH_SIZE:-32}; epochs=${EPOCHS:-20}; gpu=$TRAIN_GPUS; cache=$EDIT_CACHE; output=$PTR_EXP/train; extra=()
        if [ "$MODE" != train ]; then
            devices=1; batch=2; epochs=1; gpu=$EVAL_GPU; cache=$PILOT_EDIT_CACHE; output=$PTR_EXP/smoke; extra+=(--smoke)
            if [ "$MODE" = smoke-ddp ]; then devices=4; gpu=$TRAIN_GPUS; fi
        fi
        if [ -n "${RESUME_CKPT:-}" ]; then extra+=(--resume "$RESUME_CKPT"); fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$gpu" python -m navsim.planning.script.run_post_selection train \
            --candidate-cache "$PCS_CACHE" --edit-cache "$cache" --output "$output" \
            --devices "$devices" --batch-size "$batch" --epochs "$epochs" --workers "${WORKERS:-0}" --lr "${LR:-1e-4}" "${extra[@]}"
        ;;
    eval-smoke|eval)
        : "${REFINER_CKPT:?Set REFINER_CKPT to a trained PTR checkpoint}"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m navsim.planning.script.run_post_selection evaluate \
            --baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH" --pcs-scorer "$PCS_SCORER" \
            --veto "$VETO_CKPT" --refiner "$REFINER_CKPT" --metric-cache "$TEST_METRIC_CACHE" \
            --data-root "$DATA_ROOT" --output "$PTR_EXP/$MODE" --workers 0 --score-workers "${SCORE_WORKERS:-4}" "${extra[@]}"
        ;;
    *)
        echo "Usage: bash $0 {check|test|prepare-pilot|smoke|smoke-ddp|prepare|prepare-shard|diagnose|train|eval-smoke|eval}"
        echo 'No original feature cache, GTRS cache or history sidecar required. Keep PCS and both metric caches.'
        ;;
esac
