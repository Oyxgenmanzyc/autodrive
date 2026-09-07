#!/usr/bin/env bash
# Run from the new 3.1.05 checkout. All modes use navhigh; no git pull required.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
FEATURE_CACHE=${FEATURE_CACHE:-$EXP_ROOT/training_cache_3_1_02_k67}
TRAIN_METRIC_CACHE=${TRAIN_METRIC_CACHE:-$EXP_ROOT/metric_cache_navtrain_pcs_3_1_05}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
PILOT_CACHE=${PILOT_CACHE:-$EXP_ROOT/pcs_candidates_pilot_k67_3_1_05}
PCS_EXP=${PCS_EXP:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05}
LOG_ROOT=${LOG_ROOT:-$EXP_ROOT/logs/3.1.05_pcs}

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

generator_args=(
    --baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH"
    --data-root "$DATA_ROOT" --seed 0
)

case "$MODE" in
    test)
        run_logged pcs_unit_tests python -m unittest discover -s tests -p test_pcs.py
        ;;
    metric)
        metric_args=(
            train_test_split=navtrain
            worker=ray_distributed_no_torch worker.threads_per_node=64
            cache.cache_path="$TRAIN_METRIC_CACHE"
            cache.force_feature_computation=false
        )
        python navsim/planning/script/run_metric_caching.py --cfg job --resolve "${metric_args[@]}"             > "$LOG_ROOT/navtrain_metric_resolved.yaml"
        run_logged pcs_navtrain_metric python navsim/planning/script/run_metric_caching.py "${metric_args[@]}"
        ;;
    pilot)
        run_logged pcs_pilot env CUDA_VISIBLE_DEVICES=0 python -m navsim.planning.script.run_pcs cache             "${generator_args[@]}" --split navtrain --feature-cache "$FEATURE_CACHE"             --metric-cache "$TRAIN_METRIC_CACHE" --output "$PILOT_CACHE"             --max-scenes "${PILOT_SCENES:-128}" --workers 4 --score-workers 4
        run_logged pcs_pilot_diagnosis python -m navsim.planning.script.run_pcs diagnose --cache "$PILOT_CACHE"
        ;;
    smoke)
        # Explicitly allow the small pilot cache. This is never a final experiment.
        run_logged pcs_smoke env CUDA_VISIBLE_DEVICES=0 python -m navsim.planning.script.run_pcs train             --cache "$PILOT_CACHE" --output "$PCS_EXP/smoke"             --devices 1 --batch-size 2 --epochs 1 --workers 2 --smoke
        ;;
    cache)
        # Four physical GPUs, one independent cache shard per GPU.
        pids=()
        for gpu in 0 1 2 3; do
            run_logged "pcs_cache_gpu$gpu" env CUDA_VISIBLE_DEVICES="$gpu"                 python -m navsim.planning.script.run_pcs cache "${generator_args[@]}"                 --split navtrain --feature-cache "$FEATURE_CACHE" --metric-cache "$TRAIN_METRIC_CACHE"                 --output "$PCS_CACHE" --num-shards 4 --shard-index "$gpu"                 --workers 2 --score-workers 4 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then failed=1; fi
        done
        if [ "$failed" -ne 0 ]; then
            echo "A cache shard failed. Check its log; rerun this mode to resume completed entries."
            exit 1
        fi
        run_logged pcs_diagnosis python -m navsim.planning.script.run_pcs diagnose --cache "$PCS_CACHE"
        ;;
    train)
        # 32 per physical GPU, 128 global. Only scorer parameters are trained.
        run_logged pcs_train env CUDA_VISIBLE_DEVICES=0,1,2,3             python -m navsim.planning.script.run_pcs train             --cache "$PCS_CACHE" --output "$PCS_EXP/train"             --devices 4 --batch-size 32 --epochs 20 --workers 8 --precision 16-mixed --lr 3e-4
        ;;
    eval|eval-smoke)
        : "${SCORER_CKPT:?Set SCORER_CKPT to the BEST scorer checkpoint printed by training}"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        run_logged "pcs_$MODE" env CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"             python -m navsim.planning.script.run_pcs evaluate "${generator_args[@]}"             --scorer "$SCORER_CKPT" --metric-cache "$TEST_METRIC_CACHE"             --output "$PCS_EXP/$MODE" --workers 4 --score-workers 4 "${extra[@]}"
        ;;
    *)
        echo "Usage: bash scripts/pcs/run_3_1_05.sh {test|metric|pilot|smoke|cache|train|eval-smoke|eval}"
        echo "Recommended: test -> metric -> pilot -> smoke -> review diagnostic -> cache -> train -> eval-smoke -> eval"
        ;;
esac

