#!/usr/bin/env bash
# Generate better trajectories; PCS/TRV remain fixed. No Hydra overrides.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
FEATURE_CACHE=${FEATURE_CACHE:-$EXP_ROOT/training_cache_3_1_02_k67}
RECORDS=${RECORDS:-$EXP_ROOT/pcs_candidates_k67_3_1_05/records.json}
TIMING_TARGETS=${TIMING_TARGETS:-$EXP_ROOT/generator_brake_timing_3_1_05_3/targets/navtrain.pt}
PILOT_TARGETS=${PILOT_TARGETS:-$EXP_ROOT/generator_brake_timing_3_1_05_3/targets/pilot.pt}
GEN_EXP=${GEN_EXP:-$EXP_ROOT/generator_brake_timing_3_1_05_3}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
VETO_CKPT=${VETO_CKPT:-$EXP_ROOT/triple_risk_veto_k67_3_1_05_2/train/2026.09.09.14.53.12.850967/checkpoints/epoch=02.ckpt}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04,GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f,GPU-e9db5634-a044-a8fb-c294-6f372876e0e3,GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd}
EVAL_GPU=${EVAL_GPU:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04}
export NAVSIM_DEVKIT_ROOT="$CODE_ROOT" OPENSCENE_DATA_ROOT="$DATA_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$CODE_ROOT"
mkdir -p "$GEN_EXP/logs"
MODE=${1:-help}
run_logged() {
    local name=$1
    shift
    "$@" 2>&1 | tee "$GEN_EXP/logs/${name}_$(date +%Y%m%d_%H%M%S)_$$.log"
}
common=(--baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH"
        --pcs-scorer "$PCS_SCORER" --veto "$VETO_CKPT" --seed 0)
case "$MODE" in
    check)
        for file in "$BASELINE_CKPT" "$BKB_PATH" "$ANCHOR_PATH" "$PCS_SCORER" "$VETO_CKPT" "$RECORDS"; do
            test -f "$file" || { echo "Missing: $file"; exit 1; }
            ls -lh "$file"
        done
        test -d "$FEATURE_CACHE" || { echo "Missing raw feature cache: $FEATURE_CACHE. Restore it or run prepare-features; detached PCS caches cannot train the generator."; exit 1; }
        nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu --format=csv,noheader
        env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -c 'import torch; assert torch.cuda.device_count() == 4; print([(i, torch.cuda.get_device_name(i)) for i in range(4)])'
        df -h "$EXP_ROOT"
        ;;
    test)
        run_logged tests python -m unittest discover -s tests -p 'test_generator_timing*.py'
        ;;
    prepare-features)
        run_logged features python -m navsim.planning.script.prepare_generator_features \
            --records "$RECORDS" --feature-cache "$FEATURE_CACHE" --data-root "$DATA_ROOT"
        ;;
    prepare-pilot|prepare)
        target=$TIMING_TARGETS
        extra=()
        if [ "$MODE" = prepare-pilot ]; then target=$PILOT_TARGETS; extra+=(--limit 128); fi
        run_logged "$MODE" python -m navsim.planning.script.run_generator_timing prepare \
            --records "$RECORDS" --feature-cache "$FEATURE_CACHE" --data-root "$DATA_ROOT" \
            --output "$target" "${extra[@]}"
        ;;
    smoke|smoke-ddp|train-control|train-timing)
        arm=timing
        if [ "$MODE" = train-control ]; then arm=control; fi
        devices=4; batch=${BATCH_SIZE:-32}; epochs=${EPOCHS:-10}; gpus=$TRAIN_GPUS
        target=$TIMING_TARGETS; output=$GEN_EXP/train; extra=()
        if [ "$MODE" = smoke ] || [ "$MODE" = smoke-ddp ]; then
            devices=1; batch=2; epochs=1; gpus=$EVAL_GPU; target=$PILOT_TARGETS; output=$GEN_EXP/smoke
            extra+=(--smoke)
            if [ "$MODE" = smoke-ddp ]; then devices=4; gpus=$TRAIN_GPUS; fi
        fi
        if [ -n "${RESUME_CKPT:-}" ]; then extra+=(--resume "$RESUME_CKPT"); fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$gpus" \
            python -m navsim.planning.script.run_generator_timing train "${common[@]}" \
            --feature-cache "$FEATURE_CACHE" --targets "$target" --arm "$arm" \
            --output "$output" --devices "$devices" --batch-size "$batch" --epochs "$epochs" \
            --workers "${WORKERS:-0}" --precision "${PRECISION:-16-mixed}" --lr "${LR:-2e-5}" \
            --timing-weight "${TIMING_WEIGHT:-0.1}" "${extra[@]}"
        ;;
    eval-smoke|eval)
        : "${GENERATOR_CKPT:?Set GENERATOR_CKPT to timing/control generator last.ckpt}"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            python -m navsim.planning.script.run_generator_timing evaluate "${common[@]}" \
            --generator "$GENERATOR_CKPT" --data-root "$DATA_ROOT" --metric-cache "$TEST_METRIC_CACHE" \
            --output "$GEN_EXP/$MODE" --workers "${EVAL_WORKERS:-0}" --score-workers "${SCORE_WORKERS:-4}" "${extra[@]}"
        ;;
    *)
        echo "Usage: $0 {check|test|prepare-features|prepare-pilot|smoke|smoke-ddp|prepare|train-control|train-timing|eval-smoke|eval}"
        echo "Run prepare-features only if the original raw feature cache was removed."
        echo "Compare control last.ckpt and timing last.ckpt after equal training, with the same fixed PCS/TRV."
        ;;
esac
