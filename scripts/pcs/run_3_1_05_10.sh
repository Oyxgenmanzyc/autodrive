#!/usr/bin/env bash
# Risk-conditioned spatial × timing modes. Run with bash, never source.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
MODE_EXP=${MODE_EXP:-$EXP_ROOT/risk_timing_modes_3_1_05_10}
RANK_EXP=${RANK_EXP:-$EXP_ROOT/cost_ranked_pcs_3_1_05_6}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
RANK_FEATURES=${RANK_FEATURES:-$RANK_EXP/features/navtrain}
RANK_CKPT=${RANK_CKPT:-$RANK_EXP/train/2026.09.19.19.00.42.148759/checkpoints/epoch=14.ckpt}
MODE_CACHE=${MODE_CACHE:-$MODE_EXP/cache/navtrain}
TRAIN_METRIC_CACHE=${TRAIN_METRIC_CACHE:-$EXP_ROOT/metric_cache_navtrain_pcs_3_1_05}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04,GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f,GPU-e9db5634-a044-a8fb-c294-6f372876e0e3,GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd,GPU-ca2805ee-6ed8-b153-e487-087872e087ac,GPU-12dd0e4c-f7aa-3523-a136-63f0782786f6,GPU-fa24445c-8297-4539-4b52-095a9ad39eea,GPU-d9166e6d-dfbc-3785-a88e-d171f0c79c78}
EVAL_GPU=${EVAL_GPU:-GPU-ca2805ee-6ed8-b153-e487-087872e087ac}
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NAVSIM_DEVKIT_ROOT="$CODE_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT" OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NCCL_DEBUG=WARN
unset TIMING_MODES_RUN_DIR
ulimit -n 65536 2>/dev/null || true
cd "$CODE_ROOT"
mkdir -p "$MODE_EXP/logs"
MODE=${1:-help}
runner=(python -m navsim.planning.script.run_timing_modes)

physical_gpus() {
    python - "$1" <<'PY'
import subprocess, sys
expected = {
 'GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04': '34',
 'GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f': '35',
 'GPU-e9db5634-a044-a8fb-c294-6f372876e0e3': '36',
 'GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd': '37',
 'GPU-ca2805ee-6ed8-b153-e487-087872e087ac': '9B',
 'GPU-12dd0e4c-f7aa-3523-a136-63f0782786f6': '9C',
 'GPU-fa24445c-8297-4539-4b52-095a9ad39eea': '9D',
 'GPU-d9166e6d-dfbc-3785-a88e-d171f0c79c78': '9E',
}
uuids = sys.argv[1].split(',')
if len(set(uuids)) != len(uuids):
    raise SystemExit('Duplicate physical GPUs')
for uuid in uuids:
    if uuid not in expected:
        raise SystemExit('Use a known full UUID, not a numeric CUDA index: '+uuid)
    row = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,pci.bus_id,name,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True).strip()
    parts = [part.strip() for part in row.split(',')]
    if parts[0] != uuid or parts[1].upper().split(':')[-2] != expected[uuid]:
        raise SystemExit('Physical UUID/PCI mismatch: '+row)
    print(row)
PY
}

logged() {
    local label=$1
    shift
    "$@" 2>&1 | tee "$MODE_EXP/logs/${label}_$(date +%Y%m%d_%H%M%S)_$$.log"
}

case "$MODE" in
    check)
        physical_gpus "$TRAIN_GPUS"
        for path in "$RANK_FEATURES/complete.json" "$RANK_CKPT" "$PCS_SCORER" \
                    "$BASELINE_CKPT" "$ANCHOR_PATH" "$BKB_PATH"; do
            test -f "$path" || { echo "Missing: $path"; exit 1; }
        done
        test -d "$TRAIN_METRIC_CACHE"
        test -d "$TEST_METRIC_CACHE"
        test -f "$PCS_CACHE/manifest.json"
        df -h "$EXP_ROOT"
        ;;
    test)
        logged test python -m unittest discover -s tests -p test_timing_modes.py
        ;;
    init-cache)
        logged init-cache "${runner[@]}" init-cache --features "$RANK_FEATURES" \
            --ranker "$RANK_CKPT" --scorer "$PCS_SCORER" \
            --candidate-cache "$PCS_CACHE" --output "$MODE_CACHE"
        ;;
    prepare-shard)
        : "${SPLIT:?Set SPLIT=train or val}"
        : "${SHARD_INDEX:?Set SHARD_INDEX in [0, NUM_SHARDS)}"
        physical_gpus "$EVAL_GPU"
        logged "prepare_${SPLIT}_${SHARD_INDEX}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
            "${runner[@]}" prepare-shard --features "$RANK_FEATURES" \
            --ranker "$RANK_CKPT" --scorer "$PCS_SCORER" \
            --candidate-cache "$PCS_CACHE" --output "$MODE_CACHE" --split "$SPLIT" \
            --metric-cache "$TRAIN_METRIC_CACHE" --num-shards "${NUM_SHARDS:-1}" \
            --shard-index "$SHARD_INDEX" --score-workers "${SCORE_WORKERS:-2}"
        ;;
    prepare-all)
        physical_gpus "$TRAIN_GPUS"
        "${runner[@]}" init-cache --features "$RANK_FEATURES" --ranker "$RANK_CKPT" \
            --scorer "$PCS_SCORER" --candidate-cache "$PCS_CACHE" --output "$MODE_CACHE"
        IFS=, read -r -a gpu_list <<< "$TRAIN_GPUS"
        for split in train val; do
            pids=()
            for i in "${!gpu_list[@]}"; do
                log="$MODE_EXP/logs/prepare_${split}_shard_${i}.log"
                env EVAL_GPU="${gpu_list[$i]}" SPLIT="$split" SHARD_INDEX="$i" \
                    NUM_SHARDS="${#gpu_list[@]}" SCORE_WORKERS="${SCORE_WORKERS:-2}" \
                    bash "$CODE_ROOT/scripts/pcs/run_3_1_05_10.sh" prepare-shard > "$log" 2>&1 &
                pids+=("$!")
                echo "$split shard $i: PID ${pids[-1]} GPU ${gpu_list[$i]} log $log"
            done
            failed=0
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then failed=1; fi
            done
            if (( failed )); then echo "$split shard failed; inspect logs"; exit 1; fi
        done
        "${runner[@]}" complete-cache --features "$RANK_FEATURES" \
            --ranker "$RANK_CKPT" --scorer "$PCS_SCORER" \
            --candidate-cache "$PCS_CACHE" --output "$MODE_CACHE"
        ;;
    complete-cache)
        logged complete-cache "${runner[@]}" complete-cache --features "$RANK_FEATURES" \
            --ranker "$RANK_CKPT" --scorer "$PCS_SCORER" \
            --candidate-cache "$PCS_CACHE" --output "$MODE_CACHE"
        ;;
    diagnose)
        logged diagnose "${runner[@]}" diagnose --features "$RANK_FEATURES" \
            --cache "$MODE_CACHE" --output "$MODE_EXP/oracle_diagnostics"
        ;;
    smoke|train)
        physical_gpus "$TRAIN_GPUS"
        devices=$(awk -F, '{print NF}' <<< "$TRAIN_GPUS")
        extras=()
        if [ "$MODE" = smoke ]; then extras+=(--smoke); fi
        if [ -n "${MODE_RESUME:-}" ]; then extras+=(--resume "$MODE_RESUME"); fi
        logged "$MODE" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${runner[@]}" train \
            --features "$RANK_FEATURES" --cache "$MODE_CACHE" --output "$MODE_EXP/$MODE" \
            --devices "$devices" --epochs "${EPOCHS:-20}" \
            --batch-size "${BATCH_SIZE:-32}" --workers "${WORKERS:-0}" \
            --lr "${LR:-1e-4}" "${extras[@]}"
        ;;
    validate)
        : "${MODE_CKPT:?Set MODE_CKPT to a formal timing checkpoint}"
        physical_gpus "$EVAL_GPU"
        logged validate env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" validate \
            --features "$RANK_FEATURES" --cache "$MODE_CACHE" --timing "$MODE_CKPT" \
            --output "$MODE_EXP/validation" --batch-size "${BATCH_SIZE:-32}" \
            --workers "${WORKERS:-0}"
        ;;
    eval-smoke|eval)
        : "${MODE_CKPT:?Set MODE_CKPT; inspect independent audit first}"
        physical_gpus "$EVAL_GPU"
        extras=()
        if [ "$MODE" = eval-smoke ]; then extras+=(--max-scenes 8); fi
        logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" evaluate \
            --baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH" \
            --scorer "$PCS_SCORER" --ranker "$RANK_CKPT" --timing "$MODE_CKPT" \
            --cache "$MODE_CACHE" --metric-cache "$TEST_METRIC_CACHE" \
            --data-root "$DATA_ROOT" --output "$MODE_EXP/$MODE" \
            --workers "${WORKERS:-0}" --score-workers "${SCORE_WORKERS:-2}" "${extras[@]}"
        ;;
    *)
        echo 'Usage: bash scripts/pcs/run_3_1_05_10.sh {check|test|init-cache|prepare-shard|prepare-all|complete-cache|diagnose|smoke|train|validate|eval-smoke|eval}'
        echo 'Order: check -> test -> prepare-all -> diagnose -> smoke -> train -> validate -> audit review -> eval-smoke -> eval'
        ;;
esac
