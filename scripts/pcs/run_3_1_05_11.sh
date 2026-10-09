#!/usr/bin/env bash
# Standalone generator-timing experiment. Invoke with bash; never source it.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
MODE_EXP=${MODE_EXP:-$EXP_ROOT/generator_timing_modes_3_1_05_11}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
TRAIN_METRIC_CACHE=${TRAIN_METRIC_CACHE:-$EXP_ROOT/metric_cache_navtrain_pcs_3_1_05}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
RANK_CKPT=${RANK_CKPT:-$EXP_ROOT/cost_ranked_pcs_3_1_05_6/train/2026.09.19.19.00.42.148759/checkpoints/epoch=14.ckpt}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04,GPU-2b61b5dd-693d-f98c-3f20-93e71630d30f,GPU-e9db5634-a044-a8fb-c294-6f372876e0e3,GPU-0ac26b0b-cec4-cb77-0cae-49c60e08a8cd}
EVAL_GPU=${EVAL_GPU:-GPU-cb61e34b-1bbd-919c-9df2-71ed67e97e04}
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NAVSIM_DEVKIT_ROOT="$CODE_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT" OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NCCL_DEBUG=WARN
cd "$CODE_ROOT"
mkdir -p "$MODE_EXP/logs"
MODE=${1:-help}
runner=(python -m navsim.planning.script.run_generator_timing)
common=(--candidates "$PCS_CACHE" --teachers "$MODE_EXP/teachers"
        --metric-cache "$TRAIN_METRIC_CACHE" --baseline "$BASELINE_CKPT"
        --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH")

check_gpu() {
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
    raise SystemExit('Duplicate GPU UUID')
for uuid in uuids:
    if uuid not in expected:
        raise SystemExit('Use a full physical GPU UUID: '+uuid)
    output = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,pci.bus_id,name,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True).strip()
    parts = [part.strip() for part in output.split(',')]
    if parts[0] != uuid or parts[1].upper().split(':')[-2] != expected[uuid]:
        raise SystemExit('Physical GPU mapping changed: '+output)
    print(output)
PY
}

prepare_split() {
    local split=$1 fail=0 i
    local shards=${TEACHER_SHARDS:-8}
    local -a pids=()
    for ((i=0; i<shards; i++)); do
        "${runner[@]}" prepare-shard "${common[@]}" --output "$MODE_EXP/teachers" \
            --split "$split" --shard-index "$i" --num-shards "$shards" \
            > "$MODE_EXP/logs/teacher_${split}_${i}.log" 2>&1 &
        pids+=("$!")
        echo "$split shard $i PID ${pids[-1]}"
    done
    for i in "${pids[@]}"; do wait "$i" || fail=1; done
    if [ "$fail" -ne 0 ]; then
        echo "Failed $split teacher shard; inspect $MODE_EXP/logs/teacher_${split}_*.log"
        return 1
    fi
}

case "$MODE" in
    check)
        check_gpu "$TRAIN_GPUS"
        python - "$PCS_CACHE" "$BASELINE_CKPT" "$ANCHOR_PATH" <<'PY'
import sys
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
from navsim.agents.diffusiondrive.pcs.common import provenance
root, baseline, anchor = sys.argv[1:]
for split in ('train', 'val'):
    data = CandidateDataset(root, split)
    if data.provenance != provenance(baseline, anchor, 0):
        raise ValueError('Original K67 provenance mismatch')
    print(split, len(data))
print('PASS: original generator/cache provenance')
PY
        df -h "$EXP_ROOT"
        ;;
    test)
        python -m unittest discover -s tests -p test_generator_timing.py
        ;;
    init-cache)
        "${runner[@]}" init-cache "${common[@]}" --output "$MODE_EXP/teachers"
        ;;
    pilot)
        pilot="$MODE_EXP/pilot_teachers"
        "${runner[@]}" init-cache "${common[@]}" --output "$pilot"
        "${runner[@]}" prepare-shard "${common[@]}" --output "$pilot" \
            --split train --shard-index 0 --num-shards 100000
        "${runner[@]}" prepare-shard "${common[@]}" --output "$pilot" \
            --split val --shard-index 0 --num-shards 100000
        ;;
    prepare-all)
        bash "$CODE_ROOT/scripts/pcs/run_3_1_05_11.sh" init-cache
        prepare_split train
        prepare_split val
        bash "$CODE_ROOT/scripts/pcs/run_3_1_05_11.sh" complete-cache
        ;;
    complete-cache)
        "${runner[@]}" complete-cache "${common[@]}" --output "$MODE_EXP/teachers"
        ;;
    smoke-train|train)
        check_gpu "$TRAIN_GPUS"
        devices=$(awk -F, '{print NF}' <<< "$TRAIN_GPUS")
        extra=()
        output="$MODE_EXP/train"
        if [ "$MODE" = smoke-train ]; then
            extra+=(--max-train-scenes 16)
            output="$MODE_EXP/smoke-train"
        fi
        if [ -n "${RESUME_CKPT:-}" ]; then extra+=(--resume "$RESUME_CKPT"); fi
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun --standalone --nproc_per_node="$devices" \
            -m navsim.planning.script.run_generator_timing train "${common[@]}" \
            --output "$output" --epochs "${EPOCHS:-20}" \
            --batch-size "${BATCH_SIZE:-1}" --workers "${WORKERS:-0}" \
            --lr "${LR:-1e-4}" "${extra[@]}"
        ;;
    validate-pilot|validate|navtest)
        : "${TIMING_CKPT:?Set TIMING_CKPT to the trained generator adapter checkpoint}"
        check_gpu "$EVAL_GPU"
        extra=()
        metric=$TRAIN_METRIC_CACHE
        output="$MODE_EXP/validation"
        command=validate
        if [ "$MODE" = validate-pilot ]; then
            extra+=(--max-scenes 8)
            output="$MODE_EXP/validation-pilot"
        elif [ "$MODE" = navtest ]; then
            : "${DECISION_JSON:?Set DECISION_JSON from successful independent validation}"
            metric=$TEST_METRIC_CACHE
            output="$MODE_EXP/navtest"
            command=navtest
            extra+=(--decision "$DECISION_JSON")
        fi
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" "$command" \
            --candidates "$PCS_CACHE" --teachers "$MODE_EXP/teachers" \
            --metric-cache "$metric" --baseline "$BASELINE_CKPT" \
            --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH" \
            --output "$output" --generator-checkpoint "$TIMING_CKPT" \
            --scorer "$PCS_SCORER" --ranker "$RANK_CKPT" \
            --data-root "$DATA_ROOT" --score-workers "${SCORE_WORKERS:-2}" "${extra[@]}"
        ;;
    *)
        echo 'Usage: bash scripts/pcs/run_3_1_05_11.sh {check|test|pilot|prepare-all|smoke-train|train|validate-pilot|validate|navtest}'
        echo 'Order: check -> test -> pilot -> prepare-all -> smoke-train -> train -> validate-pilot -> validate -> navtest only when pass_for_navtest=true'
        ;;
esac
