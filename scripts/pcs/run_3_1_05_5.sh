#!/usr/bin/env bash
# Paired Gate + post-selection timing refiner, 3.1.05_5. Invoke with bash, do not source.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
PTR_EXP=${PTR_EXP:-$EXP_ROOT/gated_paired_timing_3_1_05_5}
GATE_CACHE=${GATE_CACHE:-$PTR_EXP/gate_cache}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
EDIT_CACHE=${EDIT_CACHE:-$EXP_ROOT/post_selection_timing_3_1_05_4/edit_cache/navtrain}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
VETO_CKPT=${VETO_CKPT:-$EXP_ROOT/triple_risk_veto_k67_3_1_05_2/train/2026.09.09.14.53.12.850967/checkpoints/epoch=02.ckpt}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-ca2805ee-6ed8-b153-e487-087872e087ac,GPU-12dd0e4c-f7aa-3523-a136-63f0782786f6,GPU-fa24445c-8297-4539-4b52-095a9ad39eea,GPU-d9166e6d-dfbc-3785-a88e-d171f0c79c78}
EVAL_GPU=${EVAL_GPU:-GPU-ca2805ee-6ed8-b153-e487-087872e087ac}
export NAVSIM_DEVKIT_ROOT="$CODE_ROOT" OPENSCENE_DATA_ROOT="$DATA_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PTR_RUN_DIR GATED_RUN_DIR
cd "$CODE_ROOT"
mkdir -p "$PTR_EXP/logs"
MODE=${1:-help}
run_logged() {
    local label=$1
    shift
    "$@" 2>&1 | tee "$PTR_EXP/logs/${label}_$(date +%Y%m%d_%H%M%S)_$$.log"
}
verify_gpu() {
    python - "$1" <<'PY'
import subprocess, sys
expected = {
    'GPU-ca2805ee-6ed8-b153-e487-087872e087ac': '00000000:9B:00.0',
    'GPU-12dd0e4c-f7aa-3523-a136-63f0782786f6': '00000000:9C:00.0',
    'GPU-fa24445c-8297-4539-4b52-095a9ad39eea': '00000000:9D:00.0',
    'GPU-d9166e6d-dfbc-3785-a88e-d171f0c79c78': '00000000:9E:00.0',
}
uuids = sys.argv[1].split(',')
if len(set(uuids)) != len(uuids):
    raise SystemExit('Duplicate GPU UUID')
for uuid in uuids:
    if uuid not in expected:
        raise SystemExit('This release pins the four physical Ada cards by UUID; unknown UUID: '+uuid)
    line = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=pci.bus_id,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    bus, actual, name, *_ = [v.strip() for v in line.split(',')]
    if bus.upper() != expected[uuid] or actual != uuid or '5880' not in name:
        raise SystemExit('Physical GPU mapping mismatch: '+line)
    print(line)
PY
}
train_stage() {
    local stage=$1 devices=4 epochs=${EPOCHS:-20} batch=${BATCH_SIZE:-32} gpu=$TRAIN_GPUS
    local extra=() run
    if [ "$MODE" = smoke ] || [ "$MODE" = smoke-ddp ]; then
        devices=1; epochs=1; batch=4; gpu=$EVAL_GPU; extra+=(--smoke)
        if [ "$MODE" = smoke-ddp ]; then devices=4; gpu=$TRAIN_GPUS; fi
    fi
    verify_gpu "$gpu"
    if [ "$devices" -eq 4 ] && [ "$(awk -F, '{print NF}' <<< "$gpu")" -ne 4 ]; then
        echo 'Expected exactly four physical GPU UUIDs'; return 1
    fi
    run="$PTR_EXP/$MODE/$stage/$(date +%Y.%m.%d.%H.%M.%S)_$$"
    mkdir -p "$run"
    if [ -n "${RESUME_CKPT:-}" ]; then
        extra+=(--resume "$RESUME_CKPT")
    elif [ "$stage" = joint ] && [ "$MODE" != smoke ] && [ "$MODE" != smoke-ddp ]; then
        : "${PROBE_CKPT:?Set PROBE_CKPT to the best Gate probe checkpoint}"
        extra+=(--init "$PROBE_CKPT")
    fi
    run_logged "$MODE-$stage" env CUDA_VISIBLE_DEVICES="$gpu" GATED_RUN_DIR="$run" \
        python -m navsim.planning.script.run_gated_refinement train \
        --candidate-cache "$PCS_CACHE" --edit-cache "$EDIT_CACHE" --gate-cache "$GATE_CACHE" \
        --output "$PTR_EXP/$MODE/$stage" --stage "$stage" --devices "$devices" \
        --batch-size "$batch" --epochs "$epochs" --workers "${WORKERS:-0}" --lr "${LR:-1e-4}" "${extra[@]}"
    echo "Run result: $run/result.json"
}
case "$MODE" in
    check)
        for file in "$PCS_CACHE/manifest.json" "$EDIT_CACHE/manifest.json" "$EDIT_CACHE/train_records.json" "$EDIT_CACHE/val_records.json"; do
            test -f "$file" || { echo "Missing: $file"; exit 1; }
        done
        verify_gpu "$TRAIN_GPUS"
        df -h "$EXP_ROOT"
        ;;
    test)
        run_logged test python -m unittest discover -s tests -p 'test_gated_refinement.py'
        ;;
    prepare)
        run_logged labels env CUDA_VISIBLE_DEVICES='' python -m navsim.planning.script.run_gated_refinement prepare \
            --edit-cache "$EDIT_CACHE" --output "$GATE_CACHE" --min-gain "${MIN_GAIN:-0.02}"
        ;;
    smoke|smoke-ddp)
        train_stage joint
        ;;
    probe)
        train_stage probe
        ;;
    train)
        train_stage joint
        ;;
    eval-smoke|eval)
        : "${REFINER_CKPT:?Set REFINER_CKPT to a JOINT trained checkpoint}"
        verify_gpu "$EVAL_GPU"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m navsim.planning.script.run_gated_refinement evaluate \
            --baseline "$BASELINE_CKPT" --backbone "$BKB_PATH" --anchor "$ANCHOR_PATH" --pcs-scorer "$PCS_SCORER" \
            --veto "$VETO_CKPT" --refiner "$REFINER_CKPT" --metric-cache "$TEST_METRIC_CACHE" \
            --data-root "$DATA_ROOT" --output "$PTR_EXP/$MODE" --workers 0 --score-workers "${SCORE_WORKERS:-2}" "${extra[@]}"
        ;;
    *)
        echo "Usage: bash $0 {check|test|prepare|smoke|smoke-ddp|probe|train|eval-smoke|eval}"
        echo 'prepare uses full existing 98MB edit cache; no new PDM/PCS cache or GPU scoring.'
        ;;
esac
