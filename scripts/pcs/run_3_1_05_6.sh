#!/usr/bin/env bash
# Independent ranker. Run this file with bash; errors cannot exit the parent terminal.
set -euo pipefail
source "${CONDA_SH:-/home/hndx/miniconda3/etc/profile.d/conda.sh}"
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_ROOT=${EXP_ROOT:-/home/hndx/navsim_workspace/exp}
DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
RANK_EXP=${RANK_EXP:-$EXP_ROOT/cost_ranked_pcs_3_1_05_6}
PCS_CACHE=${PCS_CACHE:-$EXP_ROOT/pcs_candidates_k67_3_1_05}
RANK_FEATURES=${RANK_FEATURES:-$RANK_EXP/features/navtrain}
PILOT_RANK_FEATURES=${PILOT_RANK_FEATURES:-$RANK_EXP/features/pilot}
OOF_ROOT=${OOF_ROOT:-$RANK_EXP/oof}
PCS_SCORER=${PCS_SCORER:-$EXP_ROOT/pdm_candidate_scoring_k67_3_1_05/train/2026.09.07.19.15.17.705777/checkpoints/epoch=05.ckpt}
REFERENCE_TRV=${REFERENCE_TRV:-$EXP_ROOT/triple_risk_veto_k67_3_1_05_2/train/2026.09.09.14.53.12.850967/checkpoints/epoch=02.ckpt}
BASELINE_CKPT=${BASELINE_CKPT:-$EXP_ROOT/adaptive_multimodal_anchor_k67_3_1_02_4gpu_resume/2026.08.24.11.26.23/lightning_logs/version_0/checkpoints/epoch=99-step=266000.ckpt}
ANCHOR_PATH=${ANCHOR_PATH:-$EXP_ROOT/anchors/adaptive_v1/adaptive_anchor_bank.npy}
BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$EXP_ROOT/metric_cache}
TRAIN_GPUS=${TRAIN_GPUS:-GPU-ca2805ee-6ed8-b153-e487-087872e087ac,GPU-12dd0e4c-f7aa-3523-a136-63f0782786f6,GPU-fa24445c-8297-4539-4b52-095a9ad39eea,GPU-d9166e6d-dfbc-3785-a88e-d171f0c79c78}
EVAL_GPU=${EVAL_GPU:-GPU-ca2805ee-6ed8-b153-e487-087872e087ac}
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NAVSIM_DEVKIT_ROOT="$CODE_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT" OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NCCL_DEBUG=WARN
# Avoid accidentally resuming a previous gate/probe or reusing a previous DDP directory.
unset COST_RANK_RUN_DIR
ulimit -n 65536 2>/dev/null || true
cd "$CODE_ROOT"
mkdir -p "$RANK_EXP/logs"
MODE=${1:-help}
runner=(python -m navsim.planning.script.run_cost_rank)

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
        raise SystemExit('Use a known full GPU UUID, never a numeric CUDA index: '+uuid)
    row = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,pci.bus_id,name,memory.used,utilization.gpu',
        '--format=csv,noheader'], text=True).strip()
    parts = [p.strip() for p in row.split(',')]
    if parts[0] != uuid or parts[1].upper().split(':')[-2] != expected[uuid]:
        raise SystemExit('Physical UUID/PCI mismatch: '+row)
    print(row)
PY
}

run_logged() {
    local label=$1
    shift
    "$@" 2>&1 | tee "$RANK_EXP/logs/${label}_$(date +%Y%m%d_%H%M%S)_$$.log"
}

train_rank() {
    local smoke=$1
    local features=$RANK_FEATURES
    local output=$RANK_EXP/train
    local visible=$TRAIN_GPUS
    local extra=()
    local epochs=${EPOCHS:-20}
    if [ "$smoke" = yes ]; then
        features=$PILOT_RANK_FEATURES
        output=$RANK_EXP/smoke
        visible=$EVAL_GPU
        if [ "$MODE" = smoke-ddp ]; then visible=$TRAIN_GPUS; output=$RANK_EXP/smoke-ddp; fi
        epochs=1
        extra+=(--smoke)
    elif [ -n "${RANK_RESUME:-}" ]; then
        extra+=(--resume "$RANK_RESUME")
    fi
    physical_gpus "$visible"
    local devices
    devices=$(awk -F, '{print NF}' <<< "$visible")
    run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$visible" "${runner[@]}" train \
        --features "$features" --oof-root "$OOF_ROOT" --output "$output" \
        --devices "$devices" --epochs "$epochs" --batch-size "${BATCH_SIZE:-32}" \
        --workers "${WORKERS:-0}" --lr "${LR:-1e-4}" "${extra[@]}"
}

case "$MODE" in
    check)
        physical_gpus "$TRAIN_GPUS"
        python - "$PCS_CACHE" "$PCS_SCORER" <<'PY'
import json, sys
from pathlib import Path
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
from navsim.agents.diffusiondrive.pcs.common import load_scorer
from navsim.agents.diffusiondrive.cost_rank.data import validate_records
root = Path(sys.argv[1])
validate_records(json.loads((root/'records.json').read_text()))
for split in ('train', 'val'):
    data = CandidateDataset(root, split)
    print(split, 'scenes:', len(data))
load_scorer(sys.argv[2], 'cpu', data.provenance)
print('PASS: candidate coverage, source hashes, PCS checkpoint match.')
print('Training does not need raw training_cache, metric_cache, GTRS or PTR edit/gate caches.')
PY
        df -h "$EXP_ROOT"
        ;;
    test)
        run_logged test python -m unittest discover -s tests -p test_cost_rank.py
        ;;
    prepare-pilot|prepare)
        physical_gpus "$EVAL_GPU"
        target=$RANK_FEATURES
        limit=0
        if [ "$MODE" = prepare-pilot ]; then target=$PILOT_RANK_FEATURES; limit=${PILOT_SCENES:-128}; fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" prepare \
            --cache "$PCS_CACHE" --scorer "$PCS_SCORER" --output "$target" --limit "$limit" \
            --batch-size "${PREPARE_BATCH_SIZE:-32}" --workers "${WORKERS:-0}"
        ;;
    smoke|smoke-ddp) train_rank yes ;;
    train) train_rank no ;;
    oof-fold)
        : "${FOLD_INDEX:?Set FOLD_INDEX=0, 1 or 2}"
        physical_gpus "$TRAIN_GPUS"
        devices=$(awk -F, '{print NF}' <<< "$TRAIN_GPUS")
        resume=()
        if [ -n "${TEACHER_RESUME:-}" ]; then resume+=(--resume "$TEACHER_RESUME"); fi
        run_logged "teacher_${FOLD_INDEX}" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${runner[@]}" train-fold \
            --cache "$PCS_CACHE" --oof-root "$OOF_ROOT" --fold-index "$FOLD_INDEX" --folds 3 \
            --epochs "${TEACHER_EPOCHS:-20}" --devices "$devices" --batch-size "${BATCH_SIZE:-32}" \
            --workers "${WORKERS:-0}" --lr "${TEACHER_LR:-3e-4}" "${resume[@]}"
        physical_gpus "$EVAL_GPU"
        run_logged "predict_${FOLD_INDEX}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" predict-fold \
            --cache "$PCS_CACHE" --oof-root "$OOF_ROOT" --fold-index "$FOLD_INDEX" \
            --batch-size "${PREPARE_BATCH_SIZE:-32}" --workers "${WORKERS:-0}"
        ;;
    oof)
        if [ -n "${TEACHER_RESUME:-}" ]; then
            echo 'For teacher resume use oof-fold with the matching FOLD_INDEX.'
            exit 1
        fi
        # Three independent teachers, sequentially using the selected physical GPUs.
        for fold in 0 1 2; do
            FOLD_INDEX=$fold bash "$CODE_ROOT/scripts/pcs/run_3_1_05_6.sh" oof-fold
        done
        bash "$CODE_ROOT/scripts/pcs/run_3_1_05_6.sh" assemble-oof
        ;;
    assemble-oof)
        run_logged assemble "${runner[@]}" assemble-oof --features "$RANK_FEATURES" --oof-root "$OOF_ROOT" --folds 3
        ;;
    validate)
        : "${RANK_CKPT:?Set RANK_CKPT to a formal ranker checkpoint}"
        physical_gpus "$EVAL_GPU"
        run_logged validate env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" validate \
            --features "$RANK_FEATURES" --ranker "$RANK_CKPT" --output "$RANK_EXP/validation" \
            --batch-size "${BATCH_SIZE:-32}" --workers "${WORKERS:-0}"
        ;;
    eval-smoke|eval)
        : "${RANK_CKPT:?Set RANK_CKPT; review calibration AND audit first}"
        physical_gpus "$EVAL_GPU"
        extra=()
        if [ "$MODE" = eval-smoke ]; then extra+=(--max-scenes 8); fi
        run_logged "$MODE" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${runner[@]}" evaluate \
            --ranker "$RANK_CKPT" --scorer "$PCS_SCORER" --reference-trv "$REFERENCE_TRV" \
            --baseline "$BASELINE_CKPT" --anchor "$ANCHOR_PATH" --backbone "$BKB_PATH" \
            --metric-cache "$TEST_METRIC_CACHE" --data-root "$DATA_ROOT" --output "$RANK_EXP/$MODE" \
            --workers "${WORKERS:-0}" --score-workers "${SCORE_WORKERS:-2}" "${extra[@]}"
        ;;
    *)
        echo 'Usage: bash scripts/pcs/run_3_1_05_6.sh {check|test|prepare-pilot|smoke|smoke-ddp|prepare|oof|oof-fold|assemble-oof|train|validate|eval-smoke|eval}'
        echo 'Order: check -> test -> prepare-pilot -> smoke -> smoke-ddp -> prepare -> oof -> train -> review -> validate -> eval-smoke -> eval'
        ;;
esac
