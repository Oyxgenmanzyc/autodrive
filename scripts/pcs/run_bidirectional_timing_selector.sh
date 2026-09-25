#!/usr/bin/env bash
set -euo pipefail
if [ -n "${CONDA_SH:-}" ]; then source "$CONDA_SH"; elif [ -f /home/ubutnu/anaconda3/etc/profile.d/conda.sh ]; then source /home/ubutnu/anaconda3/etc/profile.d/conda.sh; else source /home/hndx/miniconda3/etc/profile.d/conda.sh; fi
conda activate navhigh
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXP_ROOT=${EXP_ROOT:-$HOME/navsim_workspace/exp}
RANK_ROOT=${RANK_ROOT:-$EXP_ROOT/cost_ranked_pcs_3_1_05_6}
RANK_FEATURES=${RANK_FEATURES:-$RANK_ROOT/features/navtrain}
RANK_CKPT=${RANK_CKPT:-$RANK_ROOT/train/2026.09.19.19.00.42.148759/checkpoints/epoch=14.ckpt}
METRIC_CACHE=${TRAIN_METRIC_CACHE:-$EXP_ROOT/metric_cache_navtrain_pcs_3_1_05}
OLD_ORACLE=${OLD_ORACLE:-$EXP_ROOT/bidirectional_timing_oracle_3_1_05_6/val}
SELECTOR_EXP=${SELECTOR_EXP:-$EXP_ROOT/bidirectional_timing_selector_3_1_05_7}
TRAIN_ORACLE=${TRAIN_ORACLE:-$SELECTOR_EXP/oracle_train}
EVAL_GPU=${EVAL_GPU:-0}
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" NAVSIM_DEVKIT_ROOT="$CODE_ROOT" NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-$HOME/navsim_workspace/dataset}
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ulimit -n 65536 2>/dev/null || true
cd "$CODE_ROOT"
mkdir -p "$SELECTOR_EXP/logs"
MODE=${1:-help}
case "$MODE" in
  check)
    for path in "$RANK_FEATURES/manifest.json" "$RANK_FEATURES/complete.json" "$RANK_FEATURES/train_records.json" "$RANK_FEATURES/val_records.json" "$RANK_CKPT" "$OLD_ORACLE/manifest.json"; do test -f "$path" || { echo "Missing: $path"; exit 1; }; done
    test -d "$METRIC_CACHE" || { echo "Missing: $METRIC_CACHE"; exit 1; }
    nvidia-smi -i "$EVAL_GPU" --query-gpu=uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    echo "PASS: selector prerequisites are complete"
    ;;
  test)
    python -m unittest tests.test_bidirectional_timing_selector
    ;;
  prepare-train)
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m navsim.planning.script.run_bidirectional_timing_oracle prepare --features "$RANK_FEATURES" --ranker "$RANK_CKPT" --metric-cache "$METRIC_CACHE" --output "$TRAIN_ORACLE" --split train --limit 0 --score-workers "${SCORE_WORKERS:-4}" --num-shards 1 --shard-index 0 2>&1 | tee "$SELECTOR_EXP/logs/prepare_train_$(date +%Y%m%d_%H%M%S)_$$.log"
    ;;
  train)
    train_args=(--features "$RANK_FEATURES" --train-oracle "$TRAIN_ORACLE" --val-oracle "$OLD_ORACLE" --output "$SELECTOR_EXP/train" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-32}" --workers "${WORKERS:-0}" --lr "${LR:-1e-4}")
    if [ -n "${RESUME_CKPT:-}" ]; then train_args+=(--resume "$RESUME_CKPT"); fi
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m navsim.planning.script.run_bidirectional_timing_selector "${train_args[@]}" 2>&1 | tee "$SELECTOR_EXP/logs/train_$(date +%Y%m%d_%H%M%S)_$$.log"
    ;;
  *)
    echo 'Usage: bash scripts/pcs/run_bidirectional_timing_selector.sh {check|test|prepare-train|train}'
    ;;
esac
