#!/usr/bin/env bash
# Server defaults target /home/hndx/navsim_workspace and physical GPUs 4-7.
# Environment variables and trailing Hydra arguments can override every value.
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to this checkout}"

DATA_ROOT=${DATA_ROOT:-/home/hndx/navsim_workspace/dataset}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/home/hndx/navsim_workspace/exp}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-$DATA_ROOT}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

BKB_PATH=${BKB_PATH:-$DATA_ROOT/pytorch_model.bin}
PLAN_ANCHOR_PATH=${PLAN_ANCHOR_PATH:-$DATA_ROOT/kmeans_navsim_traj_20.npy}
TRAIN_CACHE_PATH=${TRAIN_CACHE_PATH:-$NAVSIM_EXP_ROOT/training_cache}

for file in "$BKB_PATH" "$PLAN_ANCHOR_PATH"; do
  if [[ ! -f "$file" ]]; then
    printf 'Required file does not exist: %s\n' "$file" >&2
    exit 1
  fi
done
if [[ ! -d "$TRAIN_CACHE_PATH" ]]; then
  printf 'Training cache does not exist: %s\nRun the cache command in docs/experiment_3_3_1.md first.\n' \
    "$TRAIN_CACHE_PATH" >&2
  exit 1
fi

export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
cd "$NAVSIM_DEVKIT_ROOT"

python navsim/planning/script/run_training.py \
  agent=diffusiondrive_last_agent \
  experiment_name="${EXPERIMENT_NAME:-training_diffusiondrive_3_3_1_last_gate}" \
  train_test_split=navtrain \
  split=trainval \
  seed="${SEED:-0}" \
  trainer.params.max_epochs=100 \
  trainer.params.precision=16-mixed \
  +trainer.params.devices="${DEVICES:-4}" \
  dataloader.params.batch_size="${BATCH_SIZE:-32}" \
  dataloader.params.num_workers="${NUM_WORKERS:-4}" \
  cache_path="$TRAIN_CACHE_PATH" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  +agent.config.bkb_path="$BKB_PATH" \
  +agent.config.plan_anchor_path="$PLAN_ANCHOR_PATH" \
  agent.lr="${LR:-1e-4}" \
  agent.checkpoint_path="${INIT_CKPT:-null}" \
  "$@"
