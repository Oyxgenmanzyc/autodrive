#!/usr/bin/env bash
# Existing NAVSIM environment and feature cache are required. Extra Hydra args
# are passed last, so the same launcher supports baseline and LAST ablations.
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to this checkout}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the experiment output directory}"
: "${BKB_PATH:?Set BKB_PATH to the ResNet-34 pretrained weights}"
: "${PLAN_ANCHOR_PATH:?Set PLAN_ANCHOR_PATH to kmeans_navsim_traj_20.npy}"

for file in "$BKB_PATH" "$PLAN_ANCHOR_PATH"; do
  if [[ ! -f "$file" ]]; then
    printf 'Required file does not exist: %s\n' "$file" >&2
    exit 1
  fi
done

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
  +trainer.params.devices="${DEVICES:-1}" \
  dataloader.params.batch_size="${BATCH_SIZE:-16}" \
  dataloader.params.num_workers="${NUM_WORKERS:-4}" \
  cache_path="${TRAIN_CACHE_PATH:-${NAVSIM_EXP_ROOT}/training_cache}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  +agent.config.bkb_path="$BKB_PATH" \
  +agent.config.plan_anchor_path="$PLAN_ANCHOR_PATH" \
  agent.checkpoint_path="${INIT_CKPT:-null}" \
  "$@"
