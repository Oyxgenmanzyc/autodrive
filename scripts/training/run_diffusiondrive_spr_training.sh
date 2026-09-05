#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to the 3.1.05 checkout}"
: "${ANCHOR_PATH:?Set ANCHOR_PATH to the same bank used for the baseline}"
: "${BKB_PATH:?Set BKB_PATH to the ResNet34 weights}"

checkpoint_override=("agent.checkpoint_path=null")
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  export SPR_INIT_CHECKPOINT="$INIT_CHECKPOINT"
  checkpoint_override=('agent.checkpoint_path=${oc.env:SPR_INIT_CHECKPOINT}')
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
  agent=diffusiondrive_spr_agent \
  train_test_split=navtrain \
  experiment_name=diffusiondrive_spr_3_1_05 \
  "+agent.config.plan_anchor_path=$ANCHOR_PATH" \
  "+agent.config.bkb_path=$BKB_PATH" \
  "${checkpoint_override[@]}" \
  "$@"
