#!/usr/bin/env bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to the 3.1.05 checkout}"
: "${ANCHOR_PATH:?Use the exact bank and order from SPR training}"
: "${BKB_PATH:?Set BKB_PATH to the ResNet34 weights}"
: "${CHECKPOINT:?Set CHECKPOINT to a trained 3.1.05 checkpoint}"
SPR_OUTPUT="${SPR_OUTPUT:-reconstruction}"
export SPR_EVAL_CHECKPOINT="$CHECKPOINT"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  agent=diffusiondrive_spr_agent \
  train_test_split=navtest \
  worker=single_machine_thread_pool \
  "experiment_name=diffusiondrive_spr_3_1_05_$SPR_OUTPUT" \
  "+agent.config.plan_anchor_path=$ANCHOR_PATH" \
  "+agent.config.bkb_path=$BKB_PATH" \
  "agent.config.spr_output=$SPR_OUTPUT" \
  'agent.checkpoint_path=${oc.env:SPR_EVAL_CHECKPOINT}' \
  "$@"
