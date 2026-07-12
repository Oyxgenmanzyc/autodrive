# 3.21 Risk-Conditioned Mode Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add minimal timing-aware pair mining and an isolated risk-conditioned mode ranking head without changing trajectory generation.

**Architecture:** Cache a continuous GT future-front track, mine same-behavior earlier-braking pairs from detached final trajectories, and train a zero-initialized residual classification head from detached mode/risk features. Keep original decoder losses untouched and expose pair-quality metrics through the existing Lightning loss dictionary.

**Tech Stack:** Python, NumPy, PyTorch, NAVSIM TargetBuilder, PyTorch Lightning, Hydra.

---

### Task 1: Cache Future-Front Ranking Targets

**Files:**
- Modify: `navsim/agents/diffusiondrive/modules/risk_utils.py`
- Modify: `navsim/agents/diffusiondrive/transfuser_features.py`
- Test: `tests/test_diffusiondrive_risk_mode_ranking.py`

- [ ] Extend current-front selection with box metadata and add current-frame coordinate conversion for a continuous future `track_token`.
- [ ] Return `risk_front_future`, `risk_pair_context`, and `risk_pair_scene_active` tensors only when risk-aware classification is enabled.
- [ ] Version the target cache unique name so stale 3.20 caches cannot be silently reused.

### Task 2: Mine Minimal Timing Pairs

**Files:**
- Create: `navsim/agents/diffusiondrive/modules/risk_mode_ranking.py`
- Test: `tests/test_diffusiondrive_risk_mode_ranking.py`

- [ ] Derive sustained brake-onset steps from detached trajectories and current ego speed.
- [ ] Build pair masks from path similarity, one-step timing improvement, progress tolerance, and GT risk-scene validity.
- [ ] Return pair counts and interpretable timing/rank diagnostics without weighted safety scoring.

### Task 3: Add Isolated Residual Ranking Head

**Files:**
- Modify: `navsim/agents/diffusiondrive/transfuser_model_v2.py`
- Modify: `navsim/agents/diffusiondrive/transfuser_loss.py`
- Test: `tests/test_diffusiondrive_risk_mode_ranking.py`

- [ ] Return the final decoder mode feature without changing trajectory outputs.
- [ ] Add a zero-initialized MLP over detached mode features and detached pooled risk memory.
- [ ] Train final residual logits with pairwise softplus loss and normal-scene zero calibration.
- [ ] Select inference mode from original logits plus residual when `use_risk_aware_cls=True`.

### Task 4: Configure And Observe 3.21

**Files:**
- Modify: `navsim/agents/diffusiondrive/transfuser_config.py`
- Modify: `navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml`
- Modify: `CHANGELOG.txt`

- [ ] Add conservative ranking, pair, and calibration settings.
- [ ] Enable risk-aware classification while keeping Temporal Risk Cross-Attention, hard gate, soft rescore, and brake-timing loss disabled.
- [ ] Record train/validation metrics through `transfuser_loss` so existing Lightning logging captures them under `train/*` and `val/*`.

### Task 5: Verify And Publish

**Files:**
- Test: `tests/test_diffusiondrive_risk_mode_ranking.py`

- [ ] Run local AST parsing and `git diff --check`; run PyTorch tests on the NAVSIM server.
- [ ] Commit the scoped experiment and push `change/20260712-risk-mode-ranking-3.21`.
- [ ] Provide a mandatory fresh-cache command and six-GPU DDP training command with baseline-equivalent global batch size.
