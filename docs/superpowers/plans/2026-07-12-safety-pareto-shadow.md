# Safety-Pareto Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make counterfactual eligibility require measurable trajectory-safety improvement instead of a tiny reduction in mixed total cost.

**Architecture:** Compute a separate clearance-plus-timing safety cost, derive candidate-versus-base gains, and build a Pareto eligibility mask before longitudinal/lateral selection. Keep map and comfort as bounded vetoes and keep all behavior in shadow mode.

**Tech Stack:** Python, PyTorch, unittest, NAVSIM/DiffusionDrive.

---

### Task 1: Add Safety-Pareto Configuration

**Files:**
- Modify: `navsim/agents/diffusiondrive/transfuser_config.py`

- [ ] Add minimum clearance gain, minimum safety-cost gain, map-regression tolerance, comfort-regression tolerance, and lateral safety-advantage settings.
- [ ] Keep all settings inference-only and checkpoint-compatible.

### Task 2: Replace Mixed-Cost Candidate Eligibility

**Files:**
- Modify: `navsim/agents/diffusiondrive/modules/risk_shadow.py`

- [ ] Add a focused helper that computes the Pareto mask and diagnostic gains.
- [ ] Require the Pareto mask for both longitudinal and lateral candidates.
- [ ] Rank candidates by safety-cost gain.
- [ ] Add CSV diagnostics while preserving previous diagnostic columns.

### Task 3: Add Regression Tests

**Files:**
- Modify: `tests/test_diffusiondrive_risk_attention_gate.py`

- [ ] Test rejection when total cost improves only through map cost.
- [ ] Test rejection when dynamic clearance improvement is below 0.10 m.
- [ ] Test acceptance for simultaneous clearance and safety-cost improvement.

### Task 4: Record and Verify

**Files:**
- Modify: `CHANGELOG.txt`

- [ ] Document 3.20.4 motivation, behavior, thresholds, and shadow-only status.
- [ ] Run Python syntax compilation and `git diff --check` locally.
- [ ] Leave PyTorch execution for the NAVSIM server environment.
