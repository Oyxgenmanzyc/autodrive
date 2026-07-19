# 3.22.2 Lightweight Anticipatory Brake-Timing Supervision

## Finding

The 3.22 brake objective learned its labels, but its old hard gate selected only 1.595% of validation scenes and almost all selected GT trajectories were already braking at the first predicted step. The historical risk encoder was also disconnected from the trajectory decoder because Temporal Risk Cross-Attention was disabled.

## Change

3.22.2 keeps the successful GT soft brake-profile objective but moves its activation earlier. Current and delayed TTC/THW are projected over a short horizon from their observed 0.5-second trend. Supervision remains conditional on continuous front tracking and GT sustained braking, so ordinary driving is not pushed toward conservative deceleration.

The historical risk encoder, memory auxiliary objective, and historical feature input are disabled. They add no trajectory information in this experiment. PDM evaluation records the selected trajectory's braking dynamics directly instead of emitting an always-zero counterfactual field.

## Invariants

- 20 trajectory modes and anchors are unchanged.
- Original classification, trajectory, detection, and BEV losses are unchanged.
- GT boxes remain target-builder-only training data.
- Inference selection remains the original classifier argmax.
- No hard risk gate, risk rescore, counterfactual trajectory, or jerk penalty is introduced.
