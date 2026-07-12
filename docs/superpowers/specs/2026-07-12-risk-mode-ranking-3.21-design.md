# 3.21 Risk-Conditioned Mode Ranking Design

## Goal

Teach DiffusionDrive to rank an existing same-behavior trajectory higher when its braking starts at a better time in a GT high-risk following scene. The first version tests ranking capacity only; it does not add brake-timing regression or temporal cross-attention.

## Training Targets

The TargetBuilder selects the current GT front vehicle, locks its `track_token`, and follows that token for eight future frames. Each future box is transformed from that frame's ego coordinates through global coordinates into the current ego frame. Tracking stops when the token disappears; another vehicle is never spliced into the same risk segment.

The cache stores the future front boxes, validity mask, current ego speed, and a GT risk-scene flag based on current or delayed-0.5-second TTC crossing the speed-dependent T1 boundary. These targets are training-only and never enter inference features.

## Minimal Pair Mining

Pair mining uses all 20 final-layer trajectories with gradients detached. A positive/negative pair requires only:

- The two trajectories have similar lateral path and heading.
- The positive starts sustained braking at least one 0.5-second step earlier.
- The positive loses no more than 1.0 meter of path progress relative to the negative.
- The scene has a valid continuous GT front track and reaches the GT risk boundary.

TTC, clearance, DRAC, map, and comfort do not form a weighted pair score. They remain diagnostics for later analysis.

## Isolated Ranking Head

A zero-initialized MLP receives detached final mode features and detached pooled historical risk memory and predicts `risk_cls_delta [B,20]`. The original focal classification and trajectory regression remain unchanged. Pairwise softplus loss trains only this residual head. A small non-risk calibration term keeps the residual near zero on normal scenes.

Inference uses `final_cls = original_cls + risk_cls_delta`. No risk gate or soft cost re-ranking is enabled.

## Observability

Lightning logs ranking loss, risk-scene rate, valid-pair scene rate, pair count, original and adjusted pair accuracy, positive/negative brake onset, original ranks, and residual magnitude for both train and validation.

## Cache And Compatibility

The TargetBuilder uses a versioned cache name when ranking is enabled. A new training cache is mandatory because old caches do not contain future-front targets. Existing checkpoints load with `strict=False`; the zero-initialized ranking head starts output-neutral.

## Success Criteria

- No valid pair produces zero ranking loss because of missing target fields.
- No-risk scenes produce zero pair loss and calibrate the residual toward zero.
- Ranking gradients do not reach decoder trajectory features or historical risk memory.
- Initial inference is identical to baseline before the ranking head learns.
- Logs reveal whether all-20-mode timing pairs are sufficiently common to justify ranking rather than trajectory-generation supervision.
