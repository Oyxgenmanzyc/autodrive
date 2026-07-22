# 3.22 Endpoint-Conditioned Risk-Aware Temporal Transport

## Scope

This experiment changes only 3.22 longitudinal timing supervision. It preserves
the original 20 trajectory modes, the primary trajectory loss, and the
inference-only longitudinal safety shield.

## Teacher target

For the GT-matched mode, the GT trajectory is treated as a fixed polyline
`r(s)`. Future GT boxes for the current front track are projected onto this
polyline to construct an ST upper bound. A minimum-jerk candidate search keeps
the largest terminal progress that remains feasible under that bound and the
configured speed, acceleration, and jerk limits. Re-sampling `r(s)` with the
selected progress profile yields the transport target.

GT boxes and track tokens are used only by TargetBuilder during cache creation.
FeatureBuilder and inference continue to use only the original sensor inputs.

## Loss

The auxiliary loss applies only to valid transport targets and contains:

- continuous progress alignment to the transport target;
- terminal-progress alignment, preserving final intent when feasible;
- a penalty for crossing the GT-derived ST upper bound;
- acceleration and jerk alignment to the minimum-jerk target.

The existing brake-onset metrics remain diagnostics rather than the primary
supervision objective.

## Validation

The server run must create a new cache and inspect active rate, teacher shift,
progress MAE, terminal error, ST violation, early/late timing rate, total loss,
and PDM. The safety shield should remain disabled during training and enabled
only in the later inference ablation.
