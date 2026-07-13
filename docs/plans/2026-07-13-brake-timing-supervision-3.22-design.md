# 3.22 GT Soft Brake-Timing Supervision

## Hypothesis

The trajectory generator needs a direct but narrowly activated timing signal to distinguish timely sustained braking from late braking.

## Training Signal

For the mode selected by the original anchor-matching rule, derive speed and acceleration at 0.5-second intervals using the current ego speed as the initial condition. Convert acceleration to a soft brake profile:

`p_brake = sigmoid((a_threshold - acceleration) / temperature)`

Compare predicted and GT profiles with SmoothL1. A cumulative-max profile adds emphasis to the first activation time while the instantaneous term prevents the model from satisfying the objective with a single brake spike.

## Activation

The loss is active only when:

1. The same GT front-vehicle track is available for at least two future frames.
2. Current or delayed TTC is no more than one second outside the speed-dependent T1 boundary.
3. The GT ego trajectory contains at least two consecutive acceleration values below -0.5 m/s^2.

The preparation band begins before the T1 crossing and is not a fixed three-second braking rule.

## Isolation

- No risk gate or soft inference rescore.
- No Temporal Risk Cross-Attention.
- No clearance, jerk, or constant-deceleration pseudo-target loss.
- GT boxes are used only to build the training target context and never enter model features.

## Decision Metrics

The next revision should be based on active-scene coverage, onset absolute error, late/early ratio, predicted-vs-GT jerk, and PDM progress/safety changes. A smoothness loss is justified only if onset improves while jerk or acceleration oscillation clearly worsens.
