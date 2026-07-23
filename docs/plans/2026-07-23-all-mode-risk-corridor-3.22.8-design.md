# 3.22.8 All-Mode Risk-Corridor Loss

## Problem

The 3.22.7 endpoint-transport loss reached only 0.779% of training scenes,
supervised one GT-matched mode, and acted only on the final decoder layer.
It reduced late braking but drove the active trajectories toward systematic
early braking. The finally selected classifier mode was not necessarily the
mode receiving the timing gradient.

## Objective

Keep 3.22 as a loss-only contribution. Every candidate mode that still follows
the tracked front vehicle should learn a safe longitudinal response, while a
mode that moves laterally away from that vehicle should preserve its geometry
and progress.

## Loss

For each predicted mode and future step, transform the training-only tracked
front box into the current ego frame. Use the mode's detached local tangent to
measure longitudinal clearance and lateral overlap.

Only modes with a reliable multi-step overlap and an approaching preparation
zone are active. Their loss is:

`L = wsafe Lsafe + wlate Llate + wearly Learly + wdec Ldec + wjerk Ljerk`

- `Lsafe`: clearance below the speed-dependent safe gap.
- `Llate`: braking is weaker than the kinematically required deceleration
  inside the preparation zone.
- `Learly`: braking is stronger than the allowed threshold before entering the
  preparation zone, unless the GT trajectory is already braking for another
  scene constraint.
- `Ldec`: deceleration exceeds the physical bound.
- `Ljerk`: jerk exceeds the comfort-free region.

The loss is averaged over active modes and decoder layers. It is evaluated at
random diffusion timesteps during training, so the shared denoiser receives
the constraint throughout the diffusion-time distribution.

## Boundaries

- Future GT boxes are TargetBuilder-only supervision and never model features.
- No new inference module, mode ranker, gate, or trajectory post-processing.
- No change to 20 anchors, classifier logits, or diffusion sampling.
- Lateral escape modes are not forced to imitate the GT path or brake profile.

## Required Diagnostics

- active scene rate and active mode rate
- active modes per active scene
- minimum clearance and safety violation
- early- and late-braking excess
- required deceleration
- onset early/late rates

## Cache

The new target fields require a fresh cache named
`transfuser_target_all_mode_risk_corridor_v1`.
