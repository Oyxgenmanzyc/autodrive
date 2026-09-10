# 3.22 timing representation provenance

Adapted from the user's existing experiment `best_3.22.zip`, specifically
`best_3.22/navsim/agents/diffusiondrive/modules/risk_brake_timing.py`.

Archive SHA256: `dbc5857d0eedc3d71c2cf4d7538643c019547dae2be0a94394e7b26ab276881f`.

The supplied project is MIT licensed, Copyright (c) 2024 HUST Vision Lab.
Its MIT text is preserved in the repository root `LICENSE`.

Reused concepts/expressions: FP32 smoothed displacement norm, step dynamics,
sigmoid soft braking profile and cumulative onset representation. Modified:
candidate/base pair representation for a selector, explicit unknown-first-step
mask, soft consecutive-braking product, no GT target/context in forward.

The original GT-matched generator loss, future front-track activation, historical
LiDAR risk encoder and inference gates are not part of this adaptation.
