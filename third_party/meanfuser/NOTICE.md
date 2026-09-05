# MeanFuser attribution

SPR (Scene-conditioned Proposal Reconstruction) is an adaptation of MeanFuser's
Adaptive Reconstruction Module (ARM). Renaming identifies the local integration;
it does not make the upstream architecture an original contribution.

- Project: https://github.com/wjl2244/MeanFuser
- Revision: `8de8ba6244834645192e318dcc437d124cfd6872`
- Retrieved: 2026-09-05
- Original file SHA-256:
  - arm_model.py: `d6242c79b56a6a047612c527f4ae259fa8fd63b9f8f4411c0264e8026045827b`
  - utils.py: `22fc40eacf051846b23d830f16a2fb0729e08600d6492f3c90424c3f5be7fee5`
- License: Apache-2.0; the complete upstream license is included in [LICENSE](LICENSE).
- Paper: *Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction
  via MeanFlow for End-to-End Autonomous Driving*, Junli Wang et al., CVPR 2026.

Source mapping:

| Upstream source at the revision above | Local adaptation |
| --- | --- |
| `navsim/agents/meanfuser/arm_model.py` | `navsim/agents/diffusiondrive/modules/scene_proposal_reconstruction.py` |
| `navsim/agents/meanfuser/utils.py` (trajectory constants and transforms only) | `navsim/agents/diffusiondrive/modules/proposal_trajectory_utils.py` |

Retained: trajectory encoder, three TransformerDecoder layers, normalization
before the outer residual addition, reconstruction MLP widths/activations,
weight initialization, normalized displacement statistics, sin/cos heading
representation, cumulative decoding, and mean L1 reconstruction supervision.

Changes: local class/loss names and imports; proposal count from the configured
DiffusionDrive bank; an explicit 8-pose horizon check; absolute XY/yaw proposal
adapter; ego plus agent queries as memory; default detached proposal coordinates;
FP32 transforms and loss under autocast; checkpoint/bank validation.
Integration/configuration, diagnostics, tests and launchers are local additions.

The original generated candidates are sampled with no gradients. This port
reuses the final existing DiffusionDrive decoder output in each train/test pass
and detaches it by default; it does not reproduce the upstream MeanFlow sampler
or introduce another inference rollout during training.

No upstream NOTICE file was present in the repository root at this revision.
This attribution accompanies the adapted portions; it does not change the
licensing of unrelated repository files.
