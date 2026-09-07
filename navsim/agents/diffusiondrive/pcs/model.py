"""PDM Candidate Scoring (PCS): a single-stage adaptation of DiffusionDriveV2.

Architecture/metric heads: hustvl/DiffusionDriveV2 @ 1cd12a1.
See v2_blocks.py and LICENSE_DiffusionDriveV2.txt for attribution.
No candidate augmentation, trajectory modification, fine scorer, or RL.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.diffusiondrive.modules.blocks import (
    gen_sineembed_for_position, linear_relu_ln,
)
from .v2_blocks import ScorerTransformerDecoderLayer, gen_sineembed_for_position_1d

METRIC_NAMES = (
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort",
)


def combine_subscores(values):
    """NAVSIM v1 PDMS weights; values have final axis NC, DAC, EP, TTC, C."""
    return values[..., 0] * values[..., 1] * (
        5 * values[..., 2] + 5 * values[..., 3] + 2 * values[..., 4]
    ) / 12


class PDMCSHead(nn.Module):
    def __init__(self, num_poses=8, lidar_max_x=32.0, lidar_max_y=32.0):
        super().__init__()
        self.settings = dict(
            num_poses=num_poses, lidar_max_x=lidar_max_x, lidar_max_y=lidar_max_y,
        )
        config = SimpleNamespace(**self.settings)
        self.encoder = nn.Sequential(
            *linear_relu_ln(256, 1, 1, num_poses * 128),
            nn.Linear(256, 512),
        )
        self.decoder = ScorerTransformerDecoderLayer(num_poses, 256, 1024, config)
        self.heads = nn.ModuleList([
            nn.Sequential(*linear_relu_ln(512, 2 if i == 2 else 1, 2), nn.Linear(512, 1))
            for i in range(5)
        ])

    def forward(self, context):
        # Cache and live inference both use FP16 storage rounded back to FP32.
        # Geometry/logits retain FP32. The generator receives no scorer gradients.
        poses = context["proposals"].detach().float()
        xy = poses[..., :2]
        position = gen_sineembed_for_position(xy, hidden_dim=64).flatten(-2)
        heading = gen_sineembed_for_position_1d(poses[..., 2], hidden_dim=32).flatten(-2)
        features = self.encoder(torch.cat([position, heading], dim=-1))
        bev = context["bev"].detach().to(torch.float16).float()
        agents = context["agents"].detach().to(torch.float16).float()
        ego = context["ego"].detach().to(torch.float16).float()
        features = self.decoder(features, xy, bev, bev.shape[-2:], agents, ego, None, None)
        logits = torch.cat([head(features) for head in self.heads], dim=-1)
        # Composite scores/losses computed in FP32 even under AMP.
        subscores = logits.float().sigmoid()
        return {
            "logits": logits.float(), "subscores": subscores,
            "scores": combine_subscores(subscores),
        }

    @staticmethod
    def loss(output, labels):
        # Keep genuine soft PDM labels, including NC=0.5. No proxy risk labels.
        # No EP-only ranking term: first isolate the transplanted metric supervision.
        losses = F.binary_cross_entropy_with_logits(output["logits"], labels, reduction="none")
        return losses.mean(dim=(0, 1)).sum()


def select_candidates(context, scores):
    selected = scores.argmax(-1)
    base = context["base_logits"].argmax(-1)
    rows = torch.arange(len(selected), device=selected.device)
    return {
        "trajectory": context["proposals"][rows, selected],
        "selector_trajectory": context["proposals"][rows, base],
        "selected_mode": selected, "base_mode": base,
    }

