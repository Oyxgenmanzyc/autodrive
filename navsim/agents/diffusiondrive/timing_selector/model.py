"""A small selector trained independently from the generator and PCS ranker."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.diffusiondrive.timing_oracle.geometry import ACTIONS


def action_descriptors():
    family_names = ("identity", "extra_brake", "frontload_progress", "backload_progress")
    rows = []
    for family, value, strength, ramp in ACTIONS:
        one_hot = [float(family == name) for name in family_names]
        rows.append(one_hot + [value / 1.5, strength / 1.5, ramp, float(family != "identity")])
    return torch.tensor(np.asarray(rows, dtype=np.float32))


class BidirectionalTimingSelector(nn.Module):
    """Score 31 timing actions from frozen deployable scene/trajectory features."""
    def __init__(self, width=128):
        super().__init__()
        self.settings = {"width": width, "actions": len(ACTIONS)}
        self.feature_norm = nn.LayerNorm(512)
        self.scene = nn.Sequential(nn.Linear(551, width), nn.ReLU(), nn.LayerNorm(width))
        self.action = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.LayerNorm(width))
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        self.register_buffer("descriptors", action_descriptors(), persistent=True)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        pose = batch["proposal"].detach().float()
        geometry = torch.cat([
            pose[..., :2] / pose.new_tensor([60.0, 30.0]),
            pose[..., 2:3].sin(), pose[..., 2:3].cos(),
        ], -1).flatten(-2).clamp(-5.0, 5.0)
        inputs = torch.cat([
            self.feature_norm(batch["feature"].detach().float()),
            batch["subscores"].detach().float(),
            batch["pcs_score"].detach().float()[:, None],
            batch["base_probability"].detach().float()[:, None],
            geometry,
        ], -1)
        scene = self.scene(inputs)[:, None]
        actions = self.action(self.descriptors)[None].expand(len(pose), -1, -1)
        tokens = scene + actions
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        logits = self.head(self.norm(tokens + attended)).squeeze(-1).float()
        return logits - logits[:, :1]


def scalar_targets(labels, scores, direction):
    """Encode safety and PDM into one target used by the sole ranking loss."""
    labels = labels.detach().float()
    scores = scores.detach().float()
    direction = direction.detach().float()
    safe = (labels[..., [0, 1, 3, 4]] >= labels[:, :1, [0, 1, 3, 4]] - 1e-7).all(-1)
    safe &= direction >= direction[:, :1] - 1e-7
    target = scores - scores[:, :1]
    return torch.where(safe, target, target.new_full(target.shape, -1.0))


def timing_ranking_loss(logits, labels, scores, direction, min_gap=0.005,
                       temperature=0.05, cost_cap=5.0):
    """One cost-weighted pairwise objective; there are no auxiliary losses."""
    target = scalar_targets(labels, scores, direction)
    gap = target[:, :, None] - target[:, None, :]
    valid = gap >= min_gap
    cost = (gap / 0.05).clamp(0.25, cost_cap)
    predicted = logits[:, :, None] - logits[:, None, :]
    pair_loss = F.softplus(-predicted / temperature) * cost * valid
    counts = valid.sum((1, 2))
    per_scene = pair_loss.sum((1, 2)) / counts.clamp_min(1)
    active = counts > 0
    loss = per_scene.sum() / active.sum().clamp_min(1)
    return loss, {"active_scenes": active.sum(), "pairs": valid.sum()}
