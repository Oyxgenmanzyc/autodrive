"""Timing-conditioned sampling inside the frozen DiffusionDrive DDIM decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
from .geometry import FAMILIES, STEPS, seed_anchors


class TimingModeGenerator(nn.Module):
    """Train only adapters. Original K67 predictions remain byte-for-byte frozen.

    The branch starts from 134 time-indexed anchor seeds and runs both original
    decoder layers at each of the two DDIM steps.  A scene-conditioned timing
    vector derived from all eight displacement intervals enters *every* layer
    and an eight-step residual head. No final-output trajectory warp is used.
    """
    def __init__(self, frozen_head, width=256):
        super().__init__()
        self.head = frozen_head.requires_grad_(False).eval()
        anchors = self.head.plan_anchor.detach()
        if anchors.shape != (67, STEPS, 2) or width != 256:
            raise ValueError('This experiment requires the original K67/8-step/256-d head')
        self.register_buffer('mode_anchors', seed_anchors(anchors))
        self.condition = nn.Sequential(
            nn.Linear(2 + STEPS * 3 + 256 * 2, width), nn.LayerNorm(width),
            nn.ReLU(), nn.Linear(width, width),
        )
        self.layer_adapters = nn.ModuleList(nn.Linear(width, width) for _ in self.head.diff_decoder.layers)
        self.pose_adapters = nn.ModuleList(nn.Linear(width, STEPS * 2) for _ in self.head.diff_decoder.layers)
        for layer in (*self.layer_adapters, *self.pose_adapters):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        self.head.eval()
        return self

    def adapter_state(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith('head.')}

    def load_adapter_state(self, state):
        expected = set(self.adapter_state())
        if set(state) != expected:
            raise ValueError('Timing adapter checkpoint keys mismatch')
        self.load_state_dict(state, strict=False)

    def forward(self, context, noise=None):
        head = self.head
        device = self.mode_anchors.device
        bev = context['bev'].to(device=device, dtype=torch.float32)
        agents = context['agents'].to(device=device, dtype=torch.float32)
        ego = context['ego'].to(device=device, dtype=torch.float32)
        batch = bev.shape[0]
        anchors = self.mode_anchors.unsqueeze(0).expand(batch, -1, -1, -1)
        k = anchors.shape[1]
        # Mode identity and every temporal interval survive the conditioner.
        delta = anchors - F.pad(anchors[:, :, :-1], (0, 0, 1, 0))
        interval = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), dim=-1)
        codes = F.one_hot(torch.arange(len(FAMILIES), device=device).repeat(67), 2)
        codes = codes.to(anchors.dtype).unsqueeze(0).expand(batch, -1, -1)
        scene = torch.cat((ego.mean(1), agents.mean(1)), -1)
        scene = scene[:, None].expand(-1, k, -1)
        condition = self.condition(torch.cat((codes, interval.flatten(-2), scene), -1))

        scheduler = head.diffusion_scheduler
        scheduler.set_timesteps(1000, device=device)
        image = head.norm_odo(anchors)
        if noise is None:
            noise = torch.randn_like(image)
        if noise.shape != image.shape:
            raise ValueError('Timing noise shape mismatch')
        image = scheduler.add_noise(image, noise, torch.full((batch,), 8, device=device, dtype=torch.long))
        for timestep in (10, 0):
            noisy = head.denorm_odo(image.clamp(-1., 1.))
            positional = gen_sineembed_for_position(noisy, hidden_dim=64).flatten(-2)
            feature = head.plan_anchor_encoder(positional).view(batch, k, -1)
            clock = head.time_mlp(torch.full((batch,), timestep, device=device, dtype=torch.long))[:, None]
            points = noisy
            for layer, feature_adapter, pose_adapter in zip(
                    head.diff_decoder.layers, self.layer_adapters, self.pose_adapters):
                feature = feature + feature_adapter(condition)
                poses, _ = layer(feature, points, bev, bev.shape[-2:], agents, ego, clock, None, None)
                poses = torch.cat((poses[..., :2] + pose_adapter(condition).view(batch, k, STEPS, 2),
                                   poses[..., 2:]), -1)
                # This side branch needs gradients through its first layer's
                # trajectory output. The untouched K67 branch keeps its detach.
                points = poses[..., :2]
            image = scheduler.step(head.norm_odo(poses[..., :2]), timestep, image).prev_sample
        if not torch.isfinite(poses).all():
            raise ValueError('Nonfinite timing-mode proposals')
        return poses


def timing_loss(predicted, targets, valid, weight):
    """Mode-specific motion targets; never regress all modes to the same GT."""
    if predicted.shape != targets.shape or predicted.ndim != 5 or predicted.shape[2:4] != (2, STEPS):
        raise ValueError('Expected [B,K,2,8,3] trajectories')
    mask = valid.float() * weight.float()
    if mask.sum() == 0:
        raise ValueError('Batch has no safe timing teacher')
    displacement = predicted[..., :2] - targets[..., :2]
    velocity = torch.diff(F.pad(predicted[..., :2], (0, 0, 1, 0)), dim=-2)
    target_velocity = torch.diff(F.pad(targets[..., :2], (0, 0, 1, 0)), dim=-2)
    position_loss = F.smooth_l1_loss(displacement, torch.zeros_like(displacement), reduction='none').mean((-1, -2))
    velocity_loss = F.smooth_l1_loss(velocity, target_velocity, reduction='none').mean((-1, -2))
    angle = 1. - torch.cos(predicted[..., 2] - targets[..., 2])
    per_mode = position_loss + .5 * velocity_loss + .1 * angle.mean(-1)
    return (per_mode * mask).sum() / mask.sum()
