"""Dimensionless timing/strength supervision, GT used only as targets."""
import torch
import torch.nn.functional as F
from navsim.agents.diffusiondrive.generator_timing.risk_brake_timing import (
    _brake_timing_observations, _trajectory_dynamics,
)
from navsim.agents.diffusiondrive.generator_timing.data import timing_config


def timing_strength_loss(proposals, targets, anchors, enabled, ego_speed):
    proposals = proposals.float()
    target = targets['trajectory'].float()
    context = targets['brake_timing_context'].float().clone()
    # Actual current speed is available even if there is no GT front vehicle.
    context[:, 0] = ego_speed.float()
    distance = torch.linalg.vector_norm(target[:, None, :, :2] - anchors[None].float(), dim=-1).mean(-1)
    selected = proposals[torch.arange(len(proposals), device=proposals.device), distance.argmin(-1)]
    obs = _brake_timing_observations(selected, target, context, timing_config())
    active = obs['active'] & enabled
    pred = _trajectory_dynamics(selected, ego_speed, .5)
    gt = _trajectory_dynamics(target, ego_speed, .5)
    # Full acceleration profile penalizes weak AND excessive braking; unlike
    # sigmoid braking probability this does not saturate at large deceleration.
    accel = F.smooth_l1_loss(pred['acceleration']/3., gt['acceleration']/3., reduction='none').mean(-1)
    speed = F.smooth_l1_loss(pred['speed']/10., gt['speed']/10., reduction='none').mean(-1)
    # Penalize jerk only beyond the GT magnitude plus 2m/s3 slack.
    jerk = ((pred['jerk'].abs() - gt['jerk'].abs() - 2.).clamp_min(0.)/5.).mean(-1)
    def local_sum(value):
        return (value * active.float()).sum()
    # Keep sums differentiable; DDP normalization is applied once by the caller.
    sums = {'timing': local_sum(obs['raw_loss']), 'strength': local_sum(.5*accel + .5*speed),
            'jerk': local_sum(jerk)}
    diagnostics = {
        'active_count': active.float().sum(), 'scene_count': active.new_tensor(len(active), dtype=torch.float32),
        'onset_abs_error_sum': local_sum(obs['onset_abs_error_s']).detach(),
        'late_count': local_sum(obs['late_rate']).detach(),
        'accel_mae_sum': local_sum((pred['acceleration']-gt['acceleration']).abs().mean(-1)).detach(),
        'speed_mae_sum': local_sum((pred['speed']-gt['speed']).abs().mean(-1)).detach(),
        'jerk_abs_sum': local_sum(pred['jerk'].abs().mean(-1)).detach(),
    }
    return sums, diagnostics


def distributed_active_mean(value_sum, count):
    count = count.detach().clone()
    world = 1
    if torch.distributed.is_initialized():
        world = torch.distributed.get_world_size()
        torch.distributed.all_reduce(count)
    return value_sum * world / count.clamp_min(1.)
