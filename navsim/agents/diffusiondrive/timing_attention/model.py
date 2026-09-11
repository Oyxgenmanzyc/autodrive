"""Step-specific risk conditioning; original K67 parameter names remain intact.

Design inspirations (not a reproduction): BridgeAD multi-step queries,
MTR geometry-conditioned queries, ControlNet zero output initialization.
See docs/experiment_3_1_05_3_timing_attention.md for scope and sources.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn


def history_validity(tokens):
    # At least two consecutive valid measurements, including the current one.
    valid = (tokens[..., -1] > .5) & torch.isfinite(tokens).all(-1)
    enabled = valid[:, -2:].all(-1)
    return valid, enabled


class TimingQueryEncoder(nn.Module):
    """Physical braking-demand queries, not historical planning queries.

    A constant-closing-speed forecast is an uncertain input prior, not a label
    or a safety rule. History is used ONLY to estimate current gap/closing speed.
    """
    def __init__(self, width=256, steps=8):
        super().__init__()
        self.register_buffer('future_seconds', .5 * torch.arange(1, steps + 1).float())
        self.embedding = nn.Sequential(nn.Linear(10, width), nn.SiLU(), nn.LayerNorm(width))

    def descriptors(self, tokens, ego_speed):
        _, enabled = history_validity(tokens)
        current = torch.nan_to_num(tokens[:, -1].float(), nan=0., posinf=0., neginf=0.)
        gap = current[:, 0].clamp(0, 32)[:, None]
        closing = current[:, 1].clamp(0, 30)[:, None]
        time = self.future_seconds[None].expand(len(tokens), -1)
        gap_if_wait = gap - closing * time
        # TTC up to 10s; negative time margin is kept instead of wrapped/clipped to safe.
        ttc = torch.where(closing > .1, (gap / closing.clamp_min(.1)).clamp_max(10.), torch.full_like(gap, 10.))
        required_decel = (closing.square() / (2 * gap_if_wait.clamp_min(.5))).clamp_max(6.)
        broadcast = lambda value: value.expand_as(time)
        descriptors = torch.stack([
            time / 4., broadcast(gap / 32.), broadcast(closing / 15.),
            broadcast(ego_speed.float().clamp(0, 60)[:, None] / 30.),
            broadcast(current[:, 4:5].clamp(-12, 12) / 6.),
            gap_if_wait.clamp(-32, 32) / 32.,
            (gap_if_wait / ego_speed.float()[:, None].clamp_min(.5)).clamp(-10, 10) / 10.,
            (ttc - time).clamp(-10, 10) / 10., required_decel / 6.,
            broadcast(current[:, 9:10].clamp(-5, 5) / 5.),
        ], -1).clamp(-2, 2)
        return descriptors, enabled

    def forward(self, tokens, ego_speed):
        descriptors, enabled = self.descriptors(tokens, ego_speed)
        return self.embedding(descriptors), enabled


def geometry_features(poses, ego_speed, dt=.5):
    xy = poses[..., :2].detach().float()
    displacement = torch.diff(torch.cat([torch.zeros_like(xy[..., :1, :]), xy], -2), dim=-2)
    speed = torch.linalg.vector_norm(displacement, dim=-1) / dt
    initial = ego_speed[:, None, None].expand(-1, speed.shape[1], 1)
    accel = (speed - torch.cat([initial, speed[..., :-1]], -1)) / dt
    scale = xy.new_tensor([60., 30.])
    return torch.cat([(xy / scale).clamp(-2, 2),
                      (speed / 30.).clamp(0, 2)[..., None],
                      (accel / 6.).clamp(-2, 2)[..., None]], -1)


class StepTimingAttention(nn.Module):
    def __init__(self, width=256, heads=8, steps=8, residual_cap=.1):
        super().__init__()
        self.steps, self.residual_cap = steps, residual_cap
        self.content_norm = nn.LayerNorm(width)
        self.geometry = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.LayerNorm(width))
        # Physical future time, NOT the diffusion denoising timestep.
        time = torch.arange(1, steps + 1).float() / steps
        self.register_buffer('future_time', torch.stack([time, torch.sin(math.pi*time), torch.cos(math.pi*time)], -1))
        self.time_embedding = nn.Linear(3, width, bias=False)
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        # Keep step order; a plain mean would erase some timing distinctions.
        self.output = nn.Linear(steps * width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_ratio = None

    def forward(self, feature, clean_poses, timing_queries, enabled, ego_speed):
        batch, modes, width = feature.shape
        with torch.autocast(device_type=feature.device.type, enabled=False):
            x = feature.float()
            # Q explicitly encodes braking demand at each future time.
            # K/V encode THIS candidate at each future time. Candidates do not mix.
            q = self.query_norm(timing_queries.float() + self.time_embedding(self.future_time)[None])
            q = q[:, None].expand(-1, modes, -1, -1).reshape(batch*modes, self.steps, width)
            candidate = self.content_norm(x)[:, :, None] + self.geometry(geometry_features(clean_poses, ego_speed))
            key = self.query_norm((candidate + self.time_embedding(self.future_time)[None, None]) / math.sqrt(3.))
            value = self.content_norm(candidate / math.sqrt(2.))
            attended = self.attention(q, key.reshape(batch*modes, self.steps, width),
                                      value.reshape(batch*modes, self.steps, width), need_weights=False)[0]
            ordered = self.output_norm(attended).reshape(batch, modes, self.steps * width)
            rms = x.detach().square().mean(-1, keepdim=True).sqrt()
            delta = self.residual_cap * rms * torch.tanh(self.output(ordered))
            delta = torch.where(enabled[:, None, None], delta, torch.zeros_like(delta))
            self.last_ratio = (delta.detach().square().mean(-1).sqrt() / rms.squeeze(-1).clamp_min(1e-8))
            return (x + delta).to(feature.dtype)


class TimingDecoderLayer(nn.Module):
    """Original decoder operations plus one conditional regression input.

    Shares original children without renaming checkpoint keys. All original
    weights stay frozen. Explicit forward avoids fragile persistent hooks.
    """
    def __init__(self, original, residual_cap):
        super().__init__()
        for name, child in original.named_children():
            self.add_module(name, child)
        width = self.task_decoder.embed_dims
        self.timing_adapter = StepTimingAttention(width, self.cross_ego_attention.num_heads,
                                                 self.task_decoder.ego_fut_ts, residual_cap)
        self.runtime_timing = None

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        x = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        x = self.norm1(x + self.dropout(self.cross_agent_attention(x, agents_query, agents_query)[0]))
        x = self.norm2(x + self.dropout1(self.cross_ego_attention(x, ego_query, ego_query)[0]))
        x = self.norm3(self.ffn(x))
        x = self.time_modulation(x, time_embed, global_cond=None, global_img=global_img)
        base_reg, cls = self.task_decoder(x)
        if self.runtime_timing is None:
            raise RuntimeError('Timing decoder requires explicit history binding')
        timing_queries, enabled, ego_speed = self.runtime_timing
        # Query geometry comes from a provisional CLEAN prediction, not raw noise.
        clean = base_reg.detach().clone()
        clean[..., :2] += noisy_traj_points.detach()
        conditioned = self.timing_adapter(x, clean, timing_queries, enabled, ego_speed)
        reg = self.task_decoder.plan_reg_branch(conditioned).reshape_as(base_reg)
        reg = torch.cat([reg[..., :2] + noisy_traj_points,
                         reg[..., 2:].tanh() * math.pi], dim=-1)
        return reg, cls


def mutable_parameter(name):
    return name.startswith('_timing_query.') or '.timing_adapter.' in name


def install_attention(generator, residual_cap=.1):
    if not 0 < residual_cap <= .25:
        raise ValueError('residual_cap must be in (0,.25]')
    generator.requires_grad_(False)
    head = generator._trajectory_head
    if hasattr(generator, '_timing_query'):
        raise ValueError('Attention already installed')
    generator._timing_query = TimingQueryEncoder(head._d_model, head._num_poses)
    head.diff_decoder.layers = nn.ModuleList([TimingDecoderLayer(layer, residual_cap) for layer in head.diff_decoder.layers])
    generator.eval()
    return generator


@contextmanager
def bind_timing_inputs(generator, history, ego_speed):
    with torch.autocast(device_type=history.device.type, enabled=False):
        queries, enabled = generator._timing_query(history.float(), ego_speed)
    for layer in generator._trajectory_head.diff_decoder.layers:
        if layer.runtime_timing is not None:
            raise RuntimeError('Nested history binding is not supported')
        layer.runtime_timing = (queries, enabled, ego_speed.float())
    try:
        yield
    finally:
        for layer in generator._trajectory_head.diff_decoder.layers:
            layer.runtime_timing = None
