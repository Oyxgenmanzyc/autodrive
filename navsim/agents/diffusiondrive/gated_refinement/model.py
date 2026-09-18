"""Timing/strength queries attend to selected-path geometry and frozen context.

This model chooses a physically parameterized post-selection edit, not a K67
mode. No Bellman update, RL claim, candidate fusion or decoder intervention.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ..post_selection.geometry import ACTIONS


class GatedRefiner(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, width))
        self.geometry = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width))
        self.bev = nn.Conv2d(256, width, 1)
        self.agent = nn.Linear(256, width)
        self.ego = nn.Linear(256, width)
        self.cross = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width*8, width), nn.ReLU(), nn.Linear(width, 2))
        # Zero initial action scores; uncalibrated policy remains disabled.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.gate = nn.Sequential(nn.Linear(width*8, width), nn.ReLU(), nn.Linear(width, 1))
        self.reason = nn.Linear(width*8, 5)
        self.regret = nn.Linear(width*8, 1)
        self.register_buffer('actions', torch.tensor(ACTIONS, dtype=torch.float32), persistent=False)

    def forward(self, context, variants):
        variants = variants.float()
        batch, actions, steps, _ = variants.shape
        time = torch.arange(1, 9, device=variants.device, dtype=torch.float32)*.5
        descriptors = torch.cat([
            time[None, :, None].expand(actions, -1, -1)/4.,
            (self.actions / self.actions.new_tensor([4., 2., 1.]))[:, None].expand(-1, 8, -1),
        ], -1)
        q = self.query(descriptors).reshape(1, actions*8, -1).expand(batch, -1, -1)
        xy = variants[..., :2]
        previous = torch.cat([torch.zeros_like(xy[:, :, :1]), xy[:, :, :-1]], 2)
        speed = (xy-previous).norm(dim=-1)/.5
        accel = torch.diff(speed, dim=2, prepend=speed[:, :, :1])/.5
        # Time and physical units are fixed; no data-dependent scale fitting.
        g = torch.cat([xy/xy.new_tensor([60., 30.]),
                       variants[..., 2:3].sin(), variants[..., 2:3].cos(),
                       speed[..., None]/30., accel[..., None]/6.,
                       time[None, None, :, None].expand(batch, actions, -1, -1)/4.,
                       (speed-speed[:, :1])[..., None]/10.], -1).clamp(-5., 5.)
        geometry = self.geometry(g)
        bev = context['bev'].detach().half().float()
        # Same coordinate convention as existing PCS cross-BEV sampling.
        grid = (xy/32.)[..., [1, 0]]
        local = F.grid_sample(self.bev(bev), grid, align_corners=False, padding_mode='zeros')
        local = local.permute(0, 2, 3, 1)
        q = q.reshape(batch, actions, 8, -1) + geometry + local
        # Each timing action attends to the selected original trajectory and
        # scene agents/ego. No feature mixing across the old K67 candidate bank.
        memory = torch.cat([geometry[:, 0],
                            self.agent(context['agents'].detach().half().float()),
                            self.ego(context['ego'].detach().half().float())], 1)
        flat = q.flatten(1, 2)
        attended, _ = self.cross(flat, memory, memory, need_weights=False)
        features = self.norm(flat+attended).reshape(batch, actions, -1)
        raw = self.head(features).float()
        gain = raw[..., 0].tanh()
        # Identity is not a competing action. Gate decides whether to edit.
        return {'gain': gain, 'unsafe_logits': raw[..., 1],
                'gate_logits': self.gate(features[:, 0]).squeeze(-1),
                'reason_logits': self.reason(features[:, 0]),
                'regret': self.regret(features[:, 0]).sigmoid().squeeze(-1)}


def decide(output, gate_threshold=.5, risk_limit=.25):
    """Gate opens first; rank only nonidentity actions, even if all gains < 0."""
    gain = output['gain'][:, 1:]
    risk = output['unsafe_logits'][:, 1:].sigmoid()
    allowed = (risk <= risk_limit) & torch.isfinite(gain) & torch.isfinite(risk)
    best = gain.masked_fill(~allowed, -torch.inf).argmax(-1) + 1
    gate = output['gate_logits'].sigmoid()
    opened = (gate >= gate_threshold) & torch.isfinite(gate) & allowed.any(-1)
    return torch.where(opened, best, torch.zeros_like(best))


def loss_terms(output, batch, stage='joint'):
    target = batch['need'].float()
    # Balanced sampler already corrects exposure; do not multiply by huge pos_weight.
    gate = F.binary_cross_entropy_with_logits(output['gate_logits'], target)
    grouped = output['gate_logits'].reshape(-1, 4)
    if not (batch['need'].reshape(-1, 4)[:, 0].all()
            and (~batch['need'].reshape(-1, 4)[:, 1:]).all()):
        raise ValueError('Expected positive/hard-negative/hard-negative/random groups')
    pair = F.softplus(.5 - grouped[:, :1] + grouped[:, 1:3]).mean()
    reason = F.binary_cross_entropy_with_logits(output['reason_logits'], batch['reasons'])
    regret = F.smooth_l1_loss(output['regret'], batch['regret'], beta=.05)
    terms = dict(gate=gate, pair=pair, reason=reason, regret=regret)
    total = gate + .2*pair + .1*reason + .1*regret
    if stage == 'joint':
        pos = batch['need'].bool()
        if not pos.any():
            raise ValueError('No positives in paired training batch')
        log_prob = F.log_softmax(output['gain'][pos, 1:]/.05, -1)
        action = -(batch['action_target'][pos]*log_prob).sum(-1).mean()
        delta = batch['scores']-batch['scores'][:, :1]
        gain = F.smooth_l1_loss(output['gain'][pos, 1:], delta[pos, 1:], beta=.05)
        risk = F.binary_cross_entropy_with_logits(output['unsafe_logits'][:, 1:],
                                                  batch['unsafe'][:, 1:].float())
        total = total + .1*action + gain + .1*risk
        terms.update(action=action, gain=gain, risk=risk)
    return total, terms
