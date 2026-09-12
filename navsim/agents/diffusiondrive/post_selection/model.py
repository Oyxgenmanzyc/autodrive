"""Timing/strength queries attend to selected-path geometry and frozen context.

This model chooses a physically parameterized post-selection edit, not a K67
mode. No Bellman update, RL claim, candidate fusion or decoder intervention.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import ACTIONS


class PostSelectionRefiner(nn.Module):
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
        # Gain initially zero, so the strict positive-gain rule returns identity.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
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
        gain = gain - gain[:, :1]
        return {'gain': gain, 'unsafe_logits': raw[..., 1]}


def decide(output, margin=0., risk_limit=.25):
    """Deployment uses predictions only. Identity wins all ties/no-gain cases."""
    gain, risk = output['gain'], output['unsafe_logits'].sigmoid()
    eligible = (gain > margin) & (risk <= risk_limit) & torch.isfinite(gain) & torch.isfinite(risk)
    utility = torch.where(eligible, gain, torch.full_like(gain, -torch.inf))
    utility = utility.clone()
    utility[:, 0] = 0.
    return utility.argmax(-1)


def refinement_loss(output, labels, scores, direction, teacher):
    delta = scores-scores[:, :1]
    guarded = labels[..., [0, 1, 3, 4]]
    unsafe = (guarded < guarded[:, :1]-1e-7).any(-1) | (direction < direction[:, :1]-1e-7)
    # Hard harmful edits receive additional weight; unchanged/no-gain scenes
    # remain in the training set rather than supervising only rescues.
    weight = 1. + 4.*unsafe.float() + 4.*delta.abs()
    regression = (F.smooth_l1_loss(output['gain'], delta, reduction='none', beta=.05)*weight).mean()
    risk = (F.binary_cross_entropy_with_logits(output['unsafe_logits'], unsafe.float(), reduction='none')
            * (1.+4.*unsafe.float())).mean()
    ranking = F.cross_entropy(output['gain']/.05, teacher)
    return regression + .1*risk + .05*ranking
