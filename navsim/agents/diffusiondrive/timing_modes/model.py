"""One separately trained value model for spatial-times-timing candidates."""
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import MODE_NAMES


def dynamics(trajectories):
    origin = torch.zeros_like(trajectories[..., :1, :2])
    xy = torch.cat((origin, trajectories[..., :2]), dim=-2)
    speed = torch.linalg.vector_norm(xy[..., 1:, :]-xy[..., :-1, :], dim=-1) / .5
    acceleration = (speed[..., 1:]-speed[..., :-1]) / .5
    return speed.clamp(0., 40.) / 20., acceleration.clamp(-10., 10.) / 10.


class TimingModeValue(nn.Module):
    """Use deployable frozen PCS risk proxies and full timing geometry only.

    PCS subscores include its predicted TTC and collision risk. No official
    future actor tracks or PDM labels are read by this forward pass.
    """
    def __init__(self, width=128):
        super().__init__()
        self.settings = dict(width=width, timing_modes=MODE_NAMES)
        self.feature_norm = nn.LayerNorm(512)
        # 512 latent + 5 PCS risks + PCS/rank/base scores + two 32D poses
        # + 8 speeds + 7 accelerations + one-hot timing action.
        self.embed = nn.Sequential(nn.Linear(603, width), nn.ReLU(), nn.LayerNorm(width))
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        trajectory = batch['trajectories'].detach().float()
        original = batch['original'].detach().float()
        speed, acceleration = dynamics(trajectory)

        def geometry(pose):
            return torch.cat((pose[..., :2]/pose.new_tensor([60., 30.]),
                              pose[..., 2:3].sin(), pose[..., 2:3].cos()), -1).flatten(-2).clamp(-5., 5.)

        action = F.one_hot(torch.arange(len(MODE_NAMES), device=trajectory.device),
                           len(MODE_NAMES)).float().repeat(trajectory.shape[1]//len(MODE_NAMES), 1)
        action = action[None].expand(len(trajectory), -1, -1)
        x = torch.cat((self.feature_norm(batch['features'].detach().float()),
                       batch['subscores'].detach().float(),
                       batch['pcs_scores'].detach().float()[..., None],
                       batch['rank_scores'].detach().float()[..., None],
                       batch['base_probability'].detach().float()[..., None],
                       geometry(original), geometry(trajectory), speed, acceleration, action), -1)
        x = self.embed(x)
        attended, _ = self.attention(x, x, x, need_weights=False)
        value = self.head(self.norm(x+attended)).squeeze(-1)
        return value-value[:, :1]


def safe_targets(labels, scores, direction):
    """Unsafe variants receive one low scalar target in the sole rank loss."""
    reference = labels[:, :1, [0, 1, 3, 4]]
    safe = (labels[:, :, [0, 1, 3, 4]] >= reference-1e-7).all(-1)
    safe &= direction >= direction[:, :1]-1e-7
    target = scores-scores[:, :1]
    return torch.where(safe, target, target.new_full(target.shape, -1.))


def value_loss(predicted, labels, scores, direction, min_gap=.005, temperature=.05):
    targets = safe_targets(labels.detach(), scores.detach(), direction.detach())
    gap = targets[:, :, None]-targets[:, None, :]
    valid = gap >= min_gap
    cost = (gap/.05).clamp(.25, 5.)
    diff = predicted[:, :, None]-predicted[:, None, :]
    pair = F.softplus(-diff/temperature)*cost*valid
    counts = valid.sum((1, 2))
    active = counts > 0
    loss = (pair.sum((1, 2))/counts.clamp_min(1)).sum()/active.sum().clamp_min(1)
    return loss, dict(active_scenes=active.sum(), pairs=valid.sum())


def choose(predicted, threshold, allowed=None):
    if threshold < 0:
        raise ValueError('Threshold must be nonnegative')
    indices = tuple(range(1, predicted.shape[1])) if allowed is None else tuple(allowed)
    if not indices or any(index <= 0 or index >= predicted.shape[1] for index in indices):
        raise ValueError('Invalid nonidentity timing candidate subset')
    candidates = torch.as_tensor(indices, device=predicted.device)
    value, local_index = predicted[:, candidates].max(-1)
    selected = candidates[local_index]
    return torch.where(value > threshold, selected, torch.zeros_like(selected))
