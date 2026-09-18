"""Offline supervision. None of these labels may enter deployment features."""
import hashlib
import math
import numpy as np
import torch
from torch.utils.data import Sampler

REASONS = ('nc_rescue', 'dac_rescue', 'ttc_rescue', 'comfort_rescue', 'progress_gain')


def build_labels(labels, scores, direction, min_gain=.02):
    if not math.isfinite(min_gain) or not 1e-6 < min_gain <= 1.:
        raise ValueError('min_gain must exceed scoring noise tolerance')
    if not all(torch.isfinite(x).all() for x in (labels, scores, direction)):
        raise ValueError('Nonfinite supervision')
    delta = scores - scores[:, :1]
    protected = labels[..., [0, 1, 3, 4]]
    safe = (protected >= protected[:, :1] - 1e-7).all(-1)
    safe &= direction >= direction[:, :1] - 1e-7
    # Rescues require actual positive composite gain as well as no regressions.
    rescue = ((labels[..., [0, 1, 3]] > labels[:, :1, [0, 1, 3]] + 1e-7)).any(-1)
    valid = safe & (delta > 1e-6) & ((delta >= min_gain) | rescue)
    valid[:, 0] = False
    actionable = valid.any(-1)
    utility = torch.where(valid, delta, torch.full_like(delta, -torch.inf))
    teacher = utility.argmax(-1)
    teacher = torch.where(actionable, teacher, torch.zeros_like(teacher))
    rows = torch.arange(len(scores))
    chosen = labels[rows, teacher]
    base = labels[:, 0]
    reasons = torch.stack([chosen[:, j] > base[:, j] + 1e-7 for j in (0, 1, 3, 4, 2)], -1)
    reasons &= actionable[:, None]
    # Soft supervision ONLY on eligible positive-gain actions; identity excluded.
    logits = (delta[:, 1:] / .05).masked_fill(~valid[:, 1:], -1e9)
    target = logits.softmax(-1) * actionable[:, None]
    return dict(need=actionable, teacher=teacher, reasons=reasons.float(),
                regret=delta[rows, teacher], action_target=target,
                unsafe=~safe, valid_actions=valid,
                unresolved=((base[:, [0, 1, 3]] < 1.-1e-7).any(-1) & ~actionable))


def descriptors(selected):
    """Deployable path/kinematics matching only; NOT semantic actor matching."""
    xy = selected[..., :2].float()
    step = torch.diff(xy, dim=1, prepend=torch.zeros_like(xy[:, :1]))
    speed = step.norm(dim=-1) / .5
    accel = torch.diff(speed, dim=1) / .5
    heading = selected[..., 2].float()
    return torch.cat([xy.flatten(1), speed, accel, heading.sin(), heading.cos()], -1)


def match_negatives(selected, need, records, count=4):
    """Nearest negatives in TRAIN only, excluding the positive's recording log."""
    positive = torch.where(need)[0]
    negative = torch.where(~need)[0]
    if not len(positive) or len(negative) < count:
        raise ValueError('Need positive scenes and enough negatives; use the FULL edit cache')
    desc = descriptors(selected)
    scale = desc.std(0).clamp_min(.1)
    desc = (desc - desc.mean(0)) / scale
    pairs = torch.full((len(need), count), -1, dtype=torch.long)
    logs = [r['log_name'] for r in records]
    neg_logs = np.array([logs[i] for i in negative.tolist()])
    for chunk in positive.split(32):
        distance = torch.cdist(desc[chunk], desc[negative])
        for row, index in enumerate(chunk.tolist()):
            distance[row, torch.from_numpy(neg_logs == logs[index])] = torch.inf
        values, local = distance.topk(count, largest=False)
        if not torch.isfinite(values).all():
            raise ValueError('Not enough negatives from other training logs')
        pairs[chunk] = negative[local]
    return pairs


def calibration_mask(records):
    """Fixed log-level half split: calibration vs untouched validation audit."""
    logs = sorted({r['log_name'] for r in records},
                  key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(logs) < 2:
        raise ValueError('Need at least two validation logs')
    chosen = set(logs[:len(logs)//2])
    return torch.tensor([r['log_name'] in chosen for r in records], dtype=torch.bool)


class PairedBatchSampler(Sampler):
    """Per group: positive, two matched negatives, one random negative.

    One epoch visits every positive once (DDP padding at most one batch/rank).
    set_epoch changes order/negative choices. Sampling is identical across ranks
    before disjoint slicing. No 16x oversampling hidden inside a full-data epoch.
    """
    def __init__(self, need, pairs, batch_size, rank=0, world_size=1):
        if batch_size < 4 or batch_size % 4:
            raise ValueError('Per-GPU batch size must be divisible by four')
        self.positive = torch.where(need)[0]
        self.negative = torch.where(~need)[0]
        if not len(self.positive) or not len(self.negative):
            raise ValueError('Paired training needs both classes')
        self.pairs, self.groups = pairs, batch_size//4
        self.rank, self.world, self.epoch = rank, world_size, 0
        self.hard_negatives = torch.empty(0, dtype=torch.long)

    def set_hard_negatives(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if len(indices) and not torch.isin(indices, self.negative).all():
            raise ValueError('Mined negatives must come from negative training scenes')
        self.hard_negatives = indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.positive)/(self.world*self.groups))

    def __iter__(self):
        g = torch.Generator().manual_seed(1701+self.epoch)
        order = self.positive[torch.randperm(len(self.positive), generator=g)]
        total = len(self)*self.groups*self.world
        order = order.repeat(math.ceil(total/len(order)))[:total]
        order = order.reshape(-1, self.world, self.groups)[:, self.rank]
        g.manual_seed(3101+self.epoch*self.world+self.rank)
        for group in order:
            batch = []
            for p in group.tolist():
                near = self.pairs[p]
                if (near < 0).any():
                    raise ValueError('Missing matched negatives')
                choice = torch.randperm(len(near), generator=g)[:2]
                pool = self.negative
                if len(self.hard_negatives) and torch.rand((), generator=g) < .5:
                    pool = self.hard_negatives
                random = pool[torch.randint(len(pool), (1,), generator=g)].item()
                batch.extend([p, int(near[choice[0]]), int(near[choice[1]]), random])
            yield batch
