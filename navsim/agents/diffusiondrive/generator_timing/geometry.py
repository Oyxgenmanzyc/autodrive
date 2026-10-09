"""Mode-specific distance-versus-time profiles; geometry is only a teacher/seed.

At inference the final trajectory comes from the conditioned DDIM decoder, not
from this resampling function.  Every parent spatial mode receives two siblings.
"""
import numpy as np
import torch

FAMILIES = ('early', 'late')
STRENGTHS = (0.16, 0.32)
STEPS = 8


def seed_anchors(anchors):
    """Build 134 timing seeds from 67 anchors, with no top-k spatial pruning."""
    if anchors.ndim != 3 or anchors.shape[1:] != (STEPS, 2):
        raise ValueError('Expected [K,8,2] anchors')
    zero = torch.zeros_like(anchors[:, :1])
    points = torch.cat((zero, anchors), dim=1)
    times = torch.arange(1, STEPS + 1, device=anchors.device, dtype=anchors.dtype)
    result = []
    for sign in (-1., 1.):
        source = STEPS * (times / STEPS).pow(1. + sign * STRENGTHS[0])
        lower = source.floor().long().clamp(max=STEPS - 1)
        fraction = (source - lower).view(1, STEPS, 1)
        result.append(points[:, lower] * (1. - fraction) + points[:, lower + 1] * fraction)
    return torch.stack(result, dim=1).flatten(0, 1)


def warp_trajectory(path, family, strength):
    """Offline supervised target on the same spatial polyline, including t=0."""
    path = np.asarray(path, dtype=np.float32)
    if path.shape != (STEPS, 3) or not np.isfinite(path).all():
        raise ValueError('Expected finite [8,3] path')
    if family not in FAMILIES or strength <= 0 or strength >= 1:
        raise ValueError('Invalid timing profile')
    points = np.concatenate((np.zeros((1, 2), np.float32), path[:, :2]))
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc[-1] < 1e-4:
        return path.copy()
    heading = np.unwrap(np.concatenate(([0.], path[:, 2])))
    source = STEPS * (np.arange(1, STEPS + 1) / STEPS) ** (
        1. + (-strength if family == 'early' else strength))
    progress = np.interp(source, np.arange(STEPS + 1), arc)
    keep = np.concatenate(([True], np.diff(arc) > 1e-8))
    out = np.empty_like(path)
    for axis in (0, 1):
        out[:, axis] = np.interp(progress, arc[keep], points[keep, axis])
    angle = np.interp(progress, arc[keep], heading[keep])
    out[:, 2] = np.arctan2(np.sin(angle), np.cos(angle))
    return out


def teacher_bank(proposals):
    """[K,2,2,8,3] from all original parents, family, strength."""
    paths = np.asarray(proposals, dtype=np.float32)
    if paths.ndim != 3 or paths.shape[1:] != (STEPS, 3):
        raise ValueError('Expected [K,8,3] proposals')
    return np.stack([
        [[warp_trajectory(path, family, strength) for strength in STRENGTHS]
         for family in FAMILIES] for path in paths
    ]).astype(np.float32)


def select_teachers(original, bank, labels, scores, direction):
    """Pick PDM-safe *distinct* targets within each mode, without audit labels.

    The input scores use the same metric cache and one pairwise scoring call.
    Unsafe or indistinguishable schedules have zero training weight; they are
    never relabelled with the single human GT trajectory.
    """
    k = len(original)
    if bank.shape != (k, 2, len(STRENGTHS), STEPS, 3):
        raise ValueError('Teacher bank shape mismatch')
    n = k * 2 * len(STRENGTHS)
    if labels.shape != (k + n, 5) or scores.shape != (k + n,) or direction.shape != (k + n,):
        raise ValueError('Pairwise metric shape mismatch')
    base_labels = labels[:k, None, None]
    base_scores = scores[:k, None, None]
    base_direction = direction[:k, None, None]
    variant_labels = labels[k:].reshape(k, 2, len(STRENGTHS), 5)
    variant_scores = scores[k:].reshape(k, 2, len(STRENGTHS))
    variant_direction = direction[k:].reshape(k, 2, len(STRENGTHS))
    safety = (variant_labels[..., [0, 1, 3, 4]] >= base_labels[..., [0, 1, 3, 4]] - 1e-6).all(-1)
    safety &= (variant_labels[..., [0, 1, 3]] >= 1. - 1e-6).all(-1)
    safety &= variant_direction >= base_direction - 1e-6
    safety &= variant_scores >= base_scores - .02
    distance = np.linalg.norm(bank[..., :2] - original[:, None, None, :, :2], axis=-1).max(-1)
    safety &= distance >= .2
    # PDM first, then larger useful timing separation when scores tie.
    utility = np.where(safety, variant_scores + 1e-4 * distance, -np.inf)
    choice = utility.argmax(-1)
    row = np.arange(k)[:, None]
    family = np.arange(2)[None]
    chosen = bank[row, family, choice]
    valid = safety[row, family, choice]
    gain = variant_scores[row, family, choice] - base_scores[..., 0]
    weight = np.where(valid, 1. + 3. * np.maximum(gain, 0.), 0.).astype(np.float32)
    return chosen.astype(np.float32), valid, weight
