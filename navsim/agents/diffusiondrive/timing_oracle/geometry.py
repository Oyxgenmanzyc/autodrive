"""Fixed-path timing actions used only by the offline Oracle diagnostic.

The bank contains the exact identity, the old one-sided additional-braking
actions, and symmetric time reparameterizations.  Time warps preserve the
selected polyline and terminal point; they test when progress is made without
inventing an unobserved continuation beyond the four-second path.
"""
import math
import numpy as np

DT = 0.5
BRAKE_ACTIONS = [
    ("extra_brake", onset, strength, ramp)
    for onset in (0.0, 0.5, 1.0, 1.5)
    for strength in (0.25, 0.75, 1.5)
    for ramp in (0.5, 1.0)
]
# exp(+/-d) is reciprocal, so both timing directions have matched magnitudes.
WARP_DELTAS = (-0.30, -0.18, -0.09, 0.09, 0.18, 0.30)
ACTIONS = [("identity", 0.0, 0.0, 0.0)] + BRAKE_ACTIONS + [
    ("frontload_progress" if delta < 0 else "backload_progress", delta, 0.0, 0.0)
    for delta in WARP_DELTAS
]
BRAKE_INDICES = np.arange(0, 1 + len(BRAKE_ACTIONS), dtype=np.int64)
WARP_INDICES = np.asarray([0] + list(range(1 + len(BRAKE_ACTIONS), len(ACTIONS))), dtype=np.int64)
FRONTLOAD_INDICES = np.asarray(
    [0] + [i for i, action in enumerate(ACTIONS) if action[0] == "frontload_progress"], dtype=np.int64
)
BACKLOAD_INDICES = np.asarray(
    [0] + [i for i, action in enumerate(ACTIONS) if action[0] == "backload_progress"], dtype=np.int64
)
ALL_INDICES = np.arange(len(ACTIONS), dtype=np.int64)


def _path(trajectory):
    original = np.asarray(trajectory, dtype=np.float32)
    if original.shape != (8, 3) or not np.isfinite(original).all():
        raise ValueError("Expected finite [8,3] trajectory")
    points = np.concatenate([np.zeros((1, 2), np.float32), original[:, :2]], axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))])
    heading = np.unwrap(np.concatenate([[0.0], original[:, 2]]))
    keep = np.concatenate([[True], np.diff(arc) > 1e-8])
    return original, points, arc, heading, keep


def _sample(original, points, arc, heading, keep, progress):
    result = original.copy()
    for dim in (0, 1):
        result[:, dim] = np.interp(progress, arc[keep], points[keep, dim])
    angle = np.interp(progress, arc[keep], heading[keep])
    result[:, 2] = np.arctan2(np.sin(angle), np.cos(angle))
    return result


def action_bank(trajectory):
    """Return [31,8,3] actions; action zero is bitwise identity.

    ``frontload_progress`` advances farther during early timestamps and then
    gives that advantage back before the fixed endpoint. ``backload_progress``
    is the reciprocal timing shift.  Neither changes the spatial polyline.
    """
    original, points, arc, heading, keep = _path(trajectory)
    bank = np.repeat(original[None], len(ACTIONS), axis=0)
    if arc[-1] < 1e-6:
        return bank
    times = np.arange(9, dtype=np.float64) * DT
    fine = np.linspace(0.0, 4.0, 81)
    fine_arc = np.interp(fine, times, arc)
    h = fine[1] - fine[0]
    mid = (fine[1:] + fine[:-1]) * 0.5
    for index, (_, onset, strength, ramp) in enumerate(ACTIONS[1:1 + len(BRAKE_ACTIONS)], 1):
        elapsed = np.maximum(mid - onset, 0.0)
        dv = strength * np.where(elapsed < ramp, elapsed**2 / (2 * ramp), elapsed - ramp / 2)
        step = np.maximum(np.diff(fine_arc) - dv * h, 0.0)
        progress = np.interp(times[1:], fine, np.concatenate([[0.0], np.cumsum(step)]))
        bank[index] = _sample(original, points, arc, heading, keep, progress)
    first_warp = 1 + len(BRAKE_ACTIONS)
    for offset, delta in enumerate(WARP_DELTAS):
        gamma = math.exp(delta)
        source_time = 4.0 * np.power(times[1:] / 4.0, gamma)
        progress = np.interp(source_time, times, arc)
        bank[first_warp + offset] = _sample(original, points, arc, heading, keep, progress)
    bank[0] = original
    return bank


def safe_oracle(labels, scores, direction, indices=ALL_INDICES):
    """Choose the best action without regressing NC/DAC/TTC/comfort/direction."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or indices[0] != 0:
        raise ValueError("Oracle subsets must include identity at index zero")
    local_labels = np.asarray(labels)[..., indices, :]
    local_scores = np.asarray(scores)[..., indices]
    local_direction = np.asarray(direction)[..., indices]
    guarded = local_labels[..., [0, 1, 3, 4]]
    eligible = (guarded >= guarded[..., :1, :] - 1e-7).all(-1)
    eligible &= local_direction >= local_direction[..., :1] - 1e-7
    gain = local_scores - local_scores[..., :1]
    eligible &= gain > 1e-6
    eligible[..., 0] = True
    local = np.where(eligible, gain, -np.inf).argmax(-1)
    return indices[local]
