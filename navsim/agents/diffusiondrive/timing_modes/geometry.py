"""Continuous distance-versus-time modes on a frozen four-second spatial path."""
import math

import numpy as np

DT = 0.5
MODE_NAMES = ('identity', 'frontload', 'backload', 'early_brake')
TOP_SPATIAL = 5
SPATIAL_ONLY_INDICES = tuple(range(len(MODE_NAMES), TOP_SPATIAL*len(MODE_NAMES),
                                   len(MODE_NAMES)))


def _sample(points, arc, heading, progress):
    keep = np.concatenate(([True], np.diff(arc) > 1e-8))
    out = np.empty((8, 3), dtype=np.float32)
    for axis in (0, 1):
        out[:, axis] = np.interp(progress, arc[keep], points[keep, axis])
    angle = np.interp(progress, arc[keep], heading[keep])
    out[:, 2] = np.arctan2(np.sin(angle), np.cos(angle))
    return out


def expand_path(path):
    """Return four modes [4,8,3], with bitwise-identical identity at index 0.

    All nonidentity modes sample one monotone s(t) over the *same* polyline.
    Early/late modes have reciprocal time exponents and preserve the endpoint.
    Braking is a smooth nonnegative reduction of progress and may stop early.
    """
    original = np.asarray(path, dtype=np.float32)
    if original.shape != (8, 3) or not np.isfinite(original).all():
        raise ValueError('Expected finite [8,3] trajectory')
    bank = np.repeat(original[None], len(MODE_NAMES), axis=0)
    points = np.concatenate((np.zeros((1, 2), np.float32), original[:, :2]))
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc[-1] <= 1e-6:
        return bank
    heading = np.unwrap(np.concatenate(([0.], original[:, 2])))
    times = np.arange(9, dtype=np.float64) * DT
    for i, delta in ((1, -.18), (2, .18)):
        source = 4. * np.power(times[1:] / 4., math.exp(delta))
        progress = np.interp(source, times, arc)
        bank[i] = _sample(points, arc, heading, progress)
    fine = np.linspace(0., 4., 81)
    fine_arc = np.interp(fine, times, arc)
    mid = (fine[1:] + fine[:-1]) / 2.
    onset, ramp, strength = .5, 1., .75
    elapsed = np.maximum(mid-onset, 0.)
    deceleration = strength * np.where(elapsed < ramp,
                                       elapsed**2/(2.*ramp), elapsed-ramp/2.)
    steps = np.maximum(np.diff(fine_arc)-deceleration*np.diff(fine), 0.)
    progress = np.interp(times[1:], fine, np.concatenate(([0.], np.cumsum(steps))))
    bank[3] = _sample(points, arc, heading, progress)
    bank[0] = original
    return bank


def expand_top(paths):
    paths = np.asarray(paths, dtype=np.float32)
    if paths.shape != (TOP_SPATIAL, 8, 3):
        raise ValueError(f'Expected [{TOP_SPATIAL},8,3] spatial paths')
    return np.concatenate([expand_path(path) for path in paths], axis=0)


def top_modes(scores, count=TOP_SPATIAL):
    """Stable tie-break by original K67 index; mode zero is locked Rank choice."""
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (67,) or not np.isfinite(values).all():
        raise ValueError('Expected finite K67 scores')
    return np.argsort(-values, kind='stable')[:count].astype(np.int64)
