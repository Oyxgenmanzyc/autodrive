"""A finite, exactly reproducible brake-action space on a fixed geometric path.

No interpolation between separately scored actions at deployment. Action zero
is the unmodified trajectory, including its original float32 heading values.
This is additional braking only: it cannot accelerate or extend the path.
"""
import numpy as np

DT = 0.5
# (onset seconds, additional deceleration m/s^2, ramp seconds).
ACTIONS = [(0., 0., 0.)] + [
    (t, a, r) for t in (0., 0.5, 1., 1.5)
    for a in (0.25, 0.75, 1.5) for r in (0.5, 1.)
]


def brake_bank(trajectory):
    """Return [25,8,3] poses; every XY lies on the selected path polyline.

    Use arc length rather than global ego-y to preserve curved paths. Remove
    duplicate arc-length knots (stationary poses) for well-defined interpolation.
    Substep slowdown is ramp-integrated; it never reverses along the path.
    Physical feasibility/safety is evaluated by the official simulator, not
    asserted from these geometric constraints.
    """
    original = np.asarray(trajectory, dtype=np.float32)
    if original.shape != (8, 3) or not np.isfinite(original).all():
        raise ValueError('Expected finite [8,3] trajectory')
    points = np.concatenate([np.zeros((1, 2)), original[:, :2]], axis=0)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    arc = np.concatenate([[0.], np.cumsum(lengths)])
    heading = np.unwrap(np.concatenate([[0.], original[:, 2]]))
    keep = np.concatenate([[True], np.diff(arc) > 1e-8])
    bank = np.repeat(original[None], len(ACTIONS), axis=0)
    if arc[-1] < 1e-6:
        return bank
    times = np.arange(9) * DT
    fine = np.linspace(0., 4., 81)
    fine_arc = np.interp(fine, times, arc)
    h = fine[1] - fine[0]
    mid = (fine[1:] + fine[:-1]) * .5
    for k, (onset, strength, ramp) in enumerate(ACTIONS[1:], 1):
        elapsed = np.maximum(mid - onset, 0.)
        # Integral of a linearly increasing additional deceleration.
        dv = strength * np.where(elapsed < ramp, elapsed**2 / (2*ramp), elapsed-ramp/2)
        step = np.maximum(np.diff(fine_arc) - dv*h, 0.)
        s = np.interp(times[1:], fine, np.concatenate([[0.], np.cumsum(step)]))
        for dim in (0, 1):
            bank[k, :, dim] = np.interp(s, arc[keep], points[keep, dim])
        angle = np.interp(s, arc[keep], heading[keep])
        bank[k, :, 2] = np.arctan2(np.sin(angle), np.cos(angle))
    return bank


def safe_oracle(labels, scores, direction):
    """Offline teacher only; ties prefer identity, then the first action.

    Five terms must not regress: NC, DAC, TTC, comfort, direction. EP can fall
    only if the actual composite PDMS improves. No teacher sees navtest during
    training or threshold calibration.
    """
    guarded = labels[..., [0, 1, 3, 4]]
    eligible = (guarded >= guarded[..., :1, :] - 1e-7).all(-1)
    eligible &= direction >= direction[..., :1] - 1e-7
    gain = scores - scores[..., :1]
    eligible &= gain > 1e-6
    eligible[..., 0] = True
    return np.where(eligible, gain, -np.inf).argmax(-1)
