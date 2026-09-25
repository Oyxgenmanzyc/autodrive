"""Calibration and held-out audit for the deployable timing selector."""
import numpy as np

from navsim.agents.diffusiondrive.cost_rank.data import log_partitions
from navsim.agents.diffusiondrive.timing_oracle.geometry import ACTIONS


def _selected(logits, threshold):
    mode = np.asarray(logits).argmax(-1)
    confidence = np.asarray(logits)[np.arange(len(mode)), mode]
    return np.where(confidence > threshold, mode, 0)


def _report(rows, indices, threshold):
    logits = rows["logits"][indices]
    labels = rows["labels"][indices]
    scores = rows["scores"][indices]
    direction = rows["direction"][indices]
    mode = _selected(logits, threshold)
    arange = np.arange(len(mode))
    final_scores = scores[arange, mode]
    delta = final_scores - scores[:, 0]
    final_labels = labels[arange, mode]
    base_labels = labels[:, 0]
    final_direction = direction[arange, mode]
    base_direction = direction[:, 0]
    new_failures = {
        "new_nc_failure": int(((base_labels[:, 0] >= 1 - 1e-6) & (final_labels[:, 0] < 1 - 1e-6)).sum()),
        "new_dac_failure": int(((base_labels[:, 1] >= 1 - 1e-6) & (final_labels[:, 1] < 1 - 1e-6)).sum()),
        "new_ttc_failure": int(((base_labels[:, 3] >= 1 - 1e-6) & (final_labels[:, 3] < 1 - 1e-6)).sum()),
        "new_comfort_failure": int(((base_labels[:, 4] >= 1 - 1e-6) & (final_labels[:, 4] < 1 - 1e-6)).sum()),
        "new_direction_failure": int(((base_direction >= 1 - 1e-6) & (final_direction < 1 - 1e-6)).sum()),
    }
    families = [ACTIONS[int(value)][0] for value in mode]
    return {
        "scenes": int(len(indices)), "threshold": float(threshold),
        "identity_pdm": float(scores[:, 0].mean()), "pdm": float(final_scores.mean()),
        "gain_points": float(delta.mean() * 100), "changed": int((mode != 0).sum()),
        "beneficial": int((delta > 1e-6).sum()), "harmful": int((delta < -1e-6).sum()),
        "severe_losses": int((delta <= -0.2).sum()),
        "action_family_counts": {family: families.count(family) for family in sorted(set(families))},
        **new_failures,
        "safety_pass": bool(not any(new_failures.values()) and not (delta <= -0.2).any()),
    }


def calibrated_reports(rows, records):
    partition = log_partitions(records, 2)
    calibration = np.flatnonzero(partition == 0)
    audit = np.flatnonzero(partition == 1)
    best = np.asarray(rows["logits"])[calibration].max(-1)
    finite = best[np.isfinite(best)]
    if not len(finite):
        raise ValueError("No finite timing-selector predictions")
    thresholds = [float(np.nextafter(finite.max(), np.inf))]
    thresholds.extend(float(x) for x in np.unique(np.quantile(finite, np.linspace(0, 1, 101))))
    candidates = [_report(rows, calibration, value) for value in thresholds]
    safe = [item for item in candidates if item["safety_pass"]]
    chosen = max(safe or candidates[:1], key=lambda item: (item["pdm"], -item["changed"]))
    threshold = chosen["threshold"]
    return {
        "policy": {"enabled": bool(chosen["changed"] > 0), "threshold": threshold},
        "reports": {"calibration": chosen, "audit": _report(rows, audit, threshold)},
    }
