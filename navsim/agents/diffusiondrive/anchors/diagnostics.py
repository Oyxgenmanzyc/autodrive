"""Read-only 3.1.02 diagnostics adapted from 3.1.02_2 utilization/diversity metrics.

All distances here are legacy ADE in metres, not command-conditioned D_traj.
Nothing in this module participates in anchor construction or model selection.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from navsim.agents.diffusiondrive.anchors.metrics import (
    _validate_trajectory_array,
    evaluate_anchor_bank,
    nearest_anchor_assignment,
    pairwise_ade,
)


def utilization_summary(counts: Sequence[int]) -> dict[str, Any]:
    """Frequency, >0.1% active rate, and entropy (natural log) of one histogram."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("counts must be a nonempty [K] vector")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must contain finite nonnegative integers")
    total = int(counts.sum())
    frequency = counts.astype(np.float64) / max(total, 1)
    positive = frequency[frequency > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "sample_count": total,
        "counts": counts.astype(np.int64).tolist(),
        "frequency": frequency.tolist(),
        "used_anchor_count": int(np.count_nonzero(counts)),
        "active_mode_rate": float((frequency > 0.001).mean()),
        "entropy_nats": entropy,
        "normalized_entropy": float(entropy / np.log(len(counts))) if len(counts) > 1 else 0.0,
    }


def audit_anchor_bank(trajectories, anchors, batch_size: int = 4096) -> dict[str, Any]:
    """Legacy coverage plus static support and duplicate/nearest-neighbor audit."""
    anchors = _validate_trajectory_array("anchors", anchors)
    assignments, _ = nearest_anchor_assignment(trajectories, anchors, batch_size)
    counts = np.bincount(assignments, minlength=len(anchors))
    distances = pairwise_ade(anchors, anchors)
    pairs = distances[np.triu_indices(len(anchors), k=1)]
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1) if len(anchors) > 1 else np.array([])
    point_errors = np.linalg.norm(np.asarray(trajectories) - anchors[assignments], axis=-1)
    return {
        "distance": "legacy ADE in metres; no command restriction",
        "coverage_definition": "min_anchor(max_t(point_error)) <= threshold (original 3.1.02)",
        "coverage": evaluate_anchor_bank(trajectories, anchors, batch_size=batch_size),
        "static_assignment": utilization_summary(counts),
        "mean_FDE_of_ADE_winner": float(point_errors[:, -1].mean()),
        "p95_max_error_of_ADE_winner": float(np.quantile(point_errors.max(axis=1), 0.95)),
        "diversity": {
            "nearest_neighbor_ADE_m": nearest.tolist(),
            "mean_nearest_neighbor_ADE_m": float(nearest.mean()) if len(nearest) else None,
            "median_nearest_neighbor_ADE_m": float(np.median(nearest)) if len(nearest) else None,
            "pair_count": len(pairs),
            "pair_fraction_ADE_below_m": {
                str(threshold): float((pairs < threshold).mean()) if len(pairs) else 0.0
                for threshold in (0.10, 0.15, 0.20, 0.25)
            },
        },
    }


def empty_layer_stats(mode_count: int) -> dict[str, Any]:
    return {
        "batches": 0, "nonfinite_batches": 0, "samples_seen": 0,
        "finite_samples": 0, "top1_correct": 0,
        "weighted_cls_sum": 0.0, "weighted_reg_sum": 0.0,
        "weighted_total_sum": 0.0,
        "winner_counts": [0] * mode_count, "selected_counts": [0] * mode_count,
    }


def summarize_layer(stats: dict[str, Any]) -> dict[str, Any]:
    count = stats["finite_samples"]
    return {
        **stats,
        "trajectory_cls_loss": stats["weighted_cls_sum"] / count if count else None,
        "trajectory_reg_loss": stats["weighted_reg_sum"] / count if count else None,
        "trajectory_total_loss": stats["weighted_total_sum"] / count if count else None,
        "top1_anchor_accuracy": stats["top1_correct"] / count if count else None,
        "static_winner_utilization": utilization_summary(stats["winner_counts"]),
        "logit_selected_utilization": utilization_summary(stats["selected_counts"]),
    }


def merge_rank_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge completed rank files offline, never averaging rank entropy scalars."""
    if not reports:
        raise ValueError("No rank reports supplied")
    first = reports[0]
    identity = ("schema_version", "run_root", "epoch", "world_size", "bank_sha256", "mode_count")
    for report in reports:
        if any(report[key] != first[key] for key in identity):
            raise ValueError("Reports must belong to the same run, epoch, world size and bank")
        if set(report["layers"]) != set(first["layers"]):
            raise ValueError("Decoder layer sets differ between ranks")
    ranks = [report["rank"] for report in reports]
    if sorted(ranks) != list(range(first["world_size"])):
        raise ValueError("Supply exactly one file for every rank; partial/duplicate ranks rejected")
    result = {key: first[key] for key in identity}
    result["scope"] = "global_training_draws_including_DDP_sampler_padding"
    result["layers"] = {}
    for layer in first["layers"]:
        merged = empty_layer_stats(first["mode_count"])
        for report in reports:
            stats = report["layers"][layer]
            if sum(stats["winner_counts"]) != stats["finite_samples"]:
                raise ValueError("Winner counts do not match finite sample count")
            if sum(stats["selected_counts"]) != stats["finite_samples"]:
                raise ValueError("Selected counts do not match finite sample count")
            for key in merged:
                if key.endswith("_counts"):
                    if len(stats[key]) != first["mode_count"]:
                        raise ValueError("Histogram length does not match bank")
                    merged[key] = [a + b for a, b in zip(merged[key], stats[key])]
                else:
                    merged[key] += stats[key]
        result["layers"][layer] = summarize_layer(merged)
    return result
