"""Coverage-guided adaptive splitting of DiffusionDrive base anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from navsim.agents.diffusiondrive.anchors.metrics import (
    cluster_statistics,
    evaluate_anchor_bank,
    nearest_anchor_assignment,
)


@dataclass(frozen=True)
class AnchorBuilderConfig:
    """All tunable offline anchor-builder parameters."""

    coverage_epsilon: float = 1.0
    coverage_target: float = 0.95
    coverage_error_threshold: float = 1.0
    variance_threshold: float = 0.10
    min_split_samples: int = 50
    min_coverage_gain: float = 1e-3
    min_ade_gain: float = 1e-3
    saturation_patience: int = 3
    max_base_anchors: int = 64
    residual_modes: int = 8
    residual_min_support: int = 20
    residual_min_support_ratio: float = 0.02
    dedup_threshold: float = 0.05
    random_seed: int = 0
    assignment_batch_size: int = 4096

    def __post_init__(self) -> None:
        if not 0 < self.coverage_target <= 1:
            raise ValueError("coverage_target must be in (0, 1]")
        if self.coverage_epsilon <= 0 or self.coverage_error_threshold <= 0:
            raise ValueError("coverage thresholds must be positive")
        if self.min_split_samples < 2 or self.max_base_anchors < 1:
            raise ValueError("split sample count and anchor cap must be positive")
        if self.residual_modes < 1:
            raise ValueError("residual_modes must include at least the zero residual")


@dataclass
class BaseExpansionResult:
    base_anchors: np.ndarray
    root_ids: np.ndarray
    node_ids: np.ndarray
    parent_ids: np.ndarray
    assignments: np.ndarray
    cluster_stats: list[Dict[str, Any]]
    history: list[Dict[str, Any]]
    stop_reason: str


def _split_cluster(samples: np.ndarray, random_seed: int) -> np.ndarray:
    model = KMeans(n_clusters=2, n_init=10, random_state=random_seed)
    centers = model.fit(samples.reshape(len(samples), -1)).cluster_centers_
    return centers.reshape(2, samples.shape[1], samples.shape[2])


def _history_entry(
    trajectories: np.ndarray,
    anchors: np.ndarray,
    split_node_id: Optional[int],
    config: AnchorBuilderConfig,
) -> Dict[str, Any]:
    metrics = evaluate_anchor_bank(
        trajectories,
        anchors,
        coverage_epsilons=(0.5, 1.0, 1.5, 2.0),
        batch_size=config.assignment_batch_size,
    )
    metrics["split_node_id"] = split_node_id
    return metrics


def expand_base_anchors(
    trajectories: npt.ArrayLike,
    initial_anchors: npt.ArrayLike,
    config: AnchorBuilderConfig,
) -> BaseExpansionResult:
    """Repeatedly split the most under-covered, high-variance ADE cluster."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    anchors = np.asarray(initial_anchors, dtype=np.float32).copy()
    if trajectories.ndim != 3 or anchors.ndim != 3 or trajectories.shape[1:] != anchors.shape[1:]:
        raise ValueError("trajectories and initial_anchors must have compatible [N/K, T, 2] shapes")
    if config.max_base_anchors < len(anchors):
        raise ValueError("max_base_anchors cannot be smaller than the initial bank")

    root_ids = np.arange(len(anchors), dtype=np.int64)
    node_ids = np.arange(len(anchors), dtype=np.int64)
    parent_ids = np.full(len(anchors), -1, dtype=np.int64)
    next_node_id = len(anchors)
    history = [_history_entry(trajectories, anchors, None, config)]
    stagnant_rounds = 0
    stop_reason = "max_base_anchors"

    while len(anchors) < config.max_base_anchors:
        current_coverage = history[-1][f"coverage@{config.coverage_epsilon:g}m"]
        if current_coverage >= config.coverage_target:
            stop_reason = "coverage_target"
            break

        assignments, _ = nearest_anchor_assignment(
            trajectories, anchors, batch_size=config.assignment_batch_size
        )
        stats = cluster_statistics(trajectories, anchors, assignments, config.coverage_epsilon)
        candidates = [
            stat
            for stat in stats
            if stat["coverage_error_p95"] > config.coverage_error_threshold
            and stat["variance"] > config.variance_threshold
            and stat["support"] >= config.min_split_samples
            and stat["uncovered_count"] > 0
        ]
        if not candidates:
            stop_reason = "no_split_candidate"
            break

        selected = max(
            candidates,
            key=lambda stat: (
                stat["uncovered_count"],
                stat["coverage_error_p95"],
                stat["variance"],
            ),
        )
        split_idx = int(selected["anchor_index"])
        split_node_id = int(node_ids[split_idx])
        children = _split_cluster(trajectories[assignments == split_idx], config.random_seed + next_node_id)

        keep = np.arange(len(anchors)) != split_idx
        anchors = np.concatenate([anchors[keep], children.astype(np.float32)], axis=0)
        root_ids = np.concatenate([root_ids[keep], np.repeat(root_ids[split_idx], 2)])
        node_ids = np.concatenate([node_ids[keep], np.array([next_node_id, next_node_id + 1])])
        parent_ids = np.concatenate([parent_ids[keep], np.repeat(split_node_id, 2)])
        next_node_id += 2

        previous = history[-1]
        current = _history_entry(trajectories, anchors, split_node_id, config)
        current["split_cluster"] = selected
        history.append(current)
        coverage_gain = current[f"coverage@{config.coverage_epsilon:g}m"] - previous[
            f"coverage@{config.coverage_epsilon:g}m"
        ]
        ade_gain = previous["mean_nearest_ADE"] - current["mean_nearest_ADE"]
        stagnant_rounds = stagnant_rounds + 1 if (
            coverage_gain < config.min_coverage_gain and ade_gain < config.min_ade_gain
        ) else 0
        if stagnant_rounds >= config.saturation_patience:
            stop_reason = "saturated"
            break

    assignments, _ = nearest_anchor_assignment(
        trajectories, anchors, batch_size=config.assignment_batch_size
    )
    stats = cluster_statistics(trajectories, anchors, assignments, config.coverage_epsilon)
    return BaseExpansionResult(
        base_anchors=anchors,
        root_ids=root_ids,
        node_ids=node_ids,
        parent_ids=parent_ids,
        assignments=assignments,
        cluster_stats=stats,
        history=history,
        stop_reason=stop_reason,
    )


def config_to_dict(config: AnchorBuilderConfig) -> Dict[str, Any]:
    return asdict(config)
