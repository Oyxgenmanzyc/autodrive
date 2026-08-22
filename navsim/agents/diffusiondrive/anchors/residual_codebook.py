"""Root-conditioned tangent-normal residual quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from navsim.agents.diffusiondrive.anchors.base_expander import AnchorBuilderConfig


@dataclass
class ResidualCodebookResult:
    codebooks: Dict[int, np.ndarray]
    base_assignments: np.ndarray
    residual_assignments: np.ndarray
    support_count: np.ndarray
    support_ratio: np.ndarray
    distortion_curves: Dict[int, Dict[int, float]]


def tangent_normal_frames(anchors: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable local frame at every base-anchor waypoint."""
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.ndim != 3 or anchors.shape[-1] != 2:
        raise ValueError("anchors must have shape [K, T, 2]")
    origin = np.zeros_like(anchors[:, :1])
    tangents = anchors - np.concatenate([origin, anchors[:, :-1]], axis=1)
    for anchor_idx in range(len(tangents)):
        last_valid = np.array([1.0, 0.0])
        for time_idx in range(tangents.shape[1]):
            norm = np.linalg.norm(tangents[anchor_idx, time_idx])
            if norm > 1e-8:
                last_valid = tangents[anchor_idx, time_idx] / norm
            tangents[anchor_idx, time_idx] = last_valid
    normals = np.stack([-tangents[..., 1], tangents[..., 0]], axis=-1)
    return tangents, normals


def xy_to_local_residual(
    trajectories: npt.ArrayLike,
    assigned_anchors: npt.ArrayLike,
) -> np.ndarray:
    """Project XY errors into each assigned anchor's tangent-normal frame."""
    trajectories = np.asarray(trajectories, dtype=np.float64)
    assigned_anchors = np.asarray(assigned_anchors, dtype=np.float64)
    if trajectories.shape != assigned_anchors.shape or trajectories.ndim != 3:
        raise ValueError("trajectories and assigned_anchors must share shape [N, T, 2]")
    tangents, normals = tangent_normal_frames(assigned_anchors)
    error = trajectories - assigned_anchors
    longitudinal = (error * tangents).sum(axis=-1)
    lateral = (error * normals).sum(axis=-1)
    return np.stack([longitudinal, lateral], axis=-1)


def local_residual_to_xy(
    anchors: npt.ArrayLike,
    local_residuals: npt.ArrayLike,
) -> np.ndarray:
    """Compose anchors with tangent-normal residuals back in XY coordinates."""
    anchors = np.asarray(anchors, dtype=np.float64)
    local_residuals = np.asarray(local_residuals, dtype=np.float64)
    if anchors.shape != local_residuals.shape or anchors.ndim != 3:
        raise ValueError("anchors and local_residuals must share shape [N, T, 2]")
    tangents, normals = tangent_normal_frames(anchors)
    xy_error = local_residuals[..., :1] * tangents + local_residuals[..., 1:] * normals
    return anchors + xy_error


def _fit_residual_centers(residuals: np.ndarray, modes: int, seed: int) -> np.ndarray:
    if modes <= 0:
        return np.empty((0, residuals.shape[1], 2), dtype=np.float32)
    modes = min(modes, len(residuals))
    model = KMeans(n_clusters=modes, n_init=10, random_state=seed)
    centers = model.fit(residuals.reshape(len(residuals), -1)).cluster_centers_
    return centers.reshape(modes, residuals.shape[1], 2).astype(np.float32)


def _distortion_curve(residuals: np.ndarray, seed: int) -> Dict[int, float]:
    curve: Dict[int, float] = {}
    flattened = residuals.reshape(len(residuals), -1)
    for modes in (1, 2, 4, 8, 16):
        if modes > len(residuals):
            continue
        model = KMeans(n_clusters=modes, n_init=10, random_state=seed)
        model.fit(flattened)
        curve[modes] = float(model.inertia_ / len(residuals))
    return curve


def learn_residual_codebooks(
    trajectories: npt.ArrayLike,
    base_anchors: npt.ArrayLike,
    root_ids: npt.ArrayLike,
    base_assignments: npt.ArrayLike,
    config: AnchorBuilderConfig,
) -> ResidualCodebookResult:
    """Learn one residual vocabulary per original coarse-intention root."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    base_anchors = np.asarray(base_anchors, dtype=np.float32)
    root_ids = np.asarray(root_ids, dtype=np.int64)
    base_assignments = np.asarray(base_assignments, dtype=np.int64)
    if base_assignments.shape != (len(trajectories),):
        raise ValueError("base_assignments must have shape [N]")
    if root_ids.shape != (len(base_anchors),):
        raise ValueError("root_ids must have shape [K_base]")

    assigned_anchors = base_anchors[base_assignments]
    local_residuals = xy_to_local_residual(trajectories, assigned_anchors).astype(np.float32)
    sample_roots = root_ids[base_assignments]
    codebooks: Dict[int, np.ndarray] = {}
    distortion_curves: Dict[int, Dict[int, float]] = {}
    residual_assignments = np.zeros(len(trajectories), dtype=np.int64)

    for root_id in np.unique(root_ids):
        sample_indices = np.flatnonzero(sample_roots == root_id)
        root_residuals = local_residuals[sample_indices]
        if len(root_residuals) == 0:
            codebooks[int(root_id)] = np.zeros(
                (1, trajectories.shape[1], 2), dtype=np.float32
            )
            distortion_curves[int(root_id)] = {}
            continue
        learned = _fit_residual_centers(
            root_residuals,
            modes=config.residual_modes - 1,
            seed=config.random_seed + int(root_id),
        )
        zero = np.zeros((1, trajectories.shape[1], 2), dtype=np.float32)
        codebook = np.concatenate([zero, learned], axis=0)
        codebooks[int(root_id)] = codebook
        distortion_curves[int(root_id)] = _distortion_curve(
            root_residuals, config.random_seed + int(root_id)
        )
        distances = np.linalg.norm(root_residuals[:, None] - codebook[None], axis=-1).mean(axis=-1)
        residual_assignments[sample_indices] = distances.argmin(axis=1)

    max_modes = max(len(codebook) for codebook in codebooks.values())
    support_count = np.zeros((len(base_anchors), max_modes), dtype=np.int64)
    for sample_idx, base_idx in enumerate(base_assignments):
        support_count[base_idx, residual_assignments[sample_idx]] += 1
    base_support = support_count.sum(axis=1, keepdims=True)
    support_ratio = np.divide(
        support_count,
        base_support,
        out=np.zeros_like(support_count, dtype=np.float64),
        where=base_support > 0,
    )
    return ResidualCodebookResult(
        codebooks=codebooks,
        base_assignments=base_assignments,
        residual_assignments=residual_assignments,
        support_count=support_count,
        support_ratio=support_ratio,
        distortion_curves=distortion_curves,
    )
