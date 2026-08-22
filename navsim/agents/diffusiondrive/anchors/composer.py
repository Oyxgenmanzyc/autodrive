"""Compose data-supported base/residual pairs and remove duplicates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from navsim.agents.diffusiondrive.anchors.base_expander import AnchorBuilderConfig
from navsim.agents.diffusiondrive.anchors.residual_codebook import (
    ResidualCodebookResult,
    local_residual_to_xy,
)


@dataclass
class CompositionResult:
    anchors: np.ndarray
    source_base_indices: np.ndarray
    source_root_ids: np.ndarray
    source_residual_indices: np.ndarray
    support_count: np.ndarray
    compatibility_mask: np.ndarray
    dedup_removed_count: int


def compose_anchor_bank(
    base_anchors: npt.ArrayLike,
    root_ids: npt.ArrayLike,
    residual_result: ResidualCodebookResult,
    config: AnchorBuilderConfig,
) -> CompositionResult:
    """Compose only same-root residuals with sufficient per-base data support."""
    base_anchors = np.asarray(base_anchors, dtype=np.float32)
    root_ids = np.asarray(root_ids, dtype=np.int64)
    if root_ids.shape != (len(base_anchors),):
        raise ValueError("root_ids must have shape [K_base]")

    max_modes = residual_result.support_count.shape[1]
    compatibility_mask = np.zeros((len(base_anchors), max_modes), dtype=bool)
    candidates: list[tuple[np.ndarray, int, int, int, bool]] = []
    for base_idx, (base_anchor, root_id) in enumerate(zip(base_anchors, root_ids)):
        codebook = residual_result.codebooks[int(root_id)]
        base_total_support = int(residual_result.support_count[base_idx].sum())
        for residual_idx, residual in enumerate(codebook):
            count = int(residual_result.support_count[base_idx, residual_idx])
            ratio = float(residual_result.support_ratio[base_idx, residual_idx])
            compatible = residual_idx == 0 or (
                count >= config.residual_min_support
                and ratio >= config.residual_min_support_ratio
            )
            compatibility_mask[base_idx, residual_idx] = compatible
            if not compatible:
                continue
            composed = local_residual_to_xy(base_anchor[None], residual[None])[0].astype(np.float32)
            priority_support = base_total_support if residual_idx == 0 else count
            candidates.append((composed, base_idx, residual_idx, priority_support, residual_idx == 0))

    candidates.sort(key=lambda item: (-item[3], -int(item[4]), item[1], item[2]))
    kept: list[tuple[np.ndarray, int, int, int, bool]] = []
    for candidate in candidates:
        if kept:
            kept_anchors = np.stack([item[0] for item in kept])
            ade = np.linalg.norm(kept_anchors - candidate[0][None], axis=-1).mean(axis=-1)
            if np.any(ade < config.dedup_threshold):
                continue
        kept.append(candidate)

    if not kept:
        raise RuntimeError("composition produced an empty anchor bank")
    anchors = np.stack([item[0] for item in kept]).astype(np.float32)
    base_indices = np.array([item[1] for item in kept], dtype=np.int64)
    residual_indices = np.array([item[2] for item in kept], dtype=np.int64)
    support_count = np.array([item[3] for item in kept], dtype=np.int64)
    return CompositionResult(
        anchors=anchors,
        source_base_indices=base_indices,
        source_root_ids=root_ids[base_indices],
        source_residual_indices=residual_indices,
        support_count=support_count,
        compatibility_mask=compatibility_mask,
        dedup_removed_count=len(candidates) - len(kept),
    )
