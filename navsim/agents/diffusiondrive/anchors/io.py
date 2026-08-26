"""Serialization helpers for adaptive DiffusionDrive anchor banks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from navsim.agents.diffusiondrive.anchors.base_expander import BaseExpansionResult
from navsim.agents.diffusiondrive.anchors.anchor_dedup import AnchorDedupResult
from navsim.agents.diffusiondrive.anchors.command_anchor_bank import (
    COMMAND_ORDER,
    InitialCommandAnchorBankResult,
)
from navsim.agents.diffusiondrive.anchors.composer import CompositionResult
from navsim.agents.diffusiondrive.anchors.residual_codebook import ResidualCodebookResult
from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    TrajectoryCommand,
    TrajectoryDistanceScale,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def pack_residual_codebooks(result: ResidualCodebookResult) -> tuple[np.ndarray, np.ndarray]:
    root_count = max(result.codebooks) + 1
    max_modes = max(len(codebook) for codebook in result.codebooks.values())
    trajectory_shape = next(iter(result.codebooks.values())).shape[1:]
    packed = np.zeros((root_count, max_modes, *trajectory_shape), dtype=np.float32)
    valid = np.zeros((root_count, max_modes), dtype=bool)
    for root_id, codebook in result.codebooks.items():
        packed[root_id, : len(codebook)] = codebook
        valid[root_id, : len(codebook)] = True
    return packed, valid


def save_anchor_artifacts(
    output_dir: Path,
    stem: str,
    base_result: BaseExpansionResult,
    residual_result: ResidualCodebookResult,
    composition: CompositionResult,
    report: Dict[str, Any],
) -> Dict[str, Path]:
    """Save model-ready NPY, metadata NPZ, and human-readable JSON report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{stem}.npy"
    base_npy_path = output_dir / f"{stem}_base.npy"
    npz_path = output_dir / f"{stem}.npz"
    report_path = output_dir / f"{stem}_report.json"
    codebooks, codebook_valid = pack_residual_codebooks(residual_result)

    np.save(npy_path, composition.anchors.astype(np.float32))
    np.save(base_npy_path, base_result.base_anchors.astype(np.float32))
    np.savez_compressed(
        npz_path,
        anchors=composition.anchors,
        base_anchors=base_result.base_anchors,
        root_ids=base_result.root_ids,
        node_ids=base_result.node_ids,
        parent_ids=base_result.parent_ids,
        residual_codebooks=codebooks,
        residual_codebook_valid=codebook_valid,
        compatibility_mask=composition.compatibility_mask,
        residual_support_count=residual_result.support_count,
        residual_support_ratio=residual_result.support_ratio,
        final_source_base_indices=composition.source_base_indices,
        final_source_root_ids=composition.source_root_ids,
        final_source_residual_indices=composition.source_residual_indices,
        final_support_count=composition.support_count,
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=_json_default)
    return {
        "npy": npy_path,
        "base_npy": base_npy_path,
        "npz": npz_path,
        "report": report_path,
    }


def save_command_anchor_artifacts(
    output_dir: Path,
    stem: str,
    result: AnchorDedupResult,
    report: Dict[str, Any],
    scales: Mapping[TrajectoryCommand, TrajectoryDistanceScale],
) -> Dict[str, Path]:
    """Save the command-conditioned Base Bank and all section 11 metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{stem}.npy"
    npz_path = output_dir / f"{stem}.npz"
    report_path = output_dir / f"{stem}_report.json"

    np.save(npy_path, result.anchors.astype(np.float32))
    np.savez_compressed(
        npz_path,
        anchors=result.anchors.astype(np.float32),
        command_ids=result.command_ids,
        support=result.support,
        parent_ids=result.parent_ids,
        root_ids=result.root_ids,
        node_ids=result.node_ids,
        assignments=result.assignments,
        nearest_d_traj=result.nearest_distances,
        local_mean_d_traj=result.local_mean_distances,
        dedup_source_indices=result.source_indices,
        dedup_removed_count=np.asarray(result.removed_count, dtype=np.int64),
        delta_scales=np.asarray([scales[command].delta_scale for command in COMMAND_ORDER]),
        fde_scales=np.asarray([scales[command].fde_scale for command in COMMAND_ORDER]),
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=_json_default)
    return {"npy": npy_path, "npz": npz_path, "report": report_path}


def save_initial_command_anchor_artifacts(
    output_dir: Path,
    stem: str,
    result: InitialCommandAnchorBankResult,
) -> Dict[str, Path]:
    """Save the traceable command-conditioned initial Base Bank."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{stem}_initial.npy"
    commands_path = output_dir / f"{stem}_initial_commands.npy"
    npz_path = output_dir / f"{stem}_initial.npz"
    np.save(npy_path, result.anchors.astype(np.float32))
    np.save(commands_path, result.command_ids)
    np.savez_compressed(
        npz_path,
        anchors=result.anchors.astype(np.float32),
        command_ids=result.command_ids,
        support=result.support,
        source_indices=result.source_indices,
        command_counts=result.command_counts,
        anchor_counts=result.anchor_counts,
        sampled_counts=result.sampled_counts,
    )
    return {"npy": npy_path, "commands": commands_path, "npz": npz_path}
