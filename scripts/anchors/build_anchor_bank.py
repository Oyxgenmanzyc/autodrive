"""Build the command-conditioned Base Anchor Bank with worst-first D_traj splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navsim.agents.diffusiondrive.anchors.calibration import load_calibration_scales
from navsim.agents.diffusiondrive.anchors.anchor_dedup import (
    deduplicate_command_anchor_bank,
)
from navsim.agents.diffusiondrive.anchors.anchor_metrics import (
    command_anchor_coverage_metrics,
    command_anchor_diversity_metrics,
)
from navsim.agents.diffusiondrive.anchors.command_anchor_bank import (
    COMMAND_ORDER,
    CommandAnchorBankConfig,
    InitialCommandAnchorBankConfig,
    build_command_anchor_bank,
    build_initial_command_anchor_bank,
)
from navsim.agents.diffusiondrive.anchors.io import (
    save_command_anchor_artifacts,
    save_initial_command_anchor_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True, help="navtrain [N, 8, 2] NPY")
    parser.add_argument("--commands", type=Path, required=True, help="matching left/straight/right NPY")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="fixed trajectory_distance_calibration.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="command_anchor_bank")
    parser.add_argument("--initial-anchor-count", type=int, default=20)
    parser.add_argument("--initial-min-per-command", type=int, default=1)
    parser.add_argument("--initial-max-medoid-samples", type=int, default=4096)
    parser.add_argument("--tau-split", type=float, choices=(0.8, 1.0, 1.2), default=1.0)
    parser.add_argument(
        "--tau-dedup", type=float, choices=(0.10, 0.15, 0.20, 0.25), default=0.15
    )
    parser.add_argument("--child-min-ratio", type=float, default=0.03)
    parser.add_argument("--max-split-medoid-samples", type=int, default=4096)
    parser.add_argument("--max-total-anchors", type=int, default=72)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    return parser.parse_args()


def _plot_history(history: list[dict], output_path: Path) -> None:
    anchor_count = [entry["anchor_count"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(anchor_count, [entry["p95_nearest_d_traj"] for entry in history], marker="o")
    axes[0].set_ylabel("P95 nearest D_traj")
    axes[1].plot(anchor_count, [entry["mean_nearest_ADE"] for entry in history], marker="o")
    axes[1].set_ylabel("Mean nearest ADE")
    axes[2].plot(anchor_count, [entry["p95_max_error"] for entry in history], marker="o")
    axes[2].set_ylabel("P95 max error")
    for axis in axes:
        axis.set_xlabel("Base anchor count K")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_diversity(diversity_metrics: dict, output_path: Path) -> None:
    distribution = diversity_metrics["pairwise_d_traj_distribution"]
    edges = np.asarray(distribution["histogram_edges"], dtype=np.float32)
    counts = np.asarray(distribution["histogram_counts"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7, 4))
    if len(counts):
        axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", edgecolor="black")
    axis.set_xlabel("Same-command pairwise D_traj")
    axis.set_ylabel("Anchor pair count")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    trajectories = np.load(args.trajectories)
    commands = np.load(args.commands)
    scales = load_calibration_scales(args.calibration)
    initial_config = InitialCommandAnchorBankConfig(
        total_anchors=args.initial_anchor_count,
        min_per_command=args.initial_min_per_command,
        max_medoid_samples=args.initial_max_medoid_samples,
        distance_batch_size=args.distance_batch_size,
        random_seed=args.random_seed,
    )
    initial_result = build_initial_command_anchor_bank(
        trajectories,
        commands,
        scales,
        initial_config,
    )
    config = CommandAnchorBankConfig(
        tau_split=args.tau_split,
        child_min_ratio=args.child_min_ratio,
        max_split_medoid_samples=args.max_split_medoid_samples,
        max_total_anchors=args.max_total_anchors,
        distance_batch_size=args.distance_batch_size,
        random_seed=args.random_seed,
        assignment_batch_size=args.assignment_batch_size,
    )

    split_result = build_command_anchor_bank(
        trajectories,
        commands,
        initial_result.anchors,
        initial_result.command_ids,
        scales,
        config,
    )
    result = deduplicate_command_anchor_bank(
        trajectories,
        commands,
        split_result,
        scales,
        threshold=args.tau_dedup,
        assignment_batch_size=args.assignment_batch_size,
    )
    coverage_metrics = command_anchor_coverage_metrics(
        trajectories,
        result.anchors,
        result.assignments,
        result.nearest_distances,
    )
    diversity_metrics = command_anchor_diversity_metrics(
        result.anchors,
        result.command_ids,
        scales,
        batch_size=args.distance_batch_size,
    )
    command_anchor_counts = {
        command.value: int(np.sum(result.command_ids == command_id))
        for command_id, command in enumerate(COMMAND_ORDER)
    }
    report = {
        "split": "navtrain",
        "config": asdict(config),
        "initial_config": asdict(initial_config),
        "command_order": [command.value for command in COMMAND_ORDER],
        "calibration_path": str(args.calibration),
        "initial_anchor_count": int(len(initial_result.anchors)),
        "initial_command_counts": initial_result.command_counts,
        "initial_command_anchor_counts": initial_result.anchor_counts,
        "initial_support": initial_result.support,
        "initial_source_indices": initial_result.source_indices,
        "pre_dedup_anchor_count": int(len(split_result.anchors)),
        "final_anchor_count": int(len(result.anchors)),
        "total_k": int(len(result.anchors)),
        "k_left": command_anchor_counts["left"],
        "k_straight": command_anchor_counts["straight"],
        "k_right": command_anchor_counts["right"],
        "dedup_threshold": args.tau_dedup,
        "dedup_removed_count": result.removed_count,
        "dedup_source_indices": result.source_indices,
        "stop_reason": split_result.stop_reason,
        "command_anchor_counts": command_anchor_counts,
        "split_history": split_result.history,
        "pre_dedup_cluster_statistics": split_result.cluster_stats,
        "post_dedup_support": result.support,
        "post_dedup_local_mean_d_traj": result.local_mean_distances,
        "coverage_metrics": coverage_metrics,
        "diversity_metrics": diversity_metrics,
    }
    paths = save_command_anchor_artifacts(
        args.output_dir,
        args.stem,
        result,
        report,
        scales,
    )
    initial_paths = save_initial_command_anchor_artifacts(
        args.output_dir,
        args.stem,
        initial_result,
    )
    _plot_history(split_result.history, args.output_dir / f"{args.stem}_base_expansion.png")
    diversity_plot_path = args.output_dir / f"{args.stem}_diversity.png"
    _plot_diversity(diversity_metrics, diversity_plot_path)
    print(f"Base expansion stopped by: {split_result.stop_reason}")
    print(
        "Anchor counts: "
        f"{len(initial_result.anchors)} -> {len(split_result.anchors)} -> {len(result.anchors)}"
    )
    print(f"Initial command bank: {initial_paths['npz']}")
    print(f"Visualization tensor: {paths['npy']}")
    print(f"Model-ready artifact: {paths['npz']}")
    print(f"Diversity histogram: {diversity_plot_path}")


if __name__ == "__main__":
    main()
