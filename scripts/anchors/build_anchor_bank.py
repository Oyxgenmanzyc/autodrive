"""Build a coverage-guided, root-conditioned adaptive DiffusionDrive anchor bank."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navsim.agents.diffusiondrive.anchors.base_expander import AnchorBuilderConfig, expand_base_anchors
from navsim.agents.diffusiondrive.anchors.composer import compose_anchor_bank
from navsim.agents.diffusiondrive.anchors.io import save_anchor_artifacts
from navsim.agents.diffusiondrive.anchors.metrics import evaluate_anchor_bank
from navsim.agents.diffusiondrive.anchors.residual_codebook import learn_residual_codebooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True, help="navtrain [N, 8, 2] NPY")
    parser.add_argument("--initial-anchors", type=Path, required=True, help="original 20-anchor NPY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="adaptive_anchor_bank")
    parser.add_argument("--coverage-epsilon", type=float, default=1.0)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    parser.add_argument("--coverage-error-threshold", type=float, default=1.0)
    parser.add_argument("--variance-threshold", type=float, default=0.10)
    parser.add_argument("--min-split-samples", type=int, default=50)
    parser.add_argument("--min-coverage-gain", type=float, default=1e-3)
    parser.add_argument("--min-ade-gain", type=float, default=1e-3)
    parser.add_argument("--saturation-patience", type=int, default=3)
    parser.add_argument("--max-base-anchors", type=int, default=64)
    parser.add_argument("--residual-modes", type=int, default=8, help="total modes per root, including zero")
    parser.add_argument("--residual-min-support", type=int, default=20)
    parser.add_argument("--residual-min-support-ratio", type=float, default=0.02)
    parser.add_argument("--dedup-threshold", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    return parser.parse_args()


def _plot_history(history: list[dict], output_path: Path) -> None:
    anchor_count = [entry["anchor_count"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(anchor_count, [entry["coverage@1m"] for entry in history], marker="o")
    axes[0].set_ylabel("Coverage@1m")
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


def main() -> None:
    args = parse_args()
    trajectories = np.load(args.trajectories)
    initial_anchors = np.load(args.initial_anchors)
    config = AnchorBuilderConfig(
        coverage_epsilon=args.coverage_epsilon,
        coverage_target=args.coverage_target,
        coverage_error_threshold=args.coverage_error_threshold,
        variance_threshold=args.variance_threshold,
        min_split_samples=args.min_split_samples,
        min_coverage_gain=args.min_coverage_gain,
        min_ade_gain=args.min_ade_gain,
        saturation_patience=args.saturation_patience,
        max_base_anchors=args.max_base_anchors,
        residual_modes=args.residual_modes,
        residual_min_support=args.residual_min_support,
        residual_min_support_ratio=args.residual_min_support_ratio,
        dedup_threshold=args.dedup_threshold,
        random_seed=args.random_seed,
        assignment_batch_size=args.assignment_batch_size,
    )

    base_result = expand_base_anchors(trajectories, initial_anchors, config)
    residual_result = learn_residual_codebooks(
        trajectories,
        base_result.base_anchors,
        base_result.root_ids,
        base_result.assignments,
        config,
    )
    composition = compose_anchor_bank(
        base_result.base_anchors,
        base_result.root_ids,
        residual_result,
        config,
    )
    report = {
        "split": "navtrain",
        "config": asdict(config),
        "initial_anchor_count": int(len(initial_anchors)),
        "base_anchor_count": int(len(base_result.base_anchors)),
        "final_anchor_count": int(len(composition.anchors)),
        "base_stop_reason": base_result.stop_reason,
        "initial_metrics": evaluate_anchor_bank(trajectories, initial_anchors, batch_size=config.assignment_batch_size),
        "base_metrics": evaluate_anchor_bank(
            trajectories, base_result.base_anchors, batch_size=config.assignment_batch_size
        ),
        "final_metrics": evaluate_anchor_bank(
            trajectories, composition.anchors, batch_size=config.assignment_batch_size
        ),
        "split_history": base_result.history,
        "final_cluster_statistics": base_result.cluster_stats,
        "residual_distortion_curves": residual_result.distortion_curves,
        "residual_support": residual_result.support_count,
        "dedup_removed_count": composition.dedup_removed_count,
    }
    paths = save_anchor_artifacts(
        args.output_dir,
        args.stem,
        base_result,
        residual_result,
        composition,
        report,
    )
    _plot_history(base_result.history, args.output_dir / f"{args.stem}_base_expansion.png")
    print(f"Base expansion stopped by: {base_result.stop_reason}")
    print(f"Anchor counts: {len(initial_anchors)} -> {len(base_result.base_anchors)} -> {len(composition.anchors)}")
    print(f"Base-only anchor bank: {paths['base_npy']}")
    print(f"Model-ready anchor bank: {paths['npy']}")


if __name__ == "__main__":
    main()
