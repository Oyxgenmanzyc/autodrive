"""Build the command-conditioned calibration bank and fixed D_traj scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.anchors.calibration import (
    COMMAND_ORDER,
    CalibrationConfig,
    build_calibration_bank,
    save_calibration_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True, help="navtrain [N, 8, 2] NPY")
    parser.add_argument("--commands", type=Path, required=True, help="navtrain semantic [N] command NPY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="trajectory_distance_calibration")
    parser.add_argument("--total-anchors", type=int, default=60)
    parser.add_argument("--min-per-command", type=int, default=10)
    parser.add_argument("--max-kmedoids-samples", type=int, default=10_000)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = np.load(args.trajectories, allow_pickle=False)
    commands = np.load(args.commands, allow_pickle=False)
    config = CalibrationConfig(
        total_anchors=args.total_anchors,
        min_per_command=args.min_per_command,
        max_kmedoids_samples=args.max_kmedoids_samples,
        distance_batch_size=args.distance_batch_size,
        assignment_batch_size=args.assignment_batch_size,
        random_seed=args.random_seed,
    )
    result = build_calibration_bank(trajectories, commands, config)
    paths = save_calibration_artifacts(args.output_dir, result, config, stem=args.stem)

    print(f"Saved calibration bank: {paths['npz']}")
    print(f"Saved fixed D_traj scales: {paths['json']}")
    for command_id, command in enumerate(COMMAND_ORDER):
        scale = result.scale_for(command)
        print(
            f"{command.value}: trajectories={result.command_counts[command_id]}, "
            f"anchors={result.anchor_counts[command_id]}, "
            f"delta_scale={scale.delta_scale:.6f}, fde_scale={scale.fde_scale:.6f}"
        )


if __name__ == "__main__":
    main()
