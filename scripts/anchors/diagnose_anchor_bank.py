"""Write support/diversity JSON and plots without modifying an anchor bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.anchors.diagnostics import audit_anchor_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--anchor-bank", type=Path, required=True, help="Original 3.1.02 model-ready NPY")
    parser.add_argument("--output-dir", type=Path, required=True, help="Use a new directory for each audit")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--plot", action="store_true", help="Also export overview, individual anchors and support PNGs")
    args = parser.parse_args()
    if args.anchor_bank.suffix.lower() != ".npy":
        parser.error("Use the same .npy bank as original 3.1.02 training, not a command .npz bank")
    trajectories = np.load(args.trajectories, allow_pickle=False)
    anchors = np.load(args.anchor_bank, allow_pickle=False)
    report = audit_anchor_bank(trajectories, anchors, args.assignment_batch_size)
    report["anchor_path"] = str(args.anchor_bank.resolve())
    report["trajectory_path"] = str(args.trajectories.resolve())
    report["anchor_file_sha256"] = hashlib.sha256(args.anchor_bank.read_bytes()).hexdigest()
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "anchor_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    usage = report["static_assignment"]
    with (args.output_dir / "anchor_support.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["anchor_index", "GT_assignment_count", "GT_assignment_frequency"])
        writer.writerows(zip(range(len(anchors)), usage["counts"], usage["frequency"]))
    if args.plot:
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(anchors)))
        fig, ax = plt.subplots(figsize=(9, 9))
        for index, anchor in enumerate(anchors):
            xy = np.concatenate([np.zeros((1, 2)), anchor])
            ax.plot(xy[:, 0], xy[:, 1], color=colors[index], alpha=0.7)
            ax.text(*anchor[-1], str(index), fontsize=6)
        ax.set(xlabel="Forward x (m)", ylabel="Left y (m)", title=f"All {len(anchors)} original 3.1.02 anchors")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        fig.savefig(args.output_dir / "anchor_overview.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        # Shared axes prevent short and long trajectories from looking equally large.
        columns = 5
        rows = (len(anchors) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows), sharex=True, sharey=True, squeeze=False)
        for index, ax in enumerate(axes.flat):
            if index >= len(anchors):
                ax.set_visible(False)
                continue
            xy = np.concatenate([np.zeros((1, 2)), anchors[index]])
            ax.plot(xy[:, 0], xy[:, 1], "o-", markersize=2, color=colors[index])
            ax.set_title(f"#{index}: {usage['counts'][index]} GT", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "anchors_individual.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(range(len(anchors)), usage["counts"])
        axes[0].set(xlabel="Anchor index", ylabel="GT assignments", title="Static support (not trained selection)")
        nearest = report["diversity"]["nearest_neighbor_ADE_m"]
        if nearest:
            axes[1].hist(nearest, bins=min(20, len(nearest)))
        axes[1].set(xlabel="Nearest other anchor ADE (m)", ylabel="Anchor count", title="Diversity; excludes diagonal")
        fig.tight_layout()
        fig.savefig(args.output_dir / "anchor_support_diversity.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report["coverage"], indent=2))
    print(f"Saved read-only anchor audit: {args.output_dir}")


if __name__ == "__main__":
    main()
