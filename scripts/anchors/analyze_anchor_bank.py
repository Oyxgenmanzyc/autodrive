"""Compare one or more anchor banks on the same extracted navtrain trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navsim.agents.diffusiondrive.anchors.metrics import evaluate_anchor_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument(
        "--anchor-bank",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for Original, Base, and Base+Residual banks",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = np.load(args.trajectories)
    report = {}
    for specification in args.anchor_bank:
        if "=" not in specification:
            raise ValueError(f"Expected NAME=PATH, got {specification!r}")
        name, path = specification.split("=", 1)
        anchors = np.load(Path(path))
        report[name] = evaluate_anchor_bank(
            trajectories,
            anchors,
            batch_size=args.assignment_batch_size,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
