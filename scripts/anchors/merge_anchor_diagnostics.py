"""Merge all completed rank reports for a single epoch, outside training/DDP."""

import argparse
import json
from pathlib import Path

from navsim.agents.diffusiondrive.anchors.diagnostics import merge_rank_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True, help="Zero-based epoch (99 = training round 100)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob(f"epoch_{args.epoch:03d}_rank_*.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    merged = merge_rank_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Merged {len(paths)} rank reports into {args.output}")


if __name__ == "__main__":
    main()
