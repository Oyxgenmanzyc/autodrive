"""Compare equal-duration generator arms under the identical frozen selector."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def compare(control_dir, timing_dir):
    runs = [json.loads((Path(p) / "run.json").read_text()) for p in (control_dir, timing_dir)]
    metadata = [r["training_metadata"] for r in runs]
    if [m["arm"] for m in metadata] != ["control", "timing"]:
        raise ValueError("Expected control followed by timing evaluation")
    for key in ("baseline_provenance", "veto_sha256", "pcs_sha256", "thresholds", "target_sha256",
                "epochs", "lr", "seed", "global_batch_size", "precision", "sources", "scope", "smoke"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Unmatched experiment setting: {key}")
    tables = []
    for directory in (control_dir, timing_dir):
        summary = json.loads((Path(directory) / "summary.json").read_text())
        if not summary["completed"]:
            raise ValueError("Incomplete evaluation")
        with (Path(directory) / "paired_results.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        table = {r["token"]: r for r in rows}
        if len(table) != len(rows) or len(rows) != summary["scenes"]:
            raise ValueError("Duplicate/missing evaluation scenes")
        tables.append(table)
    if set(tables[0]) != set(tables[1]):
        raise ValueError("Scene sets differ")
    tokens = sorted(tables[0])
    for token in tokens:
        if abs(float(tables[0][token]["original_final_pdm"]) - float(tables[1][token]["original_final_pdm"])) > 1e-6:
            raise ValueError("Original generator/selector evaluation is not reproducible")
    result = {"scenes": len(tokens), "comparison": "timing minus equal-duration control"}
    keys = [k for k in tables[0][tokens[0]] if k.startswith("new_") and not k.endswith("_mode")
            and k not in ("new_vetoed", "new_has_safe_candidate")]
    for key in keys:
        values = [np.array([float(t[token][key]) for token in tokens]) for t in tables]
        result[key] = {"control": float(values[0].mean()), "timing": float(values[1].mean()),
                       "delta": float((values[1] - values[0]).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--timing", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.timing), indent=2))


if __name__ == "__main__":
    main()
