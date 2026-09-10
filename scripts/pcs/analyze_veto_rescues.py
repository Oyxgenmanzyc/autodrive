"""Read-only TRV diagnosis. Excludes average rows; never emits a deployment rule."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def describe(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    def quantile(q):
        position = (len(values) - 1) * q
        index = int(position)
        return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position-index)
    return {"count": len(values), "mean": statistics.mean(values),
            **{name: quantile(q) for name, q in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))}}


def analyze(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["token"] != "average" and r["valid"].lower() == "true"]
    if len({r["token"] for r in rows}) != len(rows):
        raise ValueError("Duplicate scene tokens")
    result = {"scenes": len(rows), "purpose": "diagnostic only, not threshold calibration"}
    for name, positive in (("correct_veto", True), ("wrong_veto", False)):
        group = [r for r in rows if r["vetoed"].lower() == "true"
                 and ((float(r["base_score"])-float(r["pcs_score"])) * (1 if positive else -1) > 1e-8)]
        result[name] = {
            "actual_gain_magnitude": describe([abs(float(r["pcs_score"])-float(r["base_score"])) for r in group]),
            "predicted_risks": {k: describe([float(r[k]) for r in group])
                                for k in ("predicted_nc_risk", "predicted_dac_risk", "predicted_ttc_risk")},
        }
    result["limitation"] = "Old TRV CSV has no predicted PDM gain; actual gain cannot be an inference threshold."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_csv")
    args = parser.parse_args()
    print(json.dumps(analyze(args.paired_csv), indent=2))
