"""Score one-sided braking and bidirectional fixed-path timing Oracles; no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from navsim.agents.diffusiondrive.cost_rank.data import CompactDataset, log_partitions
from navsim.agents.diffusiondrive.pcs.common import load_torch, sha256, write_new_json
from navsim.agents.diffusiondrive.pcs.data import metric_paths
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.agents.diffusiondrive.cost_rank.model import select
from navsim.agents.diffusiondrive.timing_oracle.geometry import (
    ACTIONS, ALL_INDICES, BACKLOAD_INDICES, BRAKE_INDICES, FRONTLOAD_INDICES,
    WARP_INDICES, action_bank, safe_oracle,
)
from navsim.planning.script.run_cost_rank import load_ranker
from navsim.planning.script.run_pcs import save_entry

BLOCK = 128
SCOPES = {
    "brake_only": BRAKE_INDICES,
    "warp_only": WARP_INDICES,
    "frontload_only": FRONTLOAD_INDICES,
    "backload_only": BACKLOAD_INDICES,
    "bidirectional": ALL_INDICES,
}


def _json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Conflicting metadata: {path}")
    else:
        write_new_json(path, value)


def _identity(args, data, rank_meta):
    runner = Path(__file__).resolve()
    geometry = runner.parents[2] / "agents/diffusiondrive/timing_oracle/geometry.py"
    return {
        "schema": "bidirectional_timing_oracle_v1",
        "feature_manifest_sha256": sha256(Path(args.features) / "manifest.json"),
        "records_sha256": sha256(Path(args.features) / f"{args.split}_records.json"),
        "ranker_sha256": sha256(args.ranker),
        "rank_policy": rank_meta["policy"],
        "metric_root": str(Path(args.metric_cache).resolve()),
        "split": args.split,
        "limit": args.limit,
        "actions": [list(action) for action in ACTIONS],
        "source_sha256": {
            runner.name: sha256(runner),
            geometry.name: sha256(geometry),
        },
        "scenes": min(len(data), args.limit) if args.limit else len(data),
    }


def prepare(args):
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard")
    data = CompactDataset(args.features, args.split)
    records = data.records[:args.limit] if args.limit else data.records
    ranker, rank_meta = load_ranker(args.ranker, "cuda:0")
    val_mismatch = (
        args.split == "val" and
        rank_meta["identity"]["val_records_sha256"] !=
        sha256(Path(args.features) / "val_records.json")
    )
    if rank_meta["identity"]["features"] != data.manifest or val_mismatch:
        raise ValueError("Ranker and compact feature cache do not match")
    identity = _identity(args, data, rank_meta)
    root = Path(args.output)
    _json(root / "manifest.json", identity)
    _json(root / f"{args.split}_records.json", records)
    paths = metric_paths(args.metric_cache)
    missing = [record["token"] for record in records if record["token"] not in paths]
    if missing:
        raise ValueError(f"Metric cache misses {len(missing)} tokens; first={missing[0]}")
    verified = False
    with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context("spawn")) as pool:
        for start in tqdm(range(0, len(records), BLOCK), desc=f"Timing Oracle {args.split} blocks"):
            block_index = start // BLOCK
            if block_index % args.num_shards != args.shard_index:
                continue
            block_records = records[start:start + BLOCK]
            target = root / args.split / f"block_{block_index:05d}.pt"
            if target.exists():
                old = load_torch(target)
                if old["tokens"] != [record["token"] for record in block_records]:
                    raise ValueError(f"Conflicting block: {target}")
                expected = (len(block_records), len(ACTIONS))
                if tuple(old["scores"].shape) != expected or tuple(old["labels"].shape) != (*expected, 5):
                    raise ValueError(f"Invalid completed block: {target}")
                continue
            entries, pending = [], []

            def finish(job):
                future, record, mode, cached_score = job
                labels, scores, direction = future.result()
                difference = float(scores[0] - cached_score)
                if abs(difference) > 0.05:
                    raise ValueError(f"Identity mismatch {difference:.6f}: {record['token']}")
                entries.append({
                    "mode": torch.tensor(mode), "cached_score": torch.tensor(cached_score),
                    "labels": torch.from_numpy(labels), "scores": torch.from_numpy(scores),
                    "direction": torch.from_numpy(direction),
                })

            for offset, record in enumerate(block_records):
                item = data[start + offset]
                gpu = {key: value.unsqueeze(0).to("cuda:0") for key, value in item.items()
                       if key in ("features", "subscores", "pcs_scores", "proposals", "base_logits")}
                with torch.no_grad():
                    residual = ranker(gpu)
                    mode = int(select(gpu["pcs_scores"], residual, rank_meta["policy"]["alpha"])[0])
                selected = item["proposals"][mode].numpy()
                variants = action_bank(selected)
                future = pool.submit(score_candidates, paths[record["token"]], variants, not verified)
                verified = True
                pending.append((future, record, mode, float(item["scores"][mode])))
                if len(pending) >= 2 * args.score_workers:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"tokens": [record["token"] for record in block_records],
                       "log_names": [record["log_name"] for record in block_records]}
            payload.update({key: torch.stack([entry[key] for entry in entries]) for key in entries[0]})
            save_entry(target, payload)
    print(f"Timing Oracle shard {args.shard_index} complete: {root}")


def _read(root, split, records):
    pieces = []
    for start in range(0, len(records), BLOCK):
        path = Path(root) / split / f"block_{start // BLOCK:05d}.pt"
        if not path.is_file():
            raise ValueError(f"Missing block: {path}")
        item = load_torch(path)
        if item["tokens"] != [record["token"] for record in records[start:start + BLOCK]]:
            raise ValueError(f"Token mismatch: {path}")
        pieces.append(item)
    return {key: torch.cat([piece[key] for piece in pieces]).numpy()
            for key in ("mode", "cached_score", "labels", "scores", "direction")}


def _cluster_ci(values, logs, seed=31056, samples=2000):
    logs = np.asarray(logs)
    unique = np.unique(logs)
    by_log = [np.asarray(values)[logs == log] for log in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(by_log), len(by_log))
        estimates[index] = np.concatenate([by_log[i] for i in chosen]).mean() * 100
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _report(data, records, mask):
    labels, scores, direction = data["labels"][mask], data["scores"][mask], data["direction"][mask]
    logs = np.asarray([record["log_name"] for record in records])[mask]
    result = {"scenes": int(mask.sum()), "identity_pdm": float(scores[:, 0].mean())}
    modes = {}
    for name, indices in SCOPES.items():
        mode = safe_oracle(labels, scores, direction, indices)
        modes[name] = mode
        rows = np.arange(len(mode))
        gain = scores[rows, mode] - scores[:, 0]
        families = [ACTIONS[index][0] for index in mode]
        result[name] = {
            "pdm": float(scores[rows, mode].mean()), "gain_points": float(gain.mean() * 100),
            "gain_95ci_by_log": _cluster_ci(gain, logs), "improved": int((gain > 1e-6).sum()),
            "action_family_counts": {family: families.count(family) for family in sorted(set(families))},
        }
    rows = np.arange(len(scores))
    brake_mode, both_mode = modes["brake_only"], modes["bidirectional"]
    incremental = scores[rows, both_mode] - scores[rows, brake_mode]
    result["bidirectional_vs_brake"] = {
        "gain_points": float(incremental.mean() * 100),
        "gain_95ci_by_log": _cluster_ci(incremental, logs, seed=31057),
        "improved_scenes": int((incremental > 1e-6).sum()),
        "criterion_gain_at_least_0_1_points": bool(incremental.mean() * 100 >= 0.1),
    }
    return result, modes


def diagnose(args):
    root = Path(args.cache)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((root / f"{manifest['split']}_records.json").read_text(encoding="utf-8"))
    data = _read(root, manifest["split"], records)
    calibration = log_partitions(records, 2) == 0
    masks = {"all": np.ones(len(records), bool), "calibration": calibration, "audit": ~calibration}
    summary, rows = {"manifest": manifest, "reports": {}}, []
    for split, mask in masks.items():
        report, modes = _report(data, records, mask)
        summary["reports"][split] = report
        indices = np.flatnonzero(mask)
        for local, index in enumerate(indices):
            row = {"token": records[index]["token"], "log_name": records[index]["log_name"],
                   "partition": split if split != "all" else "", "cached_pdm": float(data["cached_score"][index]),
                   "identity_pdm": float(data["scores"][index, 0])}
            for name in SCOPES:
                mode = int(modes[name][local])
                row[name + "_action"] = mode
                row[name + "_family"] = ACTIONS[mode][0]
                row[name + "_pdm"] = float(data["scores"][index, mode])
            if split != "all":
                rows.append(row)
    identity_error = data["scores"][:, 0] - data["cached_score"]
    summary["identity_check"] = {"mean_abs": float(np.abs(identity_error).mean()),
                                 "max_abs": float(np.abs(identity_error).max())}
    audit = summary["reports"]["audit"]["bidirectional_vs_brake"]
    threshold = 0.1
    supported = bool(audit["gain_points"] >= threshold and audit["gain_95ci_by_log"][0] > 0)
    decision = {
        "schema": "bidirectional_timing_oracle_decision_v1",
        "question": "Does a symmetric fixed-path timing action space add material Oracle value over extra braking only?",
        "audit_incremental_gain_points": audit["gain_points"],
        "audit_gain_95ci_by_log": audit["gain_95ci_by_log"],
        "required_gain_points": threshold,
        "pass": supported,
        "conclusion": (
            "Support training a bidirectional timing mode only."
            if supported else
            "Do not train a bidirectional timing mode from this fixed-path action family."
        ),
        "scope_limit": "Fixed four-second selected path; no spatial extension beyond its endpoint.",
    }
    summary["decision"] = decision
    write_new_json(root / "summary.json", summary)
    write_new_json(root / "decision.json", decision)
    with (root / "scene_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print("Scene results:", root / "scene_results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    for name in ("features", "ranker", "metric-cache", "output"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--split", default="val", choices=("train", "val"))
    prep.add_argument("--limit", type=int, default=0)
    prep.add_argument("--score-workers", type=int, default=4)
    prep.add_argument("--num-shards", type=int, default=1)
    prep.add_argument("--shard-index", type=int, default=0)
    diag = subs.add_parser("diagnose")
    diag.add_argument("--cache", required=True)
    args = parser.parse_args()
    if getattr(args, "limit", 0) < 0 or getattr(args, "score_workers", 1) < 1:
        raise ValueError("Invalid nonnegative limit or positive score-workers")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.multiprocessing.set_sharing_strategy("file_system")
    {"prepare": prepare, "diagnose": diagnose}[args.command](args)


if __name__ == "__main__":
    main()
