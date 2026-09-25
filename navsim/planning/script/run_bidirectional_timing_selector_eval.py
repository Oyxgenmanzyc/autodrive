"""Locked navtest evaluation for the independently trained timing selector."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from navsim.agents.diffusiondrive.cost_rank.model import FrozenPCS, select
from navsim.agents.diffusiondrive.pcs.common import (
    generate_context, load_scorer, load_torch, provenance, sha256, write_new_json,
)
from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.agents.diffusiondrive.timing_oracle.geometry import ACTIONS, action_bank
from navsim.agents.diffusiondrive.timing_selector.model import BidirectionalTimingSelector
from navsim.planning.script.run_cost_rank import load_ranker
from navsim.planning.script.run_pcs import loader, new_run, prepare_source, write_csv


def _selector_sources():
    root = Path(__file__).resolve().parents[2] / "agents/diffusiondrive/timing_selector"
    return {name: sha256(root / name) for name in ("model.py", "data.py", "metrics.py", "training.py")}


def load_selector(path, features, val_oracle, device):
    checkpoint = load_torch(path)
    metadata = checkpoint.get("timing_selector_metadata")
    if not metadata or metadata["identity"]["schema"] != "bidirectional_timing_selector_v1":
        raise ValueError("Unexpected timing-selector checkpoint")
    identity = metadata["identity"]
    if identity["source_sha256"] != _selector_sources():
        raise ValueError("Timing-selector implementation differs from training")
    if identity["feature_manifest_sha256"] != sha256(Path(features) / "manifest.json"):
        raise ValueError("Timing selector and frozen feature identity differ")
    if identity["val_oracle_manifest_sha256"] != sha256(Path(val_oracle) / "manifest.json"):
        raise ValueError("Timing selector and validation Oracle identity differ")
    model = BidirectionalTimingSelector(width=metadata["settings"]["width"])
    model.load_state_dict({key[6:]: value for key, value in checkpoint["state_dict"].items()
                           if key.startswith("model.")})
    policy = metadata["policy"]
    if policy.get("enabled") is not True or not np.isfinite(policy.get("threshold", np.nan)):
        raise ValueError("Timing-selector checkpoint has no enabled locked policy")
    return model.eval().to(device), metadata


def select_timing_action(logits, policy):
    if logits.ndim != 2 or logits.shape[1] != len(ACTIONS) or not torch.isfinite(logits).all():
        raise ValueError("Invalid timing logits")
    action = int(logits[0].argmax())
    confidence = float(logits[0, action])
    if not policy["enabled"] or confidence <= policy["threshold"]:
        action = 0
    return action, confidence


def evaluate(args):
    args.split, args.feature_cache = "navtest", None
    generator, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    ranker, rank_meta = load_ranker(args.ranker, device, allow_smoke=bool(args.max_scenes))
    feature_manifest = json.loads((Path(args.features) / "manifest.json").read_text(encoding="utf-8"))
    if rank_meta["identity"]["features"] != feature_manifest:
        raise ValueError("Ranker and frozen feature identity differ")
    if feature_manifest["provenance"] != identity or feature_manifest["pcs_sha256"] != sha256(args.scorer):
        raise ValueError("Ranker does not match generator/PCS")
    scorer, _ = load_scorer(args.scorer, device, identity)
    frozen = FrozenPCS(scorer).eval()
    timing, timing_meta = load_selector(args.timing_selector, args.features, args.val_oracle, device)
    rank_policy, timing_policy = rank_meta["policy"], timing_meta["policy"]
    run = new_run(args.output)
    write_new_json(run / "run.json", {
        "arguments": vars(args), "rank_policy": rank_policy, "timing_policy": timing_policy,
        "ranker_sha256": sha256(args.ranker), "timing_selector_sha256": sha256(args.timing_selector),
    })
    prefixes = ("timing", "rank", "pcs", "base")
    rows, pending = [], []
    stream = (run / "paired_results.csv").open("x", newline="", encoding="utf-8")
    writer = None

    def finish(job):
        nonlocal writer
        future, record, modes, action, confidence = job
        labels, scores, direction = future.result()
        row = {"token": record["token"], "log_name": record["log_name"], "valid": True,
               "timing_action": action, "timing_family": ACTIONS[action][0],
               "timing_confidence": confidence, "timing_changed": action != 0}
        for index, prefix in enumerate(prefixes):
            row.update({f"{prefix}_{name}": float(labels[index, j])
                        for j, name in enumerate(METRIC_NAMES)})
            row[f"{prefix}_score"] = float(scores[index])
            row[f"{prefix}_direction"] = float(direction[index])
        row.update(rank_mode=modes[0], pcs_mode=modes[1], base_mode=modes[2],
                   delta_vs_rank=float(scores[0] - scores[1]),
                   delta_vs_pcs=float(scores[0] - scores[2]))
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(max_workers=args.score_workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            for index, (record, features) in enumerate(tqdm(
                    loader(source, args.workers, batch_size=None),
                    desc="Locked Rank + bidirectional Timing navtest")):
                context = generate_context(generator, features, record["token"], args.seed, device)
                gpu = {key: value.unsqueeze(0).to(device) for key, value in context.items()}
                with torch.no_grad():
                    compact = frozen(gpu)
                    residual = ranker(compact)
                    rank_mode = int(select(compact["pcs_scores"], residual, rank_policy["alpha"])[0])
                    pcs_mode = int(compact["pcs_scores"].argmax(-1)[0])
                    base_mode = int(gpu["base_logits"].argmax(-1)[0])
                    selector_batch = {
                        "feature": compact["features"][:, rank_mode],
                        "subscores": compact["subscores"][:, rank_mode],
                        "pcs_score": compact["pcs_scores"][:, rank_mode],
                        "base_probability": compact["base_logits"].float().softmax(-1)[:, rank_mode],
                        "proposal": compact["proposals"][:, rank_mode],
                    }
                    logits = timing(selector_batch)
                    action, confidence = select_timing_action(logits, timing_policy)
                selected = context["proposals"][rank_mode].numpy()
                timing_trajectory = action_bank(selected)[action]
                trajectories = np.stack([
                    timing_trajectory, selected, context["proposals"][pcs_mode].numpy(),
                    context["proposals"][base_mode].numpy(),
                ])
                future = pool.submit(score_candidates, paths[record["token"]], trajectories, index == 0)
                pending.append((future, record, (rank_mode, pcs_mode, base_mode), action, confidence))
                if index == 0 or len(pending) >= 2 * args.score_workers:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
    finally:
        stream.close()
    if len(rows) != len(source):
        raise ValueError("Incomplete timing-selector navtest evaluation")
    summary = {"scenes": len(rows), "completed": True,
               "rank_policy": rank_policy, "timing_policy": timing_policy,
               "timing_changed": int(sum(row["timing_changed"] for row in rows)),
               "timing_action_family_counts": {family: sum(row["timing_family"] == family for row in rows)
                                                for family in sorted({x[0] for x in ACTIONS})}}
    for prefix in prefixes:
        standard = [{"token": row["token"], "valid": True,
                     **{name: row[f"{prefix}_{name}"] for name in METRIC_NAMES},
                     "driving_direction_compliance": row[f"{prefix}_direction"],
                     "score": row[f"{prefix}_score"]} for row in rows]
        average = {key: float(np.mean([row[key] for row in standard]))
                   for key in (*METRIC_NAMES, "driving_direction_compliance", "score")}
        write_csv(run / f"{prefix}.csv", standard + [{"token": "average", "valid": True, **average}])
        summary[prefix] = average
    for reference in ("rank", "pcs", "base"):
        delta = np.asarray([row["timing_score"] - row[f"{reference}_score"] for row in rows])
        summary[f"vs_{reference}"] = {
            "gain_points": float(delta.mean() * 100), "beneficial": int((delta > 1e-6).sum()),
            "harmful": int((delta < -1e-6).sum()), "severe_losses": int((delta <= -0.2).sum()),
        }
    write_new_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Locked timing-selector navtest:", run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "backbone", "anchor", "scorer", "ranker", "timing-selector",
                 "features", "val-oracle", "metric-cache", "data-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-workers", type=int, default=2)
    args = parser.parse_args()
    if args.max_scenes < 0 or args.workers < 0 or args.score_workers < 1:
        raise ValueError("Invalid evaluation settings")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy("file_system")
    evaluate(args)


if __name__ == "__main__":
    main()
