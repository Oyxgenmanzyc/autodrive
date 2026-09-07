"""PCS v3.1.05: cache, diagnose, train, evaluate. See --help and docs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from navsim.agents.diffusiondrive.pcs.common import (
    build_generator, generate_context, provenance, write_new_json, load_torch,
    load_scorer, sha256, CONFIG_ROOT,
)
from navsim.agents.diffusiondrive.pcs.data import (
    FeatureSource, CandidateDataset, entry_path, metric_paths,
)
from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES, select_candidates
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates


def loader(dataset, workers, **kwargs):
    params = dict(num_workers=workers, **kwargs)
    if workers:
        params.update(multiprocessing_context="spawn", persistent_workers=True)
    return DataLoader(dataset, **params)


def new_run(root):
    path = Path(root) / datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_entry(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(entry, temporary)
    os.replace(temporary, path)


def prepare_source(args):
    device = "cuda:0"
    for name in ("baseline", "backbone", "anchor"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"Missing {name}: {getattr(args, name)}")
    if not Path(args.metric_cache).is_dir():
        raise FileNotFoundError(f"Missing metric cache directory: {args.metric_cache}")
    if not torch.cuda.is_available():
        raise RuntimeError("Run generator/cache/evaluation in navhigh on a visible GPU")
    model, config = build_generator(args.baseline, args.backbone, args.anchor, device)
    if args.split == "navtrain" and not args.feature_cache:
        raise ValueError("navtrain requires --feature-cache pointing at the existing K67 feature cache")
    source = FeatureSource(
        config, args.split, args.feature_cache, args.data_root, args.max_scenes, args.seed,
    )
    paths = metric_paths(args.metric_cache)
    missing = [r["token"] for r in source.records if r["token"] not in paths]
    if missing:
        raise ValueError(
            f"Missing metric cache for {len(missing)} requested scenes; example {missing[:3]}. "
            "Build metric cache for the correct split. No scenes were silently skipped."
        )
    return model, config, source, paths, device


def cache(args):
    model, config, source, paths, device = prepare_source(args)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    root = Path(args.output)
    identity = provenance(args.baseline, args.anchor, args.seed)
    manifest = {
        "provenance": identity, "dataset": args.split, "limit_per_split": args.max_scenes,
        "num_candidates": model._trajectory_head.ego_fut_mode,
        "feature_cache": str(Path(args.feature_cache).resolve()) if args.split == "navtrain" else None,
        "metric_cache": str(Path(args.metric_cache).resolve()),
        "settings": {
            "num_poses": config.trajectory_sampling.num_poses,
            "lidar_max_x": config.lidar_max_x, "lidar_max_y": config.lidar_max_y,
        },
        "num_shards": args.num_shards,
        "available_source_scenes": source.available_counts,
        "scoring_config_sha256": sha256(CONFIG_ROOT / "pdm_scoring/default_scoring_parameters.yaml"),
    }
    write_new_json(root / "manifest.json", manifest)
    write_new_json(root / "records.json", source.records)
    source.records = [r for i, r in enumerate(source.records) if i % args.num_shards == args.shard_index]
    expected_count = len(source.records)
    source.records = [r for r in source.records if not entry_path(root, r).exists()]
    print(f"Shard {args.shard_index}: {len(source)} pending of {expected_count}", flush=True)
    # Bound both GPU-feature memory and outstanding CPU scoring jobs.
    pending = []
    verified = False

    def finish(job):
        future, record, context = job
        labels, scores, direction = future.result()
        if labels.shape != (manifest["num_candidates"], 5):
            raise ValueError("Unexpected PDM label shape")
        entry = {
            "token": record["token"], "log_name": record["log_name"],
            "provenance": identity, "context": context,
            "labels": torch.from_numpy(labels), "scores": torch.from_numpy(scores),
            "direction": torch.from_numpy(direction),
        }
        save_entry(entry_path(root, record), entry)

    with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context("spawn")) as pool:
        for record, features in tqdm(loader(source, args.workers, batch_size=None), desc="PCS cache"):
            context = generate_context(model, features, record["token"], args.seed, device)
            future = pool.submit(
                score_candidates, paths[record["token"]], context["proposals"].numpy(), not verified,
            )
            job = (future, record, context)
            if not verified:
                finish(job)
                verified = True
                print("Batch/official PDM equivalence checked on three candidates", flush=True)
                size = entry_path(root, record).stat().st_size
                print(f"First cache entry: {size / 2**20:.2f} MiB; measure disk budget before full caching", flush=True)
                estimated = size * sum(source.available_counts.values()) / 2**30
                print(f"Available scenes {source.available_counts}; rough full candidate cache estimate {estimated:.1f} GiB", flush=True)
            else:
                pending.append(job)
            if len(pending) >= args.score_workers * 2:
                finish(pending.pop(0))
        for job in pending:
            finish(job)
    print(f"Cache shard complete: {root}", flush=True)


def diagnose(args):
    root = Path(args.cache)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    report_dir = new_run(args.output or root / "diagnostics")
    rows = []
    for record in tqdm(records, desc="Candidate upper bounds"):
        path = entry_path(root, record)
        if not path.exists():
            raise ValueError(f"Incomplete cache: {path}")
        entry = load_torch(path)
        if entry["provenance"] != manifest["provenance"] or entry["token"] != record["token"]:
            raise ValueError("Cache identity mismatch")
        scores = entry["scores"].numpy()
        order = entry["context"]["base_logits"].argsort(descending=True).numpy()
        base = scores[order[0]]
        row = {
            **record, "base_score": float(base), "oracle_score": float(scores.max()),
            "oracle_top5": float(scores[order[:5]].max()),
            "oracle_top10": float(scores[order[:10]].max()),
            "zero_rescuable": int(base == 0 and scores.max() > 0),
        }
        rows.append(row)
    summary = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = {"count": len(subset)}
        for key in ("base_score", "oracle_score", "oracle_top5", "oracle_top10", "zero_rescuable"):
            summary[split][key] = float(np.mean([r[key] for r in subset]))
    write_csv(report_dir / "candidate_diagnostics.csv", rows)
    write_new_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Diagnostic report: {report_dir}")


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from navsim.agents.diffusiondrive.pcs.training import PCSModule
    pl.seed_everything(args.seed, workers=True)
    train_data = CandidateDataset(args.cache, "train", allow_partial=args.smoke)
    val_data = CandidateDataset(args.cache, "val", allow_partial=args.smoke)
    manifest = train_data.manifest
    if val_data.provenance != train_data.provenance:
        raise ValueError("Train and validation cache provenance differs")
    run_root = Path(args.output)
    # Lightning DDP subprocesses must share precisely the same run directory.
    if "PCS_RUN_DIR" not in os.environ:
        os.environ["PCS_RUN_DIR"] = str(new_run(run_root))
    run_dir = Path(os.environ["PCS_RUN_DIR"])
    metadata = {"provenance": manifest["provenance"], "settings": manifest["settings"]}
    module = PCSModule(metadata, lr=args.lr, epochs=args.epochs)
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"), filename="epoch={epoch:02d}",
        auto_insert_metric_name=False, monitor="val/pdm", mode="max",
        save_top_k=1, save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        max_epochs=args.epochs, precision=args.precision,
        accumulate_grad_batches=1, gradient_clip_val=1.0,
        num_sanity_val_steps=0, log_every_n_steps=20,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        logger=[CSVLogger(str(run_dir), name="csv"), TensorBoardLogger(str(run_dir), name="tensorboard")],
        callbacks=[checkpoint], default_root_dir=str(run_dir),
    )
    if trainer.is_global_zero:
        write_new_json(run_dir / "run.json", {
            "arguments": vars(args), "cache_manifest": manifest,
            "train_scenes": len(train_data), "val_scenes": len(val_data),
            "global_batch_size": args.batch_size * args.devices,
        })
    if args.resume:
        previous = load_torch(args.resume)
        if previous["pcs_metadata"] != metadata:
            raise ValueError("Resume checkpoint and candidate cache differ")
    trainer.fit(
        module,
        loader(train_data, args.workers, batch_size=args.batch_size, shuffle=True, pin_memory=True),
        loader(val_data, args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True),
        ckpt_path=args.resume,
    )
    if trainer.is_global_zero:
        print(f"Best scorer checkpoint: {checkpoint.best_model_path}")
        print(f"Last scorer checkpoint: {checkpoint.last_model_path}")
        print("Frozen generator is external: retain the original baseline checkpoint.")


def write_csv(path, rows):
    with open(path, "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    # Raw navtest observations; metric cache is read ONLY after learned selection.
    args.split = "navtest"
    args.feature_cache = None
    model, config, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    scorer, _ = load_scorer(args.scorer, device, identity)
    run_dir = new_run(args.output)
    write_new_json(run_dir / "run.json", {"arguments": vars(args), "provenance": identity})
    # Preserve each completed row if a later scenario fails. Never mark a partial
    # run as successful or silently average only successful scenarios.
    stream = (run_dir / "paired_results.csv").open("x", encoding="utf-8", newline="")
    writer = None
    pending = []
    rows = []

    def finish(job):
        nonlocal writer
        future, record, selected, base = job
        labels, scores, direction = future.result()
        row = {"token": record["token"], "log_name": record["log_name"], "valid": True}
        for i, prefix in ((0, ""), (1, "base_")):
            row.update({prefix + key: float(labels[i, j]) for j, key in enumerate(METRIC_NAMES)})
            row[prefix + "driving_direction_compliance"] = float(direction[i])
            row[prefix + "score"] = float(scores[i])
        row.update(selected_mode=selected, base_mode=base, changed=selected != base,
                   pdm_delta=float(scores[0] - scores[1]))
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context("spawn")) as pool:
            for index, (record, features) in enumerate(tqdm(
                loader(source, args.workers, batch_size=None), desc="PCS navtest",
            )):
                context = generate_context(model, features, record["token"], args.seed, device)
                gpu_context = {k: v.unsqueeze(0).to(device) for k, v in context.items()}
                with torch.no_grad():
                    output = scorer(gpu_context)
                    if not torch.isfinite(output["scores"]).all():
                        raise ValueError(f"Nonfinite learned score: {record['token']}")
                    selection = select_candidates(gpu_context, output["scores"])
                    selected = int(selection["selected_mode"][0])
                    base = int(selection["base_mode"][0])
                proposals = context["proposals"][[selected, base]].numpy()
                future = pool.submit(score_candidates, paths[record["token"]], proposals, index == 0)
                pending.append((future, record, selected, base))
                if index == 0 or len(pending) >= 2 * args.score_workers:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
    finally:
        stream.close()
    if len(rows) != len(source):
        raise ValueError("Evaluation scene count mismatch")
    selected = np.array([r["score"] for r in rows])
    base = np.array([r["base_score"] for r in rows])
    summary = {
        "scenes": len(rows), "pdm": float(selected.mean()), "base_pdm": float(base.mean()),
        "delta": float((selected - base).mean()),
        "rescued_from_zero": int(((base == 0) & (selected > 0)).sum()),
        "new_zero": int(((base > 0) & (selected == 0)).sum()),
        "changed_count": sum(r["changed"] for r in rows),
        "completed": True,
    }
    # Standard NAVSIM-style CSVs from the SAME candidates, suitable for existing analysis.
    for prefix, filename in (("", "pcs.csv"), ("base_", "base_selector.csv")):
        standard = [{
            "token": r["token"], "valid": r["valid"],
            **{k: r[prefix + k] for k in (*METRIC_NAMES, "driving_direction_compliance", "score")},
        } for r in rows]
        standard.append({
            "token": "average", "valid": True,
            **{k: float(np.mean([r[k] for r in standard]))
               for k in (*METRIC_NAMES, "driving_direction_compliance", "score")},
        })
        write_csv(run_dir / filename, standard)
    write_new_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Evaluation output: {run_dir}")


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--baseline", required=True)
        p.add_argument("--backbone", required=True)
        p.add_argument("--anchor", required=True)
        p.add_argument("--metric-cache", required=True)
        p.add_argument("--data-root", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-scenes", type=int, default=0)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--score-workers", type=int, default=4)
        if name == "cache":
            p.add_argument("--split", choices=("navtrain", "navtest"), default="navtrain")
            p.add_argument("--feature-cache")
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--num-shards", type=int, default=1)
        else:
            p.add_argument("--scorer", required=True)
    p = subs.add_parser("diagnose")
    p.add_argument("--cache", required=True)
    p.add_argument("--output")
    p = subs.add_parser("train")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", default="16-mixed", choices=("16-mixed", "32-true"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", help="Resume scorer training, including optimizer state")
    return parser


def main():
    args = parser().parse_args()
    for name in ("workers", "max_scenes"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    for name in ("devices", "score_workers", "num_shards", "batch_size", "epochs"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
