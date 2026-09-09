"""TRV 3.1.05_2: collect PCS hard decisions, train factor risks, and evaluate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from navsim.agents.diffusiondrive.pcs.common import (
    generate_context, load_torch, provenance, sha256, write_new_json,
)
from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.agents.diffusiondrive.pcs.veto import load_veto
from navsim.agents.diffusiondrive.pcs.veto_data import (
    DecisionPairDataset, prepare_pair_cache,
)
from navsim.planning.script.run_pcs import loader, new_run, prepare_source, write_csv


def prepare_pairs(args):
    prepare_pair_cache(
        args.cache, args.pcs_scorer, args.output, workers=args.workers,
        batch_size=args.batch_size, smoke=args.smoke,
    )


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from navsim.agents.diffusiondrive.pcs.veto_training import TripleRiskVetoModule

    pl.seed_everything(args.seed, workers=True)
    train_data = DecisionPairDataset(
        args.cache, args.pair_cache, "train", allow_partial=args.smoke,
    )
    val_data = DecisionPairDataset(
        args.cache, args.pair_cache, "val", allow_partial=args.smoke,
    )
    scorer_hash = sha256(args.pcs_scorer)
    if train_data.manifest["pcs_scorer_sha256"] != scorer_hash:
        raise ValueError("Decision pairs were produced by a different PCS checkpoint")
    metadata = {
        "schema": "triple_risk_veto_v1",
        "provenance": train_data.provenance,
        "pcs_scorer_sha256": scorer_hash,
        "pcs_settings": train_data.base.manifest["settings"],
        "hidden_dim": args.hidden_dim,
        "thresholds": [1.0, 1.0, 1.0],
    }
    if "TRV_RUN_DIR" not in os.environ:
        os.environ["TRV_RUN_DIR"] = str(new_run(args.output))
    run_dir = Path(os.environ["TRV_RUN_DIR"])
    module = TripleRiskVetoModule(
        metadata, args.pcs_scorer, lr=args.lr, epochs=args.epochs,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"), filename="epoch={epoch:02d}",
        auto_insert_metric_name=False, monitor="val/pdm", mode="max",
        save_top_k=1, save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        max_epochs=args.epochs, precision=args.precision,
        gradient_clip_val=1.0, num_sanity_val_steps=0, log_every_n_steps=20,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        logger=[
            CSVLogger(str(run_dir), name="csv"),
            TensorBoardLogger(str(run_dir), name="tensorboard"),
        ],
        callbacks=[checkpoint], default_root_dir=str(run_dir),
    )
    if trainer.is_global_zero:
        write_new_json(run_dir / "run.json", {
            "arguments": vars(args), "metadata": metadata,
            "pair_manifest": train_data.manifest,
            "train_scenes": len(train_data), "val_scenes": len(val_data),
            "global_batch_size": args.batch_size * args.devices,
        })
    if args.resume:
        previous = load_torch(args.resume)["veto_metadata"]
        if (
            previous["provenance"] != metadata["provenance"]
            or previous["pcs_scorer_sha256"] != scorer_hash
        ):
            raise ValueError("Resume checkpoint does not match PCS/candidate provenance")
    trainer.fit(
        module,
        loader(
            train_data, args.workers, batch_size=args.batch_size,
            shuffle=True, pin_memory=True,
        ),
        loader(
            val_data, args.workers, batch_size=args.batch_size,
            shuffle=False, pin_memory=True,
        ),
        ckpt_path=args.resume,
    )
    if trainer.is_global_zero:
        print(f"Best TRV checkpoint: {checkpoint.best_model_path}")
        print(f"Last TRV checkpoint: {checkpoint.last_model_path}")
        print("The original PCS checkpoint and frozen K67 generator remain unchanged.")


def evaluate(args):
    args.split = "navtest"
    args.feature_cache = None
    generator, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    veto, metadata = load_veto(args.veto, device, identity)
    if metadata["pcs_scorer_sha256"] != sha256(args.pcs_scorer):
        raise ValueError("TRV checkpoint was trained with a different PCS checkpoint")
    run_dir = new_run(args.output)
    write_new_json(run_dir / "run.json", {
        "arguments": vars(args), "provenance": identity,
        "veto_metadata": metadata,
    })
    stream = (run_dir / "paired_results.csv").open("x", encoding="utf-8", newline="")
    writer = None
    pending = []
    rows = []

    def finish(job):
        nonlocal writer
        future, record, decision = job
        labels, scores, direction = future.result()
        row = {"token": record["token"], "log_name": record["log_name"], "valid": True}
        for index, prefix in ((0, ""), (1, "base_"), (2, "pcs_")):
            row.update({
                prefix + key: float(labels[index, metric_index])
                for metric_index, key in enumerate(METRIC_NAMES)
            })
            row[prefix + "driving_direction_compliance"] = float(direction[index])
            row[prefix + "score"] = float(scores[index])
        row.update(decision)
        row["pdm_delta_vs_base"] = float(scores[0] - scores[1])
        row["pdm_delta_vs_pcs"] = float(scores[0] - scores[2])
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(
            max_workers=args.score_workers, mp_context=mp.get_context("spawn"),
        ) as pool:
            for index, (record, features) in enumerate(tqdm(
                loader(source, args.workers, batch_size=None), desc="TRV navtest",
            )):
                context = generate_context(
                    generator, features, record["token"], args.seed, device,
                )
                gpu_context = {
                    key: value.unsqueeze(0).to(device) for key, value in context.items()
                }
                with torch.no_grad():
                    output = veto(gpu_context)
                    if not torch.isfinite(output["risks"]).all():
                        raise ValueError(f"Nonfinite TRV risk: {record['token']}")
                    final_mode = int(output["final_mode"][0])
                    base_mode = int(output["base_mode"][0])
                    pcs_mode = int(output["pcs_mode"][0])
                    decision = {
                        "final_mode": final_mode,
                        "base_mode": base_mode,
                        "pcs_mode": pcs_mode,
                        "pcs_changed": pcs_mode != base_mode,
                        "accepted": final_mode == pcs_mode,
                        "vetoed": bool(output["vetoed"][0]),
                        "predicted_nc_risk": float(output["risks"][0, 0]),
                        "predicted_dac_risk": float(output["risks"][0, 1]),
                        "predicted_ttc_risk": float(output["risks"][0, 2]),
                    }
                proposals = context["proposals"][[final_mode, base_mode, pcs_mode]].numpy()
                future = pool.submit(
                    score_candidates, paths[record["token"]], proposals, index == 0,
                )
                pending.append((future, record, decision))
                if index == 0 or len(pending) >= 2 * args.score_workers:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
    finally:
        stream.close()
    if len(rows) != len(source):
        raise ValueError("Evaluation scene count mismatch")

    final_scores = np.asarray([row["score"] for row in rows])
    base_scores = np.asarray([row["base_score"] for row in rows])
    pcs_scores = np.asarray([row["pcs_score"] for row in rows])
    summary = {
        "scenes": len(rows),
        "pdm": float(final_scores.mean()),
        "pcs_pdm": float(pcs_scores.mean()),
        "base_pdm": float(base_scores.mean()),
        "delta_vs_pcs": float((final_scores - pcs_scores).mean()),
        "delta_vs_base": float((final_scores - base_scores).mean()),
        "vetoed_count": sum(row["vetoed"] for row in rows),
        "pcs_changed_count": sum(row["pcs_changed"] for row in rows),
        "accepted_switch_count": sum(
            row["accepted"] and row["pcs_changed"] for row in rows
        ),
        "pcs_new_zero": int(((base_scores > 0) & (pcs_scores == 0)).sum()),
        "final_new_zero": int(((base_scores > 0) & (final_scores == 0)).sum()),
        "pcs_rescued_zero": int(((base_scores == 0) & (pcs_scores > 0)).sum()),
        "final_rescued_zero": int(((base_scores == 0) & (final_scores > 0)).sum()),
        "completed": True,
    }
    for prefix, filename in (
        ("", "trv.csv"), ("pcs_", "pcs_proposer.csv"), ("base_", "base_selector.csv"),
    ):
        standard = [{
            "token": row["token"], "valid": row["valid"],
            **{
                key: row[prefix + key]
                for key in (*METRIC_NAMES, "driving_direction_compliance", "score")
            },
        } for row in rows]
        standard.append({
            "token": "average", "valid": True,
            **{
                key: float(np.mean([row[key] for row in standard]))
                for key in (*METRIC_NAMES, "driving_direction_compliance", "score")
            },
        })
        write_csv(run_dir / filename, standard)
    write_new_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Evaluation output: {run_dir}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    pairs = commands.add_parser("prepare-pairs")
    pairs.add_argument("--cache", required=True)
    pairs.add_argument("--pcs-scorer", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--workers", type=int, default=4)
    pairs.add_argument("--batch-size", type=int, default=32)
    pairs.add_argument("--smoke", action="store_true")

    training = commands.add_parser("train")
    training.add_argument("--cache", required=True)
    training.add_argument("--pair-cache", required=True)
    training.add_argument("--pcs-scorer", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--devices", type=int, default=4)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--hidden-dim", type=int, default=256)
    training.add_argument("--precision", default="16-mixed", choices=("16-mixed", "32-true"))
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--resume")

    evaluation = commands.add_parser("evaluate")
    for name in (
        "baseline", "backbone", "anchor", "metric_cache", "data_root", "output",
    ):
        evaluation.add_argument("--" + name.replace("_", "-"), required=True)
    evaluation.add_argument("--pcs-scorer", required=True)
    evaluation.add_argument("--veto", required=True)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--max-scenes", type=int, default=0)
    evaluation.add_argument("--workers", type=int, default=4)
    evaluation.add_argument("--score-workers", type=int, default=4)
    return root


def main():
    args = parser().parse_args()
    for name in ("workers", "max_scenes"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    for name in ("devices", "score_workers", "batch_size", "epochs", "hidden_dim"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    {"prepare-pairs": prepare_pairs, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
