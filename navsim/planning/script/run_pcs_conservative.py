"""PCS 3.1.05_2: compact GTRS augmentation and conservative base-relative selection."""
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
    generate_context, provenance, load_torch,
)
from navsim.agents.diffusiondrive.pcs.conservative import (
    initialize_from_pcs, load_conservative_scorer, scorer_output, select_candidates,
)
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
from navsim.agents.diffusiondrive.pcs.gtrs import (
    GTRSAugmentedDataset, prepare_compact_cache,
)
from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.planning.script.run_pcs import loader, new_run, prepare_source, write_csv


def prepare_gtrs(args):
    prepare_compact_cache(
        args.cache, args.gtrs_pdm, args.gtrs_vocabulary, args.output,
        count=args.samples, seed=args.seed,
    )


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from navsim.agents.diffusiondrive.pcs.conservative_training import (
        ConservativeAdvantageModule,
    )

    pl.seed_everything(args.seed, workers=True)
    train_base = CandidateDataset(args.cache, "train", allow_partial=args.smoke)
    train_data = GTRSAugmentedDataset(train_base, args.gtrs_cache)
    val_data = CandidateDataset(args.cache, "val", allow_partial=args.smoke)
    if val_data.provenance != train_base.provenance:
        raise ValueError("Train and validation cache provenance differs")

    if "PCS_RUN_DIR" not in os.environ:
        os.environ["PCS_RUN_DIR"] = str(new_run(args.output))
    run_dir = Path(os.environ["PCS_RUN_DIR"])
    settings = {
        **train_base.manifest["settings"],
        "min_delta": args.min_delta,
        "min_win_probability": args.min_win_probability,
        "max_catastrophic_risk": args.max_catastrophic_risk,
        "risk_penalty": args.risk_penalty,
    }
    metadata = {
        "scorer_type": "conservative_advantage_v1",
        "provenance": train_base.manifest["provenance"],
        "settings": settings,
    }
    module = ConservativeAdvantageModule(metadata, lr=args.lr, epochs=args.epochs)
    if args.resume and args.init_scorer:
        raise ValueError("Use either --resume or --init-scorer, not both")
    if args.init_scorer:
        initialize_from_pcs(module.head, args.init_scorer, train_base.provenance)
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
        logger=[
            CSVLogger(str(run_dir), name="csv"),
            TensorBoardLogger(str(run_dir), name="tensorboard"),
        ],
        callbacks=[checkpoint], default_root_dir=str(run_dir),
    )
    if trainer.is_global_zero:
        (run_dir / "run.json").write_text(json.dumps({
            "arguments": vars(args),
            "cache_manifest": train_base.manifest,
            "gtrs_manifest": json.loads(
                (Path(args.gtrs_cache) / "manifest.json").read_text(encoding="utf-8")
            ),
            "train_scenes": len(train_data),
            "val_scenes": len(val_data),
            "global_batch_size": args.batch_size * args.devices,
            "metadata": metadata,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.resume:
        previous = load_torch(args.resume)
        if previous["pcs_metadata"] != metadata:
            raise ValueError("Resume checkpoint metadata differs from this experiment")
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
        print(f"Best conservative scorer checkpoint: {checkpoint.best_model_path}")
        print(f"Last conservative scorer checkpoint: {checkpoint.last_model_path}")
        print("Frozen K67 generator and original 3.1.05 scorer remain external and unchanged.")


def evaluate(args):
    args.split = "navtest"
    args.feature_cache = None
    model, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    scorer, _ = load_conservative_scorer(args.scorer, device, identity)
    for name in (
        "min_delta", "min_win_probability", "max_catastrophic_risk", "risk_penalty",
    ):
        value = getattr(args, name)
        if value is not None:
            scorer.settings[name] = value
    run_dir = new_run(args.output)
    (run_dir / "run.json").write_text(json.dumps({
        "arguments": vars(args), "provenance": identity,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    stream = (run_dir / "paired_results.csv").open("x", encoding="utf-8", newline="")
    writer = None
    pending = []
    rows = []

    def finish(job):
        nonlocal writer
        future, record, decision = job
        labels, scores, direction = future.result()
        row = {"token": record["token"], "log_name": record["log_name"], "valid": True}
        for i, prefix in ((0, ""), (1, "base_")):
            row.update({prefix + key: float(labels[i, j]) for j, key in enumerate(METRIC_NAMES)})
            row[prefix + "driving_direction_compliance"] = float(direction[i])
            row[prefix + "score"] = float(scores[i])
        row.update(decision)
        row["pdm_delta"] = float(scores[0] - scores[1])
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context("spawn")) as pool:
            for index, (record, features) in enumerate(tqdm(
                loader(source, args.workers, batch_size=None), desc="Conservative PCS navtest",
            )):
                context = generate_context(model, features, record["token"], args.seed, device)
                gpu_context = {key: value.unsqueeze(0).to(device) for key, value in context.items()}
                with torch.no_grad():
                    output = scorer_output(scorer, gpu_context)
                    if not all(torch.isfinite(output[key]).all() for key in (
                        "scores", "predicted_delta", "win_probability", "catastrophic_risk",
                    )):
                        raise ValueError(f"Nonfinite conservative score: {record['token']}")
                    selection = select_candidates(gpu_context, output)
                    selected = int(selection["selected_mode"][0])
                    base = int(selection["base_mode"][0])
                    decision = {
                        "selected_mode": selected,
                        "raw_selected_mode": int(selection["raw_selected_mode"][0]),
                        "base_mode": base,
                        "changed": selected != base,
                        "proposed_change": int(selection["raw_selected_mode"][0]) != base,
                        "predicted_delta": float(selection["predicted_delta"][0]),
                        "predicted_win_probability": float(selection["win_probability"][0]),
                        "predicted_catastrophic_risk": float(selection["catastrophic_risk"][0]),
                    }
                proposals = context["proposals"][[selected, base]].numpy()
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

    selected_scores = np.asarray([row["score"] for row in rows])
    base_scores = np.asarray([row["base_score"] for row in rows])
    summary = {
        "scenes": len(rows),
        "pdm": float(selected_scores.mean()),
        "base_pdm": float(base_scores.mean()),
        "delta": float((selected_scores - base_scores).mean()),
        "rescued_from_zero": int(((base_scores == 0) & (selected_scores > 0)).sum()),
        "new_zero": int(((base_scores > 0) & (selected_scores == 0)).sum()),
        "changed_count": sum(row["changed"] for row in rows),
        "proposed_change_count": sum(row["proposed_change"] for row in rows),
        "completed": True,
    }
    for prefix, filename in (("", "conservative_pcs.csv"), ("base_", "base_selector.csv")):
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
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Evaluation output: {run_dir}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    compact = commands.add_parser("prepare-gtrs")
    compact.add_argument("--cache", required=True)
    compact.add_argument("--gtrs-pdm", required=True)
    compact.add_argument("--gtrs-vocabulary", required=True)
    compact.add_argument("--output", required=True)
    compact.add_argument("--samples", type=int, default=32)
    compact.add_argument("--seed", type=int, default=0)

    training = commands.add_parser("train")
    training.add_argument("--cache", required=True)
    training.add_argument("--gtrs-cache", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--devices", type=int, default=4)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--precision", default="16-mixed", choices=("16-mixed", "32-true"))
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--resume")
    training.add_argument(
        "--init-scorer", help="Warm-start shared weights from the original 3.1.05 PCS checkpoint",
    )
    training.add_argument("--min-delta", type=float, default=0.01)
    training.add_argument("--min-win-probability", type=float, default=0.55)
    training.add_argument("--max-catastrophic-risk", type=float, default=0.10)
    training.add_argument("--risk-penalty", type=float, default=0.50)

    evaluation = commands.add_parser("evaluate")
    for name in ("baseline", "backbone", "anchor", "metric_cache", "data_root", "output"):
        evaluation.add_argument("--" + name.replace("_", "-"), required=True)
    evaluation.add_argument("--scorer", required=True)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--max-scenes", type=int, default=0)
    evaluation.add_argument("--workers", type=int, default=4)
    evaluation.add_argument("--score-workers", type=int, default=4)
    evaluation.add_argument("--min-delta", type=float)
    evaluation.add_argument("--min-win-probability", type=float)
    evaluation.add_argument("--max-catastrophic-risk", type=float)
    evaluation.add_argument("--risk-penalty", type=float)
    return root


def main():
    args = parser().parse_args()
    if hasattr(args, "samples") and args.samples <= 0:
        raise ValueError("samples must be positive")
    if hasattr(args, "devices") and args.devices <= 0:
        raise ValueError("devices must be positive")
    for name in ("min_win_probability", "max_catastrophic_risk"):
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
    if hasattr(args, "risk_penalty") and args.risk_penalty is not None and args.risk_penalty < 0:
        raise ValueError("risk_penalty cannot be negative")
    {"prepare-gtrs": prepare_gtrs, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
