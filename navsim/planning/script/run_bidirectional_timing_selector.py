"""Train an independent selector over the frozen 31-action timing bank."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from navsim.agents.diffusiondrive.pcs.common import sha256, write_new_json
from navsim.agents.diffusiondrive.timing_selector.data import TimingSelectorDataset
from navsim.agents.diffusiondrive.timing_selector.training import TimingSelectorModule


def _identity(args):
    source = Path(__file__).resolve().parents[2] / "agents/diffusiondrive/timing_selector"
    return {
        "schema": "bidirectional_timing_selector_v1",
        "feature_manifest_sha256": sha256(Path(args.features) / "manifest.json"),
        "train_oracle_manifest_sha256": sha256(Path(args.train_oracle) / "manifest.json"),
        "val_oracle_manifest_sha256": sha256(Path(args.val_oracle) / "manifest.json"),
        "source_sha256": {name: sha256(source / name)
                          for name in ("model.py", "data.py", "metrics.py", "training.py")},
    }


def train(args):
    pl.seed_everything(args.seed, workers=True)
    train_data = TimingSelectorDataset(args.features, args.train_oracle, "train")
    val_data = TimingSelectorDataset(args.features, args.val_oracle, "val")
    identity = _identity(args)
    settings = {"width": args.width}
    loss_settings = {"min_gap": args.min_gap, "temperature": args.temperature,
                     "cost_cap": args.cost_cap}
    module = TimingSelectorModule(identity, settings, loss_settings, val_data.compact.records, args.lr)
    root = Path(args.output) / datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    root.mkdir(parents=True, exist_ok=False)
    write_new_json(root / "run.json", {
        "identity": identity, "settings": settings, "loss_settings": loss_settings,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "workers": args.workers, "seed": args.seed, "devices": 1,
    })
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=root / "checkpoints", filename="epoch={epoch:02d}",
        monitor="calibration_pdm", mode="max", save_top_k=1, save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=args.epochs, precision="32-true",
        default_root_dir=str(root), callbacks=[callback], logger=pl.loggers.CSVLogger(root, "csv"),
        log_every_n_steps=50, num_sanity_val_steps=0, deterministic=True,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
    result = {"best_checkpoint": callback.best_model_path, "last_checkpoint": callback.last_model_path,
              "best_calibration_pdm": None if callback.best_model_score is None
              else float(callback.best_model_score)}
    write_new_json(root / "result.json", result)
    print("Best timing-selector checkpoint:", callback.best_model_path)
    print("Last timing-selector checkpoint:", callback.last_model_path)
    print("Run result:", root / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--val-oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--min-gap", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--cost-cap", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=31057)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("Invalid training settings")
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")
    train(args)


if __name__ == "__main__":
    main()
