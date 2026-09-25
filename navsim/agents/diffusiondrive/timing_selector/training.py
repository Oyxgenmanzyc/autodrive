"""Lightning module with one independent timing-action ranking objective."""
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from navsim.agents.diffusiondrive.timing_selector.metrics import calibrated_reports
from navsim.agents.diffusiondrive.timing_selector.model import (
    BidirectionalTimingSelector, timing_ranking_loss,
)


class TimingSelectorModule(pl.LightningModule):
    def __init__(self, identity, settings, loss_settings, records, lr=1e-4):
        super().__init__()
        self.model = BidirectionalTimingSelector(**settings)
        self.identity = identity
        self.loss_settings = loss_settings
        self.records = records
        self.lr = lr
        self.policy = {"enabled": False, "threshold": None}
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss, stats = timing_ranking_loss(logits, batch["labels"], batch["scores"],
                                          batch["direction"], **self.loss_settings)
        self.log("train/rank_loss", loss, on_step=False, on_epoch=True,
                 batch_size=len(batch["index"]))
        self.log("train/pairs", stats["pairs"].float(), on_step=False, on_epoch=True,
                 batch_size=len(batch["index"]))
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_rows.append({
            "index": batch["index"].detach().cpu(),
            "logits": self.model(batch).detach().cpu(),
            "labels": batch["labels"].detach().cpu(),
            "scores": batch["scores"].detach().cpu(),
            "direction": batch["direction"].detach().cpu(),
        })

    def on_validation_epoch_end(self):
        merged = {key: torch.cat([row[key] for row in self.validation_rows]).numpy()
                  for key in self.validation_rows[0]}
        self.validation_rows.clear()
        order = np.argsort(merged["index"])
        if not np.array_equal(merged["index"][order], np.arange(len(self.records))):
            raise ValueError("Validation coverage incomplete")
        merged = {key: value[order] for key, value in merged.items()}
        report = calibrated_reports(merged, self.records)
        self.policy = report["policy"]
        for split in ("calibration", "audit"):
            for metric in ("pdm", "gain_points"):
                self.log(f"{split}_{metric}", report["reports"][split][metric],
                         prog_bar=True, sync_dist=False)
        if self.global_rank == 0:
            root = Path(self.trainer.default_root_dir)
            (root / f"epoch_{self.current_epoch:02d}_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            torch.save(merged, root / f"epoch_{self.current_epoch:02d}_predictions.pt")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["timing_selector_metadata"] = {
            "identity": self.identity, "settings": self.model.settings,
            "loss_settings": self.loss_settings, "policy": self.policy, "lr": self.lr,
        }

    def on_load_checkpoint(self, checkpoint):
        metadata = checkpoint["timing_selector_metadata"]
        expected = {"identity": self.identity, "settings": self.model.settings,
                    "loss_settings": self.loss_settings, "lr": self.lr}
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError("Timing selector resume source/settings mismatch")
        self.policy = metadata["policy"]
