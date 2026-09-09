"""Train TRV only on the decisions actually made by the frozen PCS proposer."""
import torch
import pytorch_lightning as pl

from .common import load_scorer
from .veto import TripleRiskVeto, calibrate_thresholds, relative_risk_targets


class TripleRiskVetoModule(pl.LightningModule):
    def __init__(self, metadata, pcs_checkpoint, lr=3e-4, epochs=10):
        super().__init__()
        proposer, _ = load_scorer(
            pcs_checkpoint, "cpu", metadata["provenance"],
        )
        self.model = TripleRiskVeto(
            proposer, thresholds=metadata["thresholds"], hidden_dim=metadata["hidden_dim"],
        )
        self.metadata = metadata
        self.lr = lr
        self.epochs = epochs
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        output = self.model(
            batch["context"], batch["pcs_mode"], batch["base_mode"], verify_modes=True,
        )
        losses = self.model.loss(output, batch["labels"], batch["scores"])
        self.log(
            "train/pcs_mode_mismatch_rate",
            output["pcs_mode_mismatch"].float().mean(),
            on_step=False, on_epoch=True, batch_size=len(batch["scores"]),
            sync_dist=True,
        )
        for name, value in losses.items():
            self.log(
                f"train/{name}", value, on_step=name == "loss", on_epoch=True,
                batch_size=len(batch["scores"]), sync_dist=True,
            )
        return losses["loss"]

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        output = self.model(
            batch["context"], batch["pcs_mode"], batch["base_mode"], verify_modes=True,
        )
        rows = torch.arange(len(batch["pcs_mode"]), device=self.device)
        pcs_scores = batch["scores"][rows, batch["pcs_mode"]]
        base_scores = batch["scores"][rows, batch["base_mode"]]
        targets, _, _ = relative_risk_targets(
            batch["labels"], batch["pcs_mode"], batch["base_mode"],
        )
        packed = torch.cat([
            batch["index"].double().unsqueeze(-1),
            pcs_scores.double().unsqueeze(-1),
            base_scores.double().unsqueeze(-1),
            output["pcs_mode"].ne(output["base_mode"]).double().unsqueeze(-1),
            output["risks"].double(),
            targets.double(),
        ], dim=-1)
        self.validation_rows.extend(packed.cpu().tolist())
        self.log(
            "val/pcs_mode_mismatch_rate",
            output["pcs_mode_mismatch"].float().mean(),
            on_step=False, on_epoch=True, batch_size=len(batch["scores"]),
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        rows = self.validation_rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for rank_rows in gathered for row in rank_rows]
        unique = {int(row[0]): row[1:] for row in rows}
        values = torch.tensor(list(unique.values()), dtype=torch.float64)
        if not len(values):
            raise ValueError("Empty validation set")
        pcs_scores, base_scores = values[:, 0], values[:, 1]
        changed = values[:, 2].bool()
        risks, targets = values[:, 3:6], values[:, 6:9]
        thresholds, result = calibrate_thresholds(
            risks, pcs_scores, base_scores, changed,
        )
        self.model.thresholds = thresholds
        self.metadata["thresholds"] = thresholds
        metrics = {
            "val/pdm": result["pdm"],
            "val/pcs_pdm": result["pcs_pdm"],
            "val/base_pdm": result["base_pdm"],
            "val/gain_vs_pcs": result["pdm"] - result["pcs_pdm"],
            "val/gain_vs_base": result["pdm"] - result["base_pdm"],
            "val/accepted_switch_rate": (
                result["accepted_switch_count"] / max(result["changed_count"], 1)
            ),
            "val/vetoed_count": float(result["vetoed_count"]),
            "val/new_zero_count": float(result["new_zero_count"]),
            "val/rescued_count": float(result["rescued_count"]),
            "val/nc_risk_rate": targets[:, 0].gt(0).double().mean().item(),
            "val/dac_risk_rate": targets[:, 1].gt(0).double().mean().item(),
            "val/ttc_risk_rate": targets[:, 2].gt(0).double().mean().item(),
            "val/scene_count": float(len(values)),
        }
        for index, name in enumerate(("nc", "dac", "ttc")):
            metrics[f"val/{name}_threshold"] = thresholds[index]
        for name, value in metrics.items():
            self.log(
                name, value, sync_dist=False,
                prog_bar=name in ("val/pdm", "val/gain_vs_pcs"),
            )
        self.validation_rows = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.risk_heads.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["veto_metadata"] = self.metadata

    def on_load_checkpoint(self, checkpoint):
        metadata = checkpoint["veto_metadata"]
        if (
            metadata["provenance"] != self.metadata["provenance"]
            or metadata["pcs_scorer_sha256"] != self.metadata["pcs_scorer_sha256"]
        ):
            raise ValueError("Resume checkpoint does not match PCS/candidate provenance")
        self.metadata = metadata
        self.model.thresholds = list(metadata["thresholds"])
