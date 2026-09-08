"""Train the conservative advantage scorer on fixed K67 plus GTRS candidates."""
import torch
import pytorch_lightning as pl

from .conservative import (
    ConservativeAdvantageScorer, scorer_output, select_candidates,
)


class ConservativeAdvantageModule(pl.LightningModule):
    def __init__(self, metadata, lr=3e-4, epochs=20):
        super().__init__()
        self.head = ConservativeAdvantageScorer(**metadata["settings"])
        self.metadata = metadata
        self.lr = lr
        self.epochs = epochs
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        output = scorer_output(self.head, batch["context"])
        losses = self.head.losses(
            output, batch["context"], batch["labels"], batch["scores"],
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
        output = scorer_output(self.head, batch["context"])
        selection = select_candidates(batch["context"], output)
        chosen, raw, base = (
            selection["selected_mode"], selection["raw_selected_mode"], selection["base_mode"],
        )
        rows = torch.arange(len(chosen), device=chosen.device)
        scores = batch["scores"]
        labels = batch["labels"]
        selected_labels = labels[rows, chosen]
        base_labels = labels[rows, base]
        dominated = (
            (selected_labels <= base_labels).all(-1)
            & (selected_labels < base_labels).any(-1)
        )
        packed = torch.stack([
            batch["index"].double(),
            scores[rows, chosen].double(),
            scores[rows, base].double(),
            scores.max(-1).values.double(),
            (chosen != base).double(),
            (raw != base).double(),
            dominated.double(),
            selection["predicted_delta"].double(),
            selection["win_probability"].double(),
            selection["catastrophic_risk"].double(),
        ], dim=-1)
        self.validation_rows.extend(packed.cpu().tolist())

    def on_validation_epoch_end(self):
        rows = self.validation_rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for rank_rows in gathered for row in rank_rows]
        unique = {int(row[0]): row[1:] for row in rows}
        values = torch.tensor(list(unique.values()), dtype=torch.float64, device=self.device)
        if not len(values):
            raise ValueError("Empty validation set")
        selected, base, oracle, changed, proposed, dominated, delta, win, risk = values.unbind(-1)
        metrics = {
            "val/pdm": selected.mean(),
            "val/base_pdm": base.mean(),
            "val/gain": (selected - base).mean(),
            "val/oracle": oracle.mean(),
            "val/change_rate": changed.mean(),
            "val/proposed_change_rate": proposed.mean(),
            "val/dominated_count": dominated.sum(),
            "val/rescued_count": ((base == 0) & (selected > 0)).sum(),
            "val/new_zero_count": ((base > 0) & (selected == 0)).sum(),
            "val/decision_delta": delta.mean(),
            "val/decision_win_probability": win.mean(),
            "val/decision_catastrophic_risk": risk.mean(),
            "val/scene_count": len(values),
        }
        for name, value in metrics.items():
            self.log(name, value, sync_dist=False, prog_bar=name in ("val/pdm", "val/gain"))
        self.validation_rows = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["pcs_metadata"] = self.metadata

