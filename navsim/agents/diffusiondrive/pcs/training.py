"""Train only PCS on fixed inference candidates; select checkpoints on held-out logs."""
import torch
import pytorch_lightning as pl
from .model import PDMCSHead, select_candidates


class PCSModule(pl.LightningModule):
    def __init__(self, metadata, lr=3e-4, epochs=20):
        super().__init__()
        self.head = PDMCSHead(**metadata["settings"])
        self.metadata = metadata
        self.lr = lr
        self.epochs = epochs
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        output = self.head(batch["context"])
        loss = self.head.loss(output, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 batch_size=len(batch["scores"]), sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        output = self.head(batch["context"])
        selection = select_candidates(batch["context"], output["scores"])
        chosen, base = selection["selected_mode"], selection["base_mode"]
        rows = torch.arange(len(chosen), device=chosen.device)
        scores = batch["scores"]
        packed = torch.stack([
            batch["index"].double(),
            scores[rows, chosen].double(), scores[rows, base].double(),
            scores.max(-1).values.double(), (chosen != base).double(),
        ], dim=-1)
        self.validation_rows.extend(packed.cpu().tolist())

    def on_validation_epoch_end(self):
        rows = self.validation_rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [r for rank_rows in gathered for r in rank_rows]
        # Lightning's distributed sampler pads validation with duplicate examples.
        # Deduplicate by stable dataset index before computing checkpoint metrics.
        unique = {int(row[0]): row[1:] for row in rows}
        values = torch.tensor(list(unique.values()), dtype=torch.float64, device=self.device)
        if not len(values):
            raise ValueError("Empty validation set")
        selected, base, oracle, changed = values.unbind(-1)
        metrics = {
            "val/pdm": selected.mean(), "val/base_pdm": base.mean(),
            "val/gain": (selected - base).mean(), "val/oracle": oracle.mean(),
            "val/change_rate": changed.mean(),
            "val/rescued_count": ((base == 0) & (selected > 0)).sum(),
            "val/new_zero_count": ((base > 0) & (selected == 0)).sum(),
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
