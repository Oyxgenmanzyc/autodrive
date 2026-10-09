"""One loss, one optimizer; frozen generator, PCS and Ranker remain external."""
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from .metrics import reports
from .model import TimingModeValue, value_loss


class TimingModeModule(pl.LightningModule):
    def __init__(self, identity, lr=1e-4, smoke=False):
        super().__init__()
        self.model = TimingModeValue()
        self.identity, self.lr, self.smoke = identity, lr, smoke
        self.policy = dict(enabled=False, threshold=.4)
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        predicted = self.model(batch)
        loss, stats = value_loss(predicted, batch['labels'], batch['scores'], batch['direction'])
        self.log('train/value_loss', loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=len(batch['index']))
        self.log('train/pairs', stats['pairs'].float(), on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=len(batch['index']))
        return loss

    def validation_step(self, batch, batch_idx):
        predicted = self.model(batch)
        self.validation_rows.append({key: value.detach().cpu() for key, value in
                                     dict(index=batch['index'], calibration=batch['calibration'],
                                          labels=batch['labels'], scores=batch['scores'],
                                          direction=batch['direction'], predicted=predicted).items()})

    def on_validation_epoch_end(self):
        if not self.validation_rows:
            raise ValueError('Empty timing validation loader')
        local = {key: torch.cat([r[key] for r in self.validation_rows]) for key in self.validation_rows[0]}
        self.validation_rows.clear()
        gathered = [local]
        if torch.distributed.is_initialized():
            gathered = [None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        merged = {key: torch.cat([r[key] for r in gathered]) for key in local}
        indices, first = np.unique(merged['index'].numpy(), return_index=True)
        expected = len(self.trainer.val_dataloaders.dataset)
        if not np.array_equal(indices, np.arange(expected)):
            raise ValueError('Timing validation coverage incomplete')
        merged = {key: value[first] for key, value in merged.items()}
        if not all(torch.isfinite(value).all() for value in merged.values()):
            raise ValueError('Nonfinite timing validation values')
        report = reports(merged)
        self.policy = report['policy']
        for split in ('calibration', 'audit'):
            for metric in ('pdm', 'gain_points'):
                self.log(split+'_'+metric,
                         torch.tensor(report['reports'][split]['policy'][metric],
                                      device=self.device, dtype=torch.float64),
                         sync_dist=False, prog_bar=True)
        if self.global_rank == 0:
            root = Path(self.trainer.default_root_dir)
            (root/f'epoch_{self.current_epoch:02d}_report.json').write_text(
                json.dumps(report, indent=2), encoding='utf-8')
            torch.save(merged, root/f'epoch_{self.current_epoch:02d}_predictions.pt')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['timing_modes_metadata'] = dict(identity=self.identity,
                                                   settings=self.model.settings,
                                                   lr=self.lr, smoke=self.smoke,
                                                   policy=self.policy)

    def on_load_checkpoint(self, checkpoint):
        metadata = checkpoint['timing_modes_metadata']
        if (metadata['identity'] != self.identity or metadata['settings'] != self.model.settings or
                metadata['lr'] != self.lr or metadata['smoke'] != self.smoke):
            raise ValueError('Timing-mode resume source/settings mismatch')
        self.policy = metadata['policy']
