"""Separate OOF teacher and residual-ranker training, never a mixed objective."""
import json
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from .model import CostRanker, ranking_loss
from .metrics import reports


class FoldTeacher(pl.LightningModule):
    def __init__(self, identity, settings, lr):
        super().__init__()
        from navsim.agents.diffusiondrive.pcs.model import PDMCSHead
        # Intentionally random initialization; never initialize with full-data PCS.
        self.head = PDMCSHead(**settings)
        self.identity, self.lr = identity, lr

    def training_step(self, batch, batch_idx):
        loss = self.head.loss(self.head(batch['context']), batch['labels'])
        self.log('teacher_bce', loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=len(batch['index']))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['fold_metadata'] = self.identity

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('fold_metadata') != self.identity:
            raise ValueError('Fold resume identity mismatch')


class RankModule(pl.LightningModule):
    def __init__(self, identity, settings, loss_settings, lr=1e-4, smoke=False):
        super().__init__()
        # Frozen generator / PCS are external; optimizer cannot reach them.
        self.model = CostRanker(**settings)
        self.identity, self.loss_settings, self.lr = identity, loss_settings, lr
        self.smoke = smoke
        self.policy = dict(enabled=False, alpha=0.)
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        residual = self.model(batch)
        teacher = batch['pcs_scores'] if self.smoke else batch['oof_scores']
        loss, stats = ranking_loss(batch['pcs_scores'], residual, batch['scores'],
                                   teacher, **self.loss_settings)
        self.log('train/rank_loss', loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=len(batch['index']))
        self.log('train/hard_pairs', stats['hard_pairs'].float(), on_step=False,
                 on_epoch=True, sync_dist=True, batch_size=len(batch['index']))
        return loss

    def validation_step(self, batch, batch_idx):
        residual = self.model(batch)
        self.validation_rows.append({**{k: batch[k].detach().cpu() for k in
                                     ('index', 'calibration', 'pcs_scores', 'labels', 'scores', 'direction')},
                                     'residual': residual.detach().cpu()})

    def on_validation_epoch_end(self):
        if not self.validation_rows:
            raise ValueError('Empty validation loader')
        local = {k: torch.cat([r[k] for r in self.validation_rows]) for k in self.validation_rows[0]}
        self.validation_rows.clear()
        gathered = [local]
        if torch.distributed.is_initialized():
            gathered = [None]*torch.distributed.get_world_size()
            # One scheduled collective, entered by EVERY rank every epoch.
            torch.distributed.all_gather_object(gathered, local)
        merged = {k: torch.cat([r[k] for r in gathered]) for k in local}
        indices, first = np.unique(merged['index'].numpy(), return_index=True)
        expected = len(self.trainer.val_dataloaders.dataset)
        if not np.array_equal(indices, np.arange(expected)):
            raise ValueError('Validation coverage incomplete')
        merged = {k: v[first] for k, v in merged.items()}
        if not all(torch.isfinite(v).all() for v in merged.values()):
            raise ValueError('Nonfinite validation data')
        report = reports(merged)
        self.policy = report['policy']
        for split in ('calibration', 'audit'):
            for metric in ('pdm', 'gain_points'):
                self.log(split+'_'+metric,
                         torch.tensor(report['reports'][split]['policy'][metric], device=self.device,
                                      dtype=torch.float64), sync_dist=False, prog_bar=True)
        if self.global_rank == 0:
            root = Path(self.trainer.default_root_dir)
            (root/f'epoch_{self.current_epoch:02d}_report.json').write_text(
                json.dumps(report, indent=2), encoding='utf-8')
            torch.save(merged, root/f'epoch_{self.current_epoch:02d}_predictions.pt')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['rank_metadata'] = dict(identity=self.identity, settings=self.model.settings,
                                          loss_settings=self.loss_settings, policy=self.policy,
                                          lr=self.lr, smoke=self.smoke)

    def on_load_checkpoint(self, checkpoint):
        m = checkpoint['rank_metadata']
        if (m['identity'] != self.identity or m['settings'] != self.model.settings or
                m['loss_settings'] != self.loss_settings or m['lr'] != self.lr or m['smoke'] != self.smoke):
            raise ValueError('Ranker resume source/settings mismatch; do not reuse teacher/gate checkpoints')
        self.policy = m['policy']
