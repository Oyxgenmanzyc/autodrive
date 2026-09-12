"""One-stage contextual edit training with validation-only identity calibration."""
import numpy as np
import torch
import pytorch_lightning as pl
from .model import PostSelectionRefiner, decide, refinement_loss


def calibrate(gain, risk, labels, scores, direction):
    """Grid fixed before testing. Identity is an explicit available policy.

    All computations use a full, deduplicated navtrain validation split.
    Empirical aggregate safety constraints are not safety guarantees.
    """
    n = len(scores)
    rows = torch.arange(n)
    baseline = scores[:, 0].mean().item()
    best = (baseline, 0)
    policy = {'enabled': False, 'margin': 0., 'risk_limit': 0.}
    chosen = torch.zeros(n, dtype=torch.long)
    for margin in (0., .005, .01, .02, .05, .1):
        for risk_limit in (.05, .1, .2, .3, .5):
            modes = decide({'gain': gain, 'unsafe_logits': risk}, margin, risk_limit)
            selected_labels = labels[rows, modes]
            if (selected_labels[:, [0, 1, 3, 4]].mean(0)+1e-7 < labels[:, 0, [0, 1, 3, 4]].mean(0)).any():
                continue
            if direction[rows, modes].mean()+1e-7 < direction[:, 0].mean():
                continue
            key = (scores[rows, modes].mean().item(), -int((modes != 0).sum()))
            if key[0] > baseline+1e-6 and key > best:
                best, chosen = key, modes
                policy = {'enabled': True, 'margin': margin, 'risk_limit': risk_limit}
    return policy, chosen


class RefinementModule(pl.LightningModule):
    def __init__(self, identity, lr=1e-4):
        super().__init__()
        self.model = PostSelectionRefiner()
        self.identity, self.lr = identity, lr
        self.policy = {'enabled': False, 'margin': 0., 'risk_limit': 0.}
        self.validation_rows = []

    def training_step(self, batch, batch_idx):
        output = self.model(batch['context'], batch['variants'])
        loss = refinement_loss(output, batch['labels'], batch['scores'], batch['direction'], batch['teacher'])
        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(batch['index']))
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch['context'], batch['variants'])
        self.validation_rows.append({
            **{k: batch[k].detach().cpu() for k in ('index', 'labels', 'scores', 'direction')},
            'gain': output['gain'].detach().cpu(), 'risk': output['unsafe_logits'].detach().cpu()})

    def on_validation_epoch_end(self):
        local = {k: torch.cat([r[k] for r in self.validation_rows]) for k in self.validation_rows[0]}
        self.validation_rows.clear()
        all_rows = [local]
        if torch.distributed.is_initialized():
            all_rows = [None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(all_rows, local)
        merged = {k: torch.cat([r[k] for r in all_rows]) for k in local}
        _, unique = np.unique(merged['index'].numpy(), return_index=True)
        merged = {k: v[unique] for k, v in merged.items()}
        if not all(torch.isfinite(v).all() for v in merged.values()):
            raise ValueError('Nonfinite validation result')
        self.policy, modes = calibrate(**{k: merged[k] for k in ('gain', 'risk', 'labels', 'scores', 'direction')})
        scores = merged['scores']
        pdm = scores[torch.arange(len(modes)), modes].mean()
        # Every rank computes exactly the same deduplicated global result.
        self.log('val_pdm', pdm.to(self.device), sync_dist=True, prog_bar=True)
        self.log('val_gain', (pdm-scores[:, 0].mean()).to(self.device), sync_dist=True, prog_bar=True)
        if self.global_rank == 0:
            from pathlib import Path
            from navsim.agents.diffusiondrive.pcs.common import write_new_json
            write_new_json(Path(self.trainer.default_root_dir)/f'epoch_{self.current_epoch:02d}_policy.json',
                           {'policy': self.policy, 'pdm': float(pdm), 'base_pdm': float(scores[:, 0].mean()),
                            'scene_count': len(modes), 'edits': int((modes != 0).sum())})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=.01)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['refinement_metadata'] = {'identity': self.identity, 'policy': self.policy, 'lr': self.lr}

    def on_load_checkpoint(self, checkpoint):
        meta = checkpoint['refinement_metadata']
        if meta['identity'] != self.identity or meta['lr'] != self.lr:
            raise ValueError('Resume cache/settings differ')
        self.policy = meta['policy']
