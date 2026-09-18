"""Balanced gate probe, joint action learning, log-separated policy calibration."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from .model import GatedRefiner, decide, loss_terms


def gate_metrics(logits, truth):
    prob = logits.sigmoid().double()
    truth = truth.bool()
    order = torch.argsort(prob, descending=True, stable=True)
    y = truth[order].double()
    tp = y.cumsum(0)
    # Threshold-grouped average precision: tied scores cannot appear perfect.
    ends = torch.cat([prob[order][:-1] != prob[order][1:], torch.tensor([True])])
    end = torch.where(ends)[0]
    precision = tp[end] / (end+1)
    recall = tp[end] / max(int(truth.sum()), 1)
    ap = (precision * torch.diff(recall, prepend=torch.zeros(1, dtype=recall.dtype))).sum()
    result = dict(auprc=float(ap), prevalence=float(truth.double().mean()), positives=int(truth.sum()))
    result['gate_score_quantiles'] = torch.quantile(prob, torch.tensor([0., .5, .9, .99, 1.], dtype=prob.dtype)).tolist()
    for fraction in (.05, .1):
        k = max(1, int(np.ceil(len(y)*fraction)))
        # Include all ties at the budget cutoff; report actual triggering fraction.
        opened = prob >= prob[order[k-1]]
        hits = int((truth & opened).sum())
        result[f'recall_at_{int(fraction*100)}pct'] = hits/max(int(truth.sum()), 1)
        result[f'precision_at_{int(fraction*100)}pct'] = hits/max(int(opened.sum()), 1)
        result[f'actual_fraction_at_{int(fraction*100)}pct'] = float(opened.float().mean())
    return result


def outcomes(modes, labels, scores, direction):
    rows = torch.arange(len(modes))
    chosen = labels[rows, modes]
    delta = scores[rows, modes].double()-scores[:, 0].double()
    safe = bool((chosen[:, [0, 1, 3, 4]].double().mean(0)+1e-7 >=
                 labels[:, 0, [0, 1, 3, 4]].double().mean(0)).all())
    safe &= bool(direction[rows, modes].double().mean()+1e-7 >= direction[:, 0].double().mean())
    return dict(pdm=float(scores[rows, modes].double().mean()), gain_points=float(delta.mean()*100),
                edits=int((modes != 0).sum()), beneficial=int((delta > 1e-6).sum()),
                harmful=int((delta < -1e-6).sum()), safety_pass=safe,
                ttc_rescued=int(((labels[:, 0, 3] < 1) & (chosen[:, 3] == 1)).sum()),
                new_ttc=int(((labels[:, 0, 3] == 1) & (chosen[:, 3] < 1)).sum()))


def calibrate(output, labels, scores, direction):
    disabled = dict(enabled=False, gate_threshold=1., risk_limit=0.)
    baseline = outcomes(torch.zeros(len(scores), dtype=torch.long), labels, scores, direction)
    best = (baseline['pdm'], 0)
    policy, grid = disabled, []
    probabilities = output['gate_logits'].sigmoid()
    thresholds = sorted(set([0., .05, .1, .25, .5, .75, .9, .99, 1.] +
                            torch.quantile(probabilities, torch.tensor([.5, .75, .9, .95, .98, .99])).tolist()))
    for threshold in thresholds:
        for risk in (.05, .1, .2, .3, .5, .7, 1.):
            modes = decide(output, threshold, risk)
            result = outcomes(modes, labels, scores, direction)
            grid.append(dict(gate_threshold=threshold, risk_limit=risk, **result))
            key = (result['pdm'], -result['edits'])
            if result['safety_pass'] and result['pdm'] > baseline['pdm']+1e-6 and key > best:
                best = key
                policy = dict(enabled=True, gate_threshold=threshold, risk_limit=risk)
    return policy, grid


class GatedModule(pl.LightningModule):
    def __init__(self, identity, stage='probe', lr=1e-4, smoke=False):
        super().__init__()
        self.model = GatedRefiner()
        self.identity, self.stage, self.lr, self.smoke = identity, stage, lr, smoke
        # Frozen generator/PCS/TRV supply detached cached features. The small
        # contextual probe itself is trainable. Its action head is unused.
        if stage == 'probe':
            self.model.head.requires_grad_(False)
        self.policy = dict(enabled=False, gate_threshold=1., risk_limit=0.)
        self.validation_rows = []
        self.negative_scores = {}
        self.mined_negatives = []

    def on_train_epoch_start(self):
        self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
        self.trainer.train_dataloader.batch_sampler.set_hard_negatives(self.mined_negatives)

    def training_step(self, batch, batch_idx):
        output = self.model(batch['context'], batch['variants'])
        negative = ~batch['need'].bool()
        self.negative_scores.update(zip(batch['index'][negative].detach().cpu().tolist(),
                                       output['gate_logits'][negative].detach().cpu().tolist()))
        loss, terms = loss_terms(output, batch, self.stage)
        for key, value in dict(loss=loss, **terms).items():
            self.log('train/'+key, value, on_step=False, on_epoch=True, sync_dist=True,
                     batch_size=len(batch['index']))
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch+1) % 5:
            return
        gathered = [self.negative_scores]
        if torch.distributed.is_initialized():
            gathered = [None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, self.negative_scores)
        merged = {}
        for values in gathered:
            for index, score in values.items():
                merged[index] = max(score, merged.get(index, -float('inf')))
        self.mined_negatives = sorted(merged, key=lambda i: (-merged[i], i))[:1024]
        self.negative_scores.clear()

    def validation_step(self, batch, batch_idx):
        output = self.model(batch['context'], batch['variants'])
        self.validation_rows.append({**{k: batch[k].detach().cpu() for k in
                                       ('index', 'labels', 'scores', 'direction', 'need', 'calibration')},
                                     **{k: v.detach().cpu() for k, v in output.items()}})

    def on_validation_epoch_end(self):
        local = {k: torch.cat([r[k] for r in self.validation_rows]) for k in self.validation_rows[0]}
        self.validation_rows.clear()
        gathered = [local]
        if torch.distributed.is_initialized():
            gathered = [None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        merged = {k: torch.cat([r[k] for r in gathered]) for k in local}
        _, unique = np.unique(merged['index'].numpy(), return_index=True)
        merged = {k: v[unique] for k, v in merged.items()}
        if not all(torch.isfinite(v).all() for v in merged.values()):
            raise ValueError('Nonfinite validation predictions')
        mask = merged['calibration'].bool()
        if not mask.any() or mask.all():
            if not self.smoke:
                raise ValueError('Missing validation partition')
            mask = torch.arange(len(mask)) % 2 == 0
        reports = {}
        grid = []
        if self.stage == 'joint':
            self.policy, grid = calibrate(
                {k: merged[k][mask] for k in ('gate_logits', 'gain', 'unsafe_logits')},
                *[merged[k][mask] for k in ('labels', 'scores', 'direction')])
        for name, selected in [('calibration', mask), ('audit', ~mask)]:
            out = {k: merged[k][selected] for k in ('gate_logits', 'gain', 'unsafe_logits')}
            modes = decide(out, self.policy['gate_threshold'], self.policy['risk_limit']) if self.policy['enabled'] else torch.zeros(int(selected.sum()), dtype=torch.long)
            reports[name] = dict(gate=gate_metrics(out['gate_logits'], merged['need'][selected]),
                                 policy=outcomes(modes, *[merged[k][selected] for k in ('labels', 'scores', 'direction')]))
        for name in reports:
            self.log(name+'_ap', torch.tensor(reports[name]['gate']['auprc'], device=self.device),
                     sync_dist=True, prog_bar=True)
            self.log(name+'_pdm', torch.tensor(reports[name]['policy']['pdm'], device=self.device),
                     sync_dist=True, prog_bar=True)
        if self.global_rank == 0:
            root = Path(self.trainer.default_root_dir)
            name = f'epoch_{self.current_epoch:02d}'
            (root/f'{name}_report.json').write_text(json.dumps(dict(policy=self.policy, reports=reports), indent=2))
            torch.save(merged, root/f'{name}_predictions.pt')
            if grid:
                with (root/f'{name}_thresholds.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
                    writer.writeheader()
                    writer.writerows(grid)

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, weight_decay=.01)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['gate_metadata'] = dict(identity=self.identity, stage=self.stage, lr=self.lr, policy=self.policy)
        checkpoint['gate_mining'] = dict(negative_scores=self.negative_scores, mined_negatives=self.mined_negatives)

    def on_load_checkpoint(self, checkpoint):
        m = checkpoint['gate_metadata']
        if m['identity'] != self.identity or m['stage'] != self.stage or m['lr'] != self.lr:
            raise ValueError('Resume stage/cache/settings mismatch')
        self.policy = m['policy']
        mining = checkpoint.get('gate_mining', {})
        self.negative_scores = mining.get('negative_scores', {})
        self.mined_negatives = mining.get('mined_negatives', [])
