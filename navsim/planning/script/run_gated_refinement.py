"""3.1.05_5: independent Gate, paired training and unchanged 25-action scoring."""
import argparse
import json
import os
from pathlib import Path
import torch
from navsim.agents.diffusiondrive.pcs.common import load_torch, write_new_json
from navsim.agents.diffusiondrive.gated_refinement.data import prepare, GatedDataset, code_identity
from navsim.agents.diffusiondrive.gated_refinement.labels import PairedBatchSampler
from navsim.planning.script.run_pcs import loader, new_run


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader, DistributedSampler
    from navsim.agents.diffusiondrive.gated_refinement.training import GatedModule
    pl.seed_everything(0, workers=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    sets = {s: GatedDataset(args.candidate_cache, args.edit_cache, args.gate_cache, s) for s in ('train', 'val')}
    if sets['train'].gate_manifest != sets['val'].gate_manifest:
        raise ValueError('Train/val cache identities differ')
    identity = sets['train'].gate_manifest
    model = GatedModule(identity, args.stage, args.lr, args.smoke)
    if args.init:
        initial = load_torch(args.init)
        meta = initial['gate_metadata']
        if args.stage != 'joint' or meta['stage'] != 'probe' or meta['identity'] != identity:
            raise ValueError('Joint initialization requires a matching Gate probe checkpoint')
        model.load_state_dict(initial['state_dict'], strict=True)
    if args.stage == 'joint' and not (args.init or args.resume or args.smoke):
        raise ValueError('Run probe first, then pass --init; use --resume only for same-stage continuation')
    if args.init and args.resume:
        raise ValueError('Choose init or resume, not both')

    class Data(pl.LightningDataModule):
        def train_dataloader(self):
            sampler = PairedBatchSampler(sets['train'].targets['need'], sets['train'].targets['pairs'],
                                          args.batch_size, self.trainer.global_rank, self.trainer.world_size)
            return DataLoader(sets['train'], batch_sampler=sampler, num_workers=args.workers)

        def val_dataloader(self):
            sampler = DistributedSampler(sets['val'], num_replicas=self.trainer.world_size,
                                         rank=self.trainer.global_rank, shuffle=False, drop_last=False)
            return loader(sets['val'], args.workers, batch_size=args.batch_size, sampler=sampler)

    if 'GATED_RUN_DIR' not in os.environ:
        os.environ['GATED_RUN_DIR'] = str(new_run(args.output))
    run = Path(os.environ['GATED_RUN_DIR'])
    monitor = 'calibration_ap' if args.stage == 'probe' else 'calibration_pdm'
    checkpoint = ModelCheckpoint(dirpath=run/'checkpoints', monitor=monitor, mode='max',
                                  save_top_k=1, save_last=True, filename='epoch={epoch:02d}',
                                  auto_insert_metric_name=False)
    trainer = pl.Trainer(default_root_dir=str(run), accelerator='gpu', devices=args.devices,
                         strategy='ddp' if args.devices > 1 else 'auto', precision='32-true',
                         use_distributed_sampler=False, max_epochs=args.epochs, callbacks=[checkpoint],
                         logger=CSVLogger(str(run), name='csv'), num_sanity_val_steps=0,
                         limit_train_batches=2 if args.smoke else 1.,
                         limit_val_batches=2 if args.smoke else 1., gradient_clip_val=1.)
    if trainer.global_rank == 0:
        write_new_json(run/'run.json', dict(arguments=vars(args), identity=identity,
                       train_scenes=len(sets['train']), train_positives=int(sets['train'].targets['need'].sum()),
                       val_scenes=len(sets['val']), sampling='1 positive + 2 matched negative + 1 random negative'))
    trainer.fit(model, datamodule=Data(), ckpt_path=args.resume)
    if trainer.global_rank == 0:
        write_new_json(run/'result.json', dict(best_checkpoint=checkpoint.best_model_path,
                       last_checkpoint=checkpoint.last_model_path, monitor=monitor,
                       best_score=float(checkpoint.best_model_score)))
        print('Best checkpoint:', checkpoint.best_model_path)
        print('Last checkpoint:', checkpoint.last_model_path)
        print('Read epoch_*_report.json for calibration AND independent audit outcomes.')
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--edit-cache', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--min-gain', type=float, default=.02)
    tr = sub.add_parser('train')
    for name in ('candidate-cache', 'edit-cache', 'gate-cache', 'output'):
        tr.add_argument('--'+name, required=True)
    tr.add_argument('--stage', choices=('probe', 'joint'), default='probe')
    tr.add_argument('--devices', type=int, default=4)
    tr.add_argument('--batch-size', type=int, default=32)
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--workers', type=int, default=0)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--smoke', action='store_true')
    tr.add_argument('--init')
    tr.add_argument('--resume')
    ev = sub.add_parser('evaluate')
    for name in ('baseline', 'backbone', 'anchor', 'pcs-scorer', 'veto', 'refiner', 'metric-cache', 'data-root', 'output'):
        ev.add_argument('--'+name, required=True)
    ev.add_argument('--seed', type=int, default=0)
    ev.add_argument('--max-scenes', type=int, default=0)
    ev.add_argument('--workers', type=int, default=0)
    ev.add_argument('--score-workers', type=int, default=2)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.edit_cache, args.output, args.min_gain)
    elif args.command == 'train':
        train(args)
    else:
        evaluate(args)


# Evaluation is defined below so training imports no extra online machinery.

def evaluate(args):
    import csv
    import numpy as np
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    from navsim.agents.diffusiondrive.pcs.common import provenance, sha256, generate_context
    from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
    from navsim.agents.diffusiondrive.post_selection.geometry import brake_bank, safe_oracle
    from navsim.agents.diffusiondrive.gated_refinement.model import GatedRefiner, decide
    from navsim.planning.script.run_post_selection import checked_veto
    from navsim.planning.script.run_pcs import prepare_source
    args.split, args.feature_cache = 'navtest', None
    generator, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    veto = checked_veto(args, identity)
    ckpt = load_torch(args.refiner)
    meta = ckpt['gate_metadata']
    if meta['stage'] != 'joint' or meta['identity']['sources'] != code_identity():
        raise ValueError('Require a matching joint checkpoint, not a Gate-only probe')
    identity_meta = meta['identity']['edit_manifest']
    if (identity_meta['provenance'] != identity
            or identity_meta['veto_sha256'] != sha256(args.veto)):
        raise ValueError('Refiner baseline/selector/source identity mismatch')
    model = GatedRefiner().eval().to(device)
    state = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state, strict=True)
    policy = meta['policy']
    run = new_run(args.output)
    write_new_json(run/'run.json', {'arguments': vars(args), 'metadata': meta,
                                   'refiner_sha256': sha256(args.refiner)})
    rows, pending = [], []
    with (run/'paired_results.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = None

        def finish(job):
            nonlocal writer
            future, token, mode, output_gain, output_risk, gate_probability = job
            labels, scores, direction = future.result()
            oracle = int(safe_oracle(labels, scores, direction))
            row = {'token': token, 'action': mode, 'predicted_gain': output_gain,
                   'predicted_unsafe': output_risk, 'gate_probability': gate_probability, 'original_pdm': float(scores[0]),
                   'final_pdm': float(scores[mode]), 'gain': float(scores[mode]-scores[0]),
                   'controlled_oracle_pdm': float(scores[oracle])}
            for prefix, k in (('original', 0), ('final', mode)):
                for j, name in enumerate(('nc', 'dac', 'progress', 'ttc', 'comfort')):
                    row[f'{prefix}_{name}'] = float(labels[k, j])
                row[f'{prefix}_direction'] = float(direction[k])
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)

        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context('spawn')) as pool:
            for i, (record, features) in enumerate(tqdm(loader(source, args.workers, batch_size=None), desc='PTR navtest')):
                context = generate_context(generator, features, record['token'], args.seed, device)
                gpu = {k: v.unsqueeze(0).to(device) for k, v in context.items()}
                with torch.no_grad():
                    selected = int(veto(gpu)['final_mode'][0])
                    variants = brake_bank(context['proposals'][selected].numpy())
                    output = model(gpu, torch.from_numpy(variants).unsqueeze(0).to(device))
                    if not all(torch.isfinite(v).all() for v in output.values()):
                        raise ValueError('Nonfinite refiner output')
                    mode = int(decide(output, policy['gate_threshold'], policy['risk_limit'])[0]) if policy['enabled'] else 0
                # Decision is finished BEFORE official labels are computed.
                future = pool.submit(score_candidates, paths[record['token']], variants, i == 0)
                pending.append((future, record['token'], mode, float(output['gain'][0, mode]),
                                float(output['unsafe_logits'][0, mode].sigmoid()),
                                float(output['gate_logits'][0].sigmoid())))
                if len(pending) >= args.score_workers*2:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
    if len(rows) != len(source) or not rows:
        raise ValueError('Incomplete evaluation')
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('token', 'action')}
    summary.update(scenes=len(rows), completed=True, policy=policy,
                   edits=sum(r['action'] != 0 for r in rows),
                   beneficial_edits=sum(r['gain'] > 1e-6 for r in rows),
                   harmful_edits=sum(r['gain'] < -1e-6 for r in rows))
    write_new_json(run/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print(f'Paired evaluation: {run}')


if __name__ == '__main__':
    main()
