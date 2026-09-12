"""PTR 3.1.05_4: controlled Oracle, compact labels, training, online evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from navsim.agents.diffusiondrive.pcs.common import (
    load_torch, sha256, write_new_json, provenance, generate_context, CONFIG_ROOT,
)
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset, metric_paths
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.agents.diffusiondrive.pcs.veto import load_veto
from navsim.agents.diffusiondrive.post_selection.geometry import ACTIONS, brake_bank, safe_oracle
from navsim.agents.diffusiondrive.post_selection.data import (
    cache_identity, read_blocks, RefinementDataset, source_identity,
)
from navsim.agents.diffusiondrive.post_selection.model import PostSelectionRefiner, decide
from navsim.planning.script.run_pcs import loader, save_entry, new_run, prepare_source


def checked_veto(args, identity):
    veto, meta = load_veto(args.veto, 'cuda:0', identity)
    if meta['pcs_scorer_sha256'] != sha256(args.pcs_scorer):
        raise ValueError('PCS checkpoint does not match frozen TRV')
    veto.requires_grad_(False)
    return veto


def prepare(args):
    from omegaconf import OmegaConf
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard')
    source = {s: CandidateDataset(args.candidate_cache, s, allow_partial=bool(args.limit)) for s in ('train', 'val')}
    official = OmegaConf.load(CONFIG_ROOT/'training/default_train_val_test_log_split.yaml')
    if set(official.train_logs) & set(official.val_logs):
        raise ValueError('Official train/val log overlap')
    for split, dataset in source.items():
        allowed = set(official.train_logs if split == 'train' else official.val_logs)
        if any(r['log_name'] not in allowed for r in dataset.records):
            raise ValueError('Record is outside official training split')
    root = Path(args.output)
    identity = cache_identity(source['train'], args.veto, args.metric_cache, args.limit)
    if source['val'].provenance != identity['provenance']:
        raise ValueError('Train/val provenance differs')
    veto = checked_veto(args, identity['provenance'])
    write_new_json(root/'manifest.json', identity)
    paths = metric_paths(args.metric_cache)
    for split, dataset in source.items():
        records = dataset.records[:args.limit] if args.limit else dataset.records
        write_new_json(root/f'{split}_records.json', records)
        if any(r['token'] not in paths for r in records):
            raise ValueError(f'Missing {split} metric cache; restore navtrain metric cache')
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context('spawn')) as pool:
            verified = False
            for start in tqdm(range(0, len(records), 128), desc=f'PTR {split} blocks'):
                if (start//128) % args.num_shards != args.shard_index:
                    continue
                block_records = records[start:start+128]
                path = root/split/f'block_{start//128:05d}.pt'
                if path.exists():
                    old = load_torch(path)
                    if old['tokens'] != [r['token'] for r in block_records]:
                        raise ValueError(f'Conflicting block: {path}')
                    continue
                entries, pending = [], []

                def finish(job):
                    future, selected, mode, expected = job
                    labels, scores, direction = future.result()
                    np.testing.assert_allclose(scores[0], expected, atol=1e-5, rtol=0,
                                               err_msg='Identity no longer matches original PDM')
                    entries.append({'selected': selected.clone(), 'mode': torch.tensor(mode),
                                    'labels': torch.from_numpy(labels), 'scores': torch.from_numpy(scores),
                                    'direction': torch.from_numpy(direction)})

                for offset, record in enumerate(block_records):
                    item = dataset[start+offset]
                    context = item['context']
                    with torch.no_grad():
                        choice = veto({k: v.unsqueeze(0).to('cuda:0') for k, v in context.items()})
                    mode = int(choice['final_mode'][0])
                    selected = context['proposals'][mode].clone()
                    variants = brake_bank(selected.numpy())
                    future = pool.submit(score_candidates, paths[record['token']], variants, not verified)
                    verified = True
                    pending.append((future, selected, mode, float(item['scores'][mode])))
                    if len(pending) >= args.score_workers*2:
                        finish(pending.pop(0))
                for job in pending:
                    finish(job)
                save_entry(path, {'tokens': [r['token'] for r in block_records],
                                  **{k: torch.stack([e[k] for e in entries]) for k in entries[0]}})
    print(f'PTR shard {args.shard_index} complete: {root}')


def diagnose(args):
    root = Path(args.edit_cache)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['sources'] != source_identity():
        raise ValueError('Cache source differs')
    summary = {}
    for split in ('train', 'val'):
        records = json.loads((root/f'{split}_records.json').read_text())
        data = read_blocks(root, split, records)
        labels, scores, direction = [data[k].numpy() for k in ('labels', 'scores', 'direction')]
        modes = safe_oracle(labels, scores, direction)
        rows = np.arange(len(scores))
        gain = scores[rows, modes]-scores[:, 0]
        failed = labels[:, 0, 3] < 1.
        summary[split] = {'scenes': len(scores), 'original_pdm': float(scores[:, 0].mean()),
                          'controlled_oracle_pdm': float(scores[rows, modes].mean()),
                          'gain_points': float(gain.mean()*100), 'improved': int((gain>1e-6).sum()),
                          'ttc_failed': int(failed.sum()),
                          'ttc_rescued': int((failed & (labels[rows, modes, 3] == 1.)).sum()),
                          'teacher_action_counts': np.bincount(modes, minlength=len(ACTIONS)).tolist()}
    write_new_json(root/'summary.json', summary)
    print(json.dumps(summary, indent=2))


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from navsim.agents.diffusiondrive.post_selection.training import RefinementModule
    pl.seed_everything(0, workers=True)
    if not args.smoke:
        summary = json.loads((Path(args.edit_cache)/'summary.json').read_text())
        if summary['train']['improved'] == 0 or summary['val']['improved'] == 0:
            raise ValueError('Controlled Oracle has no demonstrated gain; do not train this action space')
    train_set = RefinementDataset(args.candidate_cache, args.edit_cache, 'train', args.smoke)
    val_set = RefinementDataset(args.candidate_cache, args.edit_cache, 'val', args.smoke)
    if train_set.manifest != val_set.manifest:
        raise ValueError('Training and validation cache identity mismatch')
    model = RefinementModule(train_set.manifest, args.lr)
    if args.resume:
        ckpt = load_torch(args.resume)
        if ckpt['refinement_metadata']['identity'] != train_set.manifest:
            raise ValueError('Resume cache mismatch')
    # Lightning subprocesses inherit this run path; separate invocations do not.
    if 'PTR_RUN_DIR' not in os.environ:
        os.environ['PTR_RUN_DIR'] = str(new_run(args.output))
    run = Path(os.environ['PTR_RUN_DIR'])
    callback = ModelCheckpoint(dirpath=run/'checkpoints', monitor='val_pdm', mode='max',
                               save_top_k=1, save_last=True, filename='epoch={epoch:02d}',
                               auto_insert_metric_name=False)
    trainer = pl.Trainer(default_root_dir=str(run), accelerator='gpu', devices=args.devices,
                         strategy='ddp' if args.devices > 1 else 'auto', precision='32-true',
                         max_epochs=args.epochs, callbacks=[callback],
                         logger=CSVLogger(str(run), name='csv'), num_sanity_val_steps=0,
                         limit_train_batches=2 if args.smoke else 1.,
                         limit_val_batches=2 if args.smoke else 1., gradient_clip_val=1.)
    if trainer.global_rank == 0:
        write_new_json(run/'run.json', {'arguments': vars(args), 'identity': train_set.manifest,
                                       'train_scenes': len(train_set), 'val_scenes': len(val_set)})
    trainer.fit(model, loader(train_set, args.workers, batch_size=args.batch_size, shuffle=True),
                loader(val_set, args.workers, batch_size=args.batch_size), ckpt_path=args.resume)
    if trainer.global_rank == 0:
        print(f'Best validation PDM checkpoint: {callback.best_model_path}')
        print(f'Last checkpoint: {callback.last_model_path}')
        print('Calibration may choose identity. This means no validated learned gain.')
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def evaluate(args):
    args.split, args.feature_cache = 'navtest', None
    generator, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    veto = checked_veto(args, identity)
    ckpt = load_torch(args.refiner)
    meta = ckpt['refinement_metadata']
    if (meta['identity']['provenance'] != identity or meta['identity']['sources'] != source_identity()
            or meta['identity']['veto_sha256'] != sha256(args.veto)):
        raise ValueError('Refiner baseline/selector/source identity mismatch')
    model = PostSelectionRefiner().eval().to(device)
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
            future, token, mode, output_gain, output_risk = job
            labels, scores, direction = future.result()
            oracle = int(safe_oracle(labels, scores, direction))
            row = {'token': token, 'action': mode, 'predicted_gain': output_gain,
                   'predicted_unsafe': output_risk, 'original_pdm': float(scores[0]),
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
                    mode = int(decide(output, policy['margin'], policy['risk_limit'])[0]) if policy['enabled'] else 0
                # Decision is finished BEFORE official labels are computed.
                future = pool.submit(score_candidates, paths[record['token']], variants, i == 0)
                pending.append((future, record['token'], mode, float(output['gain'][0, mode]),
                                float(output['unsafe_logits'][0, mode].sigmoid())))
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('candidate-cache', 'veto', 'pcs-scorer', 'metric-cache', 'output'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--limit', type=int, default=0)
    prep.add_argument('--score-workers', type=int, default=4)
    prep.add_argument('--num-shards', type=int, default=1)
    prep.add_argument('--shard-index', type=int, default=0)
    diag = sub.add_parser('diagnose')
    diag.add_argument('--edit-cache', required=True)
    tr = sub.add_parser('train')
    for name in ('candidate-cache', 'edit-cache', 'output'):
        tr.add_argument('--'+name, required=True)
    tr.add_argument('--devices', type=int, default=4)
    tr.add_argument('--batch-size', type=int, default=32)
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--workers', type=int, default=0)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--smoke', action='store_true')
    tr.add_argument('--resume')
    ev = sub.add_parser('evaluate')
    for name in ('baseline', 'backbone', 'anchor', 'pcs-scorer', 'veto', 'refiner', 'metric-cache', 'data-root', 'output'):
        ev.add_argument('--'+name, required=True)
    ev.add_argument('--seed', type=int, default=0)
    ev.add_argument('--max-scenes', type=int, default=0)
    ev.add_argument('--workers', type=int, default=0)
    ev.add_argument('--score-workers', type=int, default=4)
    args = p.parse_args()
    if getattr(args, 'score_workers', 1) < 1:
        raise ValueError('score-workers must be positive')
    if getattr(args, 'limit', 0) < 0 or getattr(args, 'max_scenes', 0) < 0:
        raise ValueError('Scene limits cannot be negative')
    {'prepare': prepare, 'diagnose': diagnose, 'train': train, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
