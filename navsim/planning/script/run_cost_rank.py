"""3.1.05_6: frozen PCS + separately trained cost-weighted ranking correction."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from navsim.agents.diffusiondrive.pcs.common import load_torch, load_scorer, sha256, write_new_json
from navsim.agents.diffusiondrive.cost_rank.data import CompactDataset, source_hashes
from navsim.agents.diffusiondrive.cost_rank.model import CostRanker, FrozenPCS, select
from navsim.agents.diffusiondrive.cost_rank.pipeline import (
    loader, prepare, fold_identity, predict_fold, assemble_oof, validate_oof,
)
from navsim.agents.diffusiondrive.cost_rank.metrics import reports


def make_run(root):
    # DDP subprocesses inherit the path. Bash launcher clears stale parent values.
    if 'COST_RANK_RUN_DIR' not in os.environ:
        root = Path(root)/datetime.now().strftime('%Y.%m.%d.%H.%M.%S.%f')
        root.mkdir(parents=True, exist_ok=False)
        os.environ['COST_RANK_RUN_DIR'] = str(root)
    return Path(os.environ['COST_RANK_RUN_DIR'])


def finish_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def train_fold(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
    from navsim.agents.diffusiondrive.cost_rank.training import FoldTeacher
    data = CandidateDataset(args.cache, 'train')
    identity, train_indices, _ = fold_identity(args, data)
    result_path = Path(args.oof_root)/f'fold_{args.fold_index}.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        old = load_torch(result['checkpoint'])
        if old['fold_metadata'] != identity or old['epoch']+1 != args.epochs:
            raise ValueError('Existing fold run differs; use a new OOF root')
        print('Completed teacher already exists:', result['checkpoint'])
        return
    pl.seed_everything(identity['seed'], workers=True)
    run = make_run(Path(args.oof_root)/f'fold_{args.fold_index}_runs')
    module = FoldTeacher(identity, data.manifest['settings'], args.lr)
    callback = ModelCheckpoint(dirpath=str(run/'checkpoints'), save_top_k=0, save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices,
                         strategy='ddp' if args.devices > 1 else 'auto',
                         precision='32-true', max_epochs=args.epochs, gradient_clip_val=1.,
                         num_sanity_val_steps=0, log_every_n_steps=20,
                         logger=CSVLogger(str(run), name='csv'), callbacks=[callback],
                         default_root_dir=str(run), enable_model_summary=True)
    if trainer.is_global_zero:
        write_new_json(run/'run.json', dict(identity=identity, arguments=vars(args)))
    trainer.fit(module, loader(Subset(data, train_indices), args.workers, batch_size=args.batch_size,
                               shuffle=True, pin_memory=True), ckpt_path=args.resume)
    if trainer.is_global_zero:
        write_new_json(result_path, dict(checkpoint=callback.last_model_path, identity=identity))
        print('OOF teacher complete:', result_path)
    finish_ddp()


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from navsim.agents.diffusiondrive.cost_rank.training import RankModule
    pl.seed_everything(args.seed, workers=True)
    oof = None if args.smoke else validate_oof(args.features, args.oof_root)
    train_data = CompactDataset(args.features, 'train',
                                None if args.smoke else Path(args.oof_root)/'oof_scores.npy', args.smoke)
    val_data = CompactDataset(args.features, 'val', smoke=args.smoke)
    identity = dict(schema='cost_rank_model_v1', features=train_data.manifest,
                    feature_manifest_sha256=sha256(Path(args.features)/'manifest.json'),
                    train_records_sha256=sha256(Path(args.features)/'train_records.json'),
                    val_records_sha256=sha256(Path(args.features)/'val_records.json'),
                    oof=oof, sources=source_hashes(), seed=args.seed)
    loss_settings = dict(min_gap=.02, temperature=.05, top_k=5, hard_weight=3., cost_cap=5.)
    module = RankModule(identity, dict(width=128, cap=.2), loss_settings, args.lr, args.smoke)
    # Before the first update, no validation tuning or extra auxiliary objective.
    first = next(iter(loader(train_data, 0, batch_size=2)))
    with torch.no_grad():
        residual = module.model(first)
    if torch.count_nonzero(residual) or not torch.equal(select(first['pcs_scores'], residual),
                                                       first['pcs_scores'].argmax(-1)):
        raise ValueError('Zero initialization must retain original PCS choices')
    run = make_run(args.output)
    callback = ModelCheckpoint(dirpath=str(run/'checkpoints'), filename='epoch={epoch:02d}',
                               auto_insert_metric_name=False, monitor='calibration_pdm', mode='max',
                               save_top_k=1, save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices,
                         strategy='ddp' if args.devices > 1 else 'auto',
                         max_epochs=args.epochs, precision='32-true', gradient_clip_val=1.,
                         num_sanity_val_steps=0, log_every_n_steps=20,
                         limit_train_batches=2 if args.smoke else 1.,
                         logger=CSVLogger(str(run), name='csv'), callbacks=[callback],
                         default_root_dir=str(run))
    if trainer.is_global_zero:
        write_new_json(run/'run.json', dict(arguments=vars(args), identity=identity,
                         train_scenes=len(train_data), val_scenes=len(val_data),
                         global_batch_size=args.batch_size*args.devices, objective='single_cost_weighted_rank',
                         frozen='K67 and original PCS; neither is part of the optimizer'))
    trainer.fit(module,
                loader(train_data, args.workers, batch_size=args.batch_size, shuffle=True, pin_memory=True),
                loader(val_data, args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True),
                ckpt_path=args.resume)
    if trainer.is_global_zero:
        write_new_json(run/'result.json', dict(best_checkpoint=callback.best_model_path,
                         last_checkpoint=callback.last_model_path, smoke=args.smoke,
                         note='Checkpoint/alpha selected on calibration only. Read audit report before navtest.'))
        print('Best ranking checkpoint:', callback.best_model_path)
        print('Last ranking checkpoint:', callback.last_model_path)
        print('Run result:', run/'result.json')
    finish_ddp()


def load_ranker(path, device, allow_smoke=False):
    checkpoint = load_torch(path)
    meta = checkpoint['rank_metadata']
    if meta['identity']['schema'] != 'cost_rank_model_v1' or meta['identity']['sources'] != source_hashes():
        raise ValueError('Ranking checkpoint schema/source mismatch')
    if meta['smoke'] and not allow_smoke:
        raise ValueError('Smoke checkpoint cannot be used for formal evaluation')
    model = CostRanker(**meta['settings'])
    model.load_state_dict({k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
    return model.eval().to(device), meta


def validate(args):
    model, meta = load_ranker(args.ranker, 'cuda:0')
    data = CompactDataset(args.features, 'val')
    if (meta['identity']['features'] != data.manifest or
            meta['identity']['val_records_sha256'] != sha256(Path(args.features)/'val_records.json')):
        raise ValueError('Validation cache differs from training identity')
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader(data, args.workers, batch_size=args.batch_size), desc='Cached validation'):
            residual = model({k: v.to('cuda:0') for k, v in batch.items()})
            rows.append({**{k: batch[k] for k in ('index', 'calibration', 'pcs_scores', 'labels', 'scores', 'direction')},
                         'residual': residual.cpu()})
    merged = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
    result = reports(merged, meta['policy'])
    run = make_run(args.output)
    write_new_json(run/'report.json', result)
    torch.save(merged, run/'predictions.pt')
    print(json.dumps(result, indent=2))
    print('Validation report:', run/'report.json')


def evaluate(args):
    from navsim.planning.script.run_pcs import prepare_source, new_run, write_csv
    from navsim.agents.diffusiondrive.pcs.common import provenance, generate_context
    from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
    from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES
    from navsim.agents.diffusiondrive.pcs.veto import load_veto
    args.split, args.feature_cache = 'navtest', None
    generator, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    ranker, meta = load_ranker(args.ranker, device, allow_smoke=bool(args.max_scenes))
    fm = meta['identity']['features']
    if fm['provenance'] != identity or fm['pcs_sha256'] != sha256(args.scorer):
        raise ValueError('Ranker does not match original generator/PCS')
    scorer, _ = load_scorer(args.scorer, device, identity)
    frozen = FrozenPCS(scorer).eval()
    reference, reference_meta = load_veto(args.reference_trv, device, identity)
    if reference_meta['pcs_scorer_sha256'] != fm['pcs_sha256']:
        raise ValueError('Reference TRV uses a different original PCS')
    policy = meta['policy']
    run = new_run(args.output)
    write_new_json(run/'run.json', dict(arguments=vars(args), policy=policy, rank_metadata=meta))
    rows, pending = [], []
    stream = (run/'paired_results.csv').open('x', newline='', encoding='utf-8')
    writer = None
    prefixes = ('rank', 'pcs', 'trv', 'base')

    def finish(job):
        nonlocal writer
        future, record, modes = job
        labels, scores, direction = future.result()
        row = dict(token=record['token'], log_name=record['log_name'], valid=True)
        for i, prefix in enumerate(prefixes):
            row.update({prefix+'_'+name: float(labels[i, j]) for j, name in enumerate(METRIC_NAMES)})
            row[prefix+'_score'], row[prefix+'_direction'] = float(scores[i]), float(direction[i])
            row[prefix+'_mode'] = modes[i]
        row['delta_vs_pcs'] = float(scores[0]-scores[1])
        row['delta_vs_trv'] = float(scores[0]-scores[2])
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context('spawn')) as pool:
            for index, (record, features) in enumerate(tqdm(loader(source, args.workers, batch_size=None),
                                                          desc='Fixed K67: rank vs PCS vs TRV')):
                context = generate_context(generator, features, record['token'], args.seed, device)
                gpu = {k: v.unsqueeze(0).to(device) for k, v in context.items()}
                with torch.no_grad():
                    compact = frozen(gpu)
                    residual = ranker(compact)
                    pcs_mode = compact['pcs_scores'].argmax(-1)
                    base_mode = gpu['base_logits'].argmax(-1)
                    reference_output = reference(gpu, selected_mode=pcs_mode, base_mode=base_mode)
                    modes = [int(select(compact['pcs_scores'], residual, policy['alpha'])[0]),
                             int(pcs_mode[0]), int(reference_output['final_mode'][0]), int(base_mode[0])]
                # True labels are read only AFTER all deployable choices are fixed.
                future = pool.submit(score_candidates, paths[record['token']],
                                     context['proposals'][modes].numpy(), index == 0)
                pending.append((future, record, modes))
                if index == 0 or len(pending) >= 2*args.score_workers:
                    finish(pending.pop(0))
            for job in pending:
                finish(job)
    finally:
        stream.close()
    if len(rows) != len(source):
        raise ValueError('Incomplete navtest evaluation')
    summary = dict(scenes=len(rows), completed=True, policy=policy)
    for prefix in prefixes:
        standard = [dict(token=r['token'], valid=True, score=r[prefix+'_score'],
                         driving_direction_compliance=r[prefix+'_direction'],
                         **{k: r[prefix+'_'+k] for k in METRIC_NAMES}) for r in rows]
        average = {k: float(np.mean([r[k] for r in standard])) for k in
                   (*METRIC_NAMES, 'driving_direction_compliance', 'score')}
        write_csv(run/f'{prefix}.csv', standard+[dict(token='average', valid=True, **average)])
        summary[prefix] = average
    for ref in ('pcs', 'trv'):
        delta = np.asarray([r['rank_score']-r[ref+'_score'] for r in rows])
        summary['vs_'+ref] = dict(gain_points=float(delta.mean()*100), beneficial=int((delta > 1e-6).sum()),
                                  harmful=int((delta < -1e-6).sum()), severe_losses=int((delta <= -.2).sum()))
    write_new_json(run/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print('Paired navtest:', run)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    q = subs.add_parser('prepare')
    q.add_argument('--cache', required=True)
    q.add_argument('--scorer', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--limit', type=int, default=0)
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--workers', type=int, default=0)
    q = subs.add_parser('train-fold')
    q.add_argument('--cache', required=True)
    q.add_argument('--oof-root', required=True)
    q.add_argument('--fold-index', type=int, required=True)
    q.add_argument('--folds', type=int, default=3)
    q.add_argument('--epochs', type=int, default=20)
    q.add_argument('--devices', type=int, default=4)
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--workers', type=int, default=0)
    q.add_argument('--lr', type=float, default=3e-4)
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--resume')
    q = subs.add_parser('predict-fold')
    q.add_argument('--cache', required=True)
    q.add_argument('--oof-root', required=True)
    q.add_argument('--fold-index', type=int, required=True)
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--workers', type=int, default=0)
    q = subs.add_parser('assemble-oof')
    q.add_argument('--features', required=True)
    q.add_argument('--oof-root', required=True)
    q.add_argument('--folds', type=int, default=3)
    q = subs.add_parser('train')
    q.add_argument('--features', required=True)
    q.add_argument('--oof-root')
    q.add_argument('--output', required=True)
    q.add_argument('--epochs', type=int, default=20)
    q.add_argument('--devices', type=int, default=4)
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--workers', type=int, default=0)
    q.add_argument('--lr', type=float, default=1e-4)
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--resume')
    q = subs.add_parser('validate')
    q.add_argument('--features', required=True)
    q.add_argument('--ranker', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--workers', type=int, default=0)
    q = subs.add_parser('evaluate')
    for key in ('baseline', 'backbone', 'anchor', 'scorer', 'reference-trv', 'ranker',
                'metric-cache', 'data-root', 'output'):
        q.add_argument('--'+key, required=True)
    q.add_argument('--max-scenes', type=int, default=0)
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--workers', type=int, default=0)
    q.add_argument('--score-workers', type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    for name in ('epochs', 'devices', 'batch_size', 'score_workers'):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(name+' must be positive')
    for name in ('workers', 'limit', 'max_scenes'):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(name+' must be nonnegative')
    if args.command == 'train' and not args.smoke and not args.oof_root:
        raise ValueError('Formal rank training requires --oof-root')
    # Same deterministic FP32 frozen PCS as cache/evaluation, no TF32 drift.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')
    {'prepare': prepare, 'train-fold': train_fold, 'predict-fold': predict_fold,
     'assemble-oof': assemble_oof, 'train': train, 'validate': validate, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
