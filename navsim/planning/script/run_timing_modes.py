"""3.1.05_10: top-five spatial paths × four timing modes, trained independently."""
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
from tqdm import tqdm

from navsim.agents.diffusiondrive.cost_rank.data import CompactDataset
from navsim.agents.diffusiondrive.cost_rank.model import FrozenPCS
from navsim.agents.diffusiondrive.cost_rank.pipeline import loader
from navsim.agents.diffusiondrive.pcs.common import (
    generate_context, load_scorer, load_torch, provenance, sha256, write_new_json,
)
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset, entry_path, metric_paths
from navsim.agents.diffusiondrive.pcs.model import METRIC_NAMES
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.agents.diffusiondrive.timing_modes.data import (
    BLOCK, VARIANTS, TimingModeDataset, inputs_from_source, implementation_hashes,
)
from navsim.agents.diffusiondrive.timing_modes.geometry import (
    MODE_NAMES, SPATIAL_ONLY_INDICES, TOP_SPATIAL, expand_top, top_modes,
)
from navsim.agents.diffusiondrive.timing_modes.metrics import reports
from navsim.agents.diffusiondrive.timing_modes.model import TimingModeValue, choose
from navsim.planning.script.run_cost_rank import load_ranker


def make_run(root):
    if 'TIMING_MODES_RUN_DIR' not in os.environ:
        path = Path(root)/datetime.now().strftime('%Y.%m.%d.%H.%M.%S.%f')
        path.mkdir(parents=True, exist_ok=False)
        os.environ['TIMING_MODES_RUN_DIR'] = str(path)
    return Path(os.environ['TIMING_MODES_RUN_DIR'])


def finish_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def cache_identity(args, features):
    return dict(schema='timing_modes_v1', features=features.manifest,
                ranker_sha256=sha256(args.ranker), scorer_sha256=sha256(args.scorer),
                candidate_manifest_sha256=sha256(Path(args.candidate_cache)/'manifest.json'),
                modes=list(MODE_NAMES),
                top_spatial=TOP_SPATIAL, block=BLOCK,
                implementation_sha256=implementation_hashes(),
                limit=features.manifest['limit'])


def init_cache(args):
    root = Path(args.output)
    train = CompactDataset(args.features, 'train', smoke=args.smoke)
    val = CompactDataset(args.features, 'val', smoke=args.smoke)
    _, rank_meta = load_ranker(args.ranker, 'cpu', allow_smoke=args.smoke)
    if rank_meta['identity']['features'] != train.manifest:
        raise ValueError('Ranker and K67 feature cache mismatch')
    if (train.manifest['pcs_sha256'] != sha256(args.scorer) or
            train.manifest['candidate_manifest_sha256'] !=
            sha256(Path(args.candidate_cache)/'manifest.json')):
        raise ValueError('PCS scorer/candidate cache differs from locked feature provenance')
    raw_train = CandidateDataset(args.candidate_cache, 'train')
    raw_val = CandidateDataset(args.candidate_cache, 'val')
    if (raw_train.provenance != train.manifest['provenance'] or
            raw_val.provenance != val.manifest['provenance'] or
            [r['token'] for r in raw_train.records] != [r['token'] for r in train.records] or
            [r['token'] for r in raw_val.records] != [r['token'] for r in val.records]):
        raise ValueError('Original K67 candidate cache differs from compact features')
    manifest = cache_identity(args, train)
    write_new_json(root/'manifest.json', manifest)
    for data in (train, val):
        write_new_json(root/f'{data.split}_records.json', data.records)
        (root/data.split).mkdir(parents=True, exist_ok=True)
    print('Timing-mode cache initialized:', root)


def verify_cache(args, data):
    root = Path(args.output)
    existing = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if existing != cache_identity(args, data):
        raise ValueError('Timing-mode cache provenance mismatch')
    records = json.loads((root/f'{data.split}_records.json').read_text(encoding='utf-8'))
    if records != data.records:
        raise ValueError('Timing-mode record order mismatch')


def rank_candidates(ranker, source, alpha, device):
    gpu = {key: value[None].to(device) for key, value in source.items()
           if key in ('features', 'subscores', 'pcs_scores', 'proposals', 'base_logits')}
    with torch.no_grad():
        residual = ranker(gpu)
    scores = source['pcs_scores'].float().numpy()+alpha*residual[0].cpu().numpy()
    return top_modes(scores), scores


def prepare_shard(args):
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index')
    data = CompactDataset(args.features, args.split, smoke=args.smoke)
    verify_cache(args, data)
    ranker, rank_meta = load_ranker(args.ranker, 'cuda:0', allow_smoke=args.smoke)
    if not rank_meta['policy']['enabled']:
        raise ValueError('Locked Ranker must be enabled')
    scorer, _ = load_scorer(args.scorer, 'cuda:0', data.manifest['provenance'])
    frozen = FrozenPCS(scorer).eval()
    paths = metric_paths(args.metric_cache)
    missing = [r['token'] for r in data.records if r['token'] not in paths]
    if missing:
        raise ValueError(f'Missing {len(missing)} metric caches; first: {missing[:3]}')
    root = Path(args.output)/args.split
    indices = [start for start in range(0, len(data), BLOCK)
               if (start//BLOCK) % args.num_shards == args.shard_index]
    pending = [start for start in indices if not (root/f'block_{start//BLOCK:05d}.pt').exists()]
    print(f'{args.split} shard {args.shard_index}: {len(pending)}/{len(indices)} blocks pending', flush=True)
    with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context('spawn')) as pool:
        for start in tqdm(pending, desc=f'Timing modes {args.split}/{args.shard_index}'):
            jobs = []
            for index in range(start, min(start+BLOCK, len(data))):
                source = data[index]
                modes, rank_scores = rank_candidates(ranker, source,
                                                     rank_meta['policy']['alpha'], 'cuda:0')
                trajectories = expand_top(source['proposals'][modes].numpy())
                token = data.records[index]['token']
                original_entry = load_torch(entry_path(args.candidate_cache, data.records[index]))
                if original_entry['token'] != token or original_entry['provenance'] != data.manifest['provenance']:
                    raise ValueError(f'Original candidate entry differs: {token}')
                scene = original_entry['context']
                variant_context = {key: scene[key][None].to('cuda:0')
                                   for key in ('bev', 'agents', 'ego')}
                variant_context['proposals'] = torch.from_numpy(trajectories)[None].to('cuda:0')
                variant_context['base_logits'] = scene['base_logits'][modes].repeat_interleave(
                    len(MODE_NAMES))[None].to('cuda:0')
                with torch.no_grad():
                    variant = frozen(variant_context)
                variant = {key: variant[key][0].cpu() for key in
                           ('features', 'subscores', 'pcs_scores')}
                if not all(torch.isfinite(value).all() for value in variant.values()):
                    raise ValueError(f'Nonfinite frozen PCS timing prediction: {token}')
                future = pool.submit(score_candidates, paths[token], trajectories, index == 0)
                jobs.append((future, source, modes, rank_scores, trajectories, variant, token))
            piece = dict(tokens=[], modes=[], rank_scores=[], trajectories=[],
                         labels=[], scores=[], direction=[], variant_features=[],
                         variant_subscores=[], variant_pcs_scores=[])
            for future, source, modes, rank_scores, trajectories, variant, token in jobs:
                labels, scores, direction = future.result()
                if labels.shape != (VARIANTS, 5) or scores.shape != (VARIANTS,):
                    raise ValueError('Unexpected PDM timing label shape')
                identity = np.arange(TOP_SPATIAL)*len(MODE_NAMES)
                np.testing.assert_allclose(scores[identity], source['scores'][modes].numpy(),
                                           atol=1e-5, rtol=0, err_msg=f'Identity score drift: {token}')
                np.testing.assert_allclose(labels[identity], source['labels'][modes].numpy(),
                                           atol=1e-5, rtol=0, err_msg=f'Identity labels drift: {token}')
                piece['tokens'].append(token)
                for key, value in (('modes', modes), ('rank_scores', rank_scores),
                                   ('trajectories', trajectories), ('labels', labels),
                                   ('scores', scores), ('direction', direction),
                                   ('variant_features', variant['features']),
                                   ('variant_subscores', variant['subscores']),
                                   ('variant_pcs_scores', variant['pcs_scores'])):
                    piece[key].append(torch.as_tensor(np.array(value, copy=True)))
            for key in piece:
                if key != 'tokens':
                    piece[key] = torch.stack(piece[key])
            target = root/f'block_{start//BLOCK:05d}.pt'
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            torch.save(piece, temporary)
            os.replace(temporary, target)
    print(f'Timing modes shard {args.shard_index} complete: {root}', flush=True)


def complete_cache(args):
    root = Path(args.output)
    counts = {}
    for split in ('train', 'val'):
        data = CompactDataset(args.features, split, smoke=args.smoke)
        verify_cache(args, data)
        for start in tqdm(range(0, len(data), BLOCK), desc=f'Check {split} blocks'):
            path = root/split/f'block_{start//BLOCK:05d}.pt'
            if not path.is_file():
                raise ValueError(f'Missing timing block {path}')
            item = load_torch(path)
            if item['tokens'] != [r['token'] for r in data.records[start:start+BLOCK]]:
                raise ValueError(f'Timing block token mismatch: {path}')
            if item['scores'].shape != (len(item['tokens']), VARIANTS):
                raise ValueError(f'Timing block score shape mismatch: {path}')
            if (item['variant_features'].shape != (len(item['tokens']), VARIANTS, 512) or
                    item['variant_subscores'].shape != (len(item['tokens']), VARIANTS, 5) or
                    item['variant_pcs_scores'].shape != (len(item['tokens']), VARIANTS)):
                raise ValueError(f'Timing block frozen PCS shape mismatch: {path}')
        counts[split] = len(data)
    write_new_json(root/'complete.json', dict(manifest_sha256=sha256(root/'manifest.json'),
                                              counts=counts))
    print('Complete timing-mode labels:', root, counts)


def diagnose(args):
    """Privileged offline upper bound, never an inference policy or threshold."""
    data = TimingModeDataset(args.features, args.cache, 'val')
    labels, scores, direction = (data.cached[key].float() for key in
                                 ('labels', 'scores', 'direction'))
    baseline = labels[:, :1, [0, 1, 3, 4]]
    safe = (labels[:, :, [0, 1, 3, 4]] >= baseline-1e-7).all(-1)
    safe &= direction >= direction[:, :1]-1e-7
    safe[:, 0] = True

    def best(allowed):
        eligible = safe.clone()
        mask = torch.zeros(VARIANTS, dtype=torch.bool)
        mask[list(allowed)] = True
        eligible &= mask
        choice = scores.masked_fill(~eligible, -float('inf')).argmax(-1)
        return scores[torch.arange(len(scores)), choice]

    full = best(range(VARIANTS))
    spatial = best((0, *SPATIAL_ONLY_INDICES))
    result = {}
    for name, mask in (('calibration', data.compact.calibration),
                       ('audit', ~data.compact.calibration)):
        reference = scores[mask, 0]
        result[name] = dict(scenes=int(mask.sum()), rank_pdm=float(reference.mean()),
                            spatial_only_oracle_pdm=float(spatial[mask].mean()),
                            full_oracle_pdm=float(full[mask].mean()),
                            timing_incremental_oracle_gain_points=float(
                                (full[mask]-spatial[mask]).double().mean()*100.))
    run = make_run(args.output)
    write_new_json(run/'oracle_report.json', result)
    print(json.dumps(result, indent=2))
    print('Privileged timing-mode upper bound:', run/'oracle_report.json')


def load_timing(path, device, expected=None, allow_smoke=False):
    checkpoint = load_torch(path)
    meta = checkpoint['timing_modes_metadata']
    if meta['identity']['schema'] != 'timing_modes_v1':
        raise ValueError('Timing-mode checkpoint schema mismatch')
    if expected is not None and meta['identity'] != expected:
        raise ValueError('Timing-mode checkpoint/cache mismatch')
    if meta['smoke'] and not allow_smoke:
        raise ValueError('Smoke timing checkpoint cannot run formal evaluation')
    model = TimingModeValue(width=meta['settings']['width'])
    if model.settings != meta['settings']:
        raise ValueError('Timing-mode architecture mismatch')
    model.load_state_dict({key[6:]: value for key, value in checkpoint['state_dict'].items()
                           if key.startswith('model.')})
    return model.eval().to(device), meta


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from navsim.agents.diffusiondrive.timing_modes.training import TimingModeModule

    pl.seed_everything(args.seed, workers=True)
    train_data = TimingModeDataset(args.features, args.cache, 'train', smoke=args.smoke)
    val_data = TimingModeDataset(args.features, args.cache, 'val', smoke=args.smoke)
    identity = json.loads((Path(args.cache)/'manifest.json').read_text(encoding='utf-8'))
    module = TimingModeModule(identity, lr=args.lr, smoke=args.smoke)
    first = next(iter(loader(train_data, 0, batch_size=2)))
    with torch.no_grad():
        if torch.count_nonzero(module.model(first)):
            raise ValueError('Zero initialization must preserve locked Rank output')
    run = make_run(args.output)
    callback = ModelCheckpoint(dirpath=str(run/'checkpoints'), filename='epoch={epoch:02d}',
                               auto_insert_metric_name=False, monitor='calibration_pdm',
                               mode='max', save_top_k=1, save_last=True)
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
                                           objective='one cost-weighted timing candidate rank loss',
                                           frozen='K67 generator, PCS and 3.1.05_6 Ranker'))
    trainer.fit(module,
                loader(train_data, args.workers, batch_size=args.batch_size,
                       shuffle=True, pin_memory=True),
                loader(val_data, args.workers, batch_size=args.batch_size,
                       shuffle=False, pin_memory=True),
                ckpt_path=args.resume)
    if trainer.is_global_zero:
        write_new_json(run/'result.json', dict(best_checkpoint=callback.best_model_path,
                                               last_checkpoint=callback.last_model_path,
                                               smoke=args.smoke,
                                               note='Select checkpoint on calibration only; inspect audit before navtest.'))
        print('Best timing-mode checkpoint:', callback.best_model_path)
        print('Last timing-mode checkpoint:', callback.last_model_path)
        print('Run result:', run/'result.json')
    finish_ddp()


def validate(args):
    identity = json.loads((Path(args.cache)/'manifest.json').read_text(encoding='utf-8'))
    model, meta = load_timing(args.timing, 'cuda:0', identity)
    data = TimingModeDataset(args.features, args.cache, 'val')
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader(data, args.workers, batch_size=args.batch_size),
                          desc='Independent timing audit'):
            prediction = model({key: value.to('cuda:0') for key, value in batch.items()
                                if key in ('features', 'subscores', 'pcs_scores', 'rank_scores',
                                           'base_probability', 'original', 'trajectories')})
            rows.append({**{key: batch[key] for key in ('index', 'calibration', 'labels',
                                                       'scores', 'direction')},
                         'predicted': prediction.cpu()})
    merged = {key: torch.cat([r[key] for r in rows]) for key in rows[0]}
    if not torch.equal(merged['index'], torch.arange(len(data))):
        raise ValueError('Cached audit coverage incomplete')
    result = reports(merged, meta['policy'])
    run = make_run(args.output)
    write_new_json(run/'report.json', result)
    print(json.dumps(result, indent=2))
    print('Locked timing-mode audit:', run/'report.json')


def evaluate(args):
    from navsim.planning.script.run_pcs import prepare_source, new_run, write_csv

    args.split, args.feature_cache = 'navtest', None
    generator, _, source, paths, device = prepare_source(args)
    identity = provenance(args.baseline, args.anchor, args.seed)
    ranker, rank_meta = load_ranker(args.ranker, device, allow_smoke=bool(args.max_scenes))
    features = rank_meta['identity']['features']
    if features['provenance'] != identity or features['pcs_sha256'] != sha256(args.scorer):
        raise ValueError('Ranker does not match original generator/PCS')
    cache_identity_from_training = json.loads((Path(args.cache)/'manifest.json').read_text(encoding='utf-8'))
    if (cache_identity_from_training['features'] != features or
            cache_identity_from_training['ranker_sha256'] != sha256(args.ranker) or
            cache_identity_from_training['scorer_sha256'] != sha256(args.scorer) or
            cache_identity_from_training['implementation_sha256'] != implementation_hashes()):
        raise ValueError('Timing training differs from locked Ranker/features')
    model, meta = load_timing(args.timing, device, cache_identity_from_training,
                              allow_smoke=bool(args.max_scenes))
    if not meta['policy']['enabled']:
        raise ValueError('Timing policy is disabled; no new hypothesis to evaluate')
    scorer, _ = load_scorer(args.scorer, device, identity)
    frozen = FrozenPCS(scorer).eval()
    run = new_run(args.output)
    write_new_json(run/'run.json', dict(arguments=vars(args), timing_policy=meta['policy'],
                                        rank_policy=rank_meta['policy']))
    rows, pending = [], []
    stream = (run/'paired_results.csv').open('x', newline='', encoding='utf-8')
    writer = None
    prefixes = ('timing', 'spatial', 'rank', 'pcs', 'base')

    def finish(job):
        nonlocal writer
        future, record, timing_index, spatial_index, rank_index, pcs_index, base_index = job
        labels, scores, direction = future.result()
        row = dict(token=record['token'], log_name=record['log_name'], valid=True,
                   timing_index=timing_index, spatial_index=spatial_index, rank_mode=rank_index,
                   pcs_mode=pcs_index, base_mode=base_index,
                   timing_family=MODE_NAMES[timing_index % len(MODE_NAMES)])
        for i, prefix in enumerate(prefixes):
            row.update({prefix+'_'+name: float(labels[i, j]) for j, name in enumerate(METRIC_NAMES)})
            row[prefix+'_score'], row[prefix+'_direction'] = float(scores[i]), float(direction[i])
        row['delta_vs_rank'] = float(scores[0]-scores[1])
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        rows.append(row)

    try:
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context('spawn')) as pool:
            for index, (record, features_input) in enumerate(tqdm(
                    loader(source, args.workers, batch_size=None), desc='Spatial × timing navtest')):
                context = generate_context(generator, features_input, record['token'], args.seed, device)
                gpu = {key: value.unsqueeze(0).to(device) for key, value in context.items()}
                with torch.no_grad():
                    compact = frozen(gpu)
                    residual = ranker(compact)
                    rank_scores = compact['pcs_scores'][0]+rank_meta['policy']['alpha']*residual[0]
                    spatial = top_modes(rank_scores.cpu().numpy())
                    trajectories = expand_top(context['proposals'][spatial].numpy())
                    selected = {key: value[0].cpu() for key, value in compact.items()}
                    variant_context = {key: gpu[key] for key in ('bev', 'agents', 'ego')}
                    variant_context['proposals'] = torch.from_numpy(trajectories)[None].to(device)
                    variant_context['base_logits'] = gpu['base_logits'][:, torch.as_tensor(
                        spatial, device=device)].repeat_interleave(
                        len(MODE_NAMES), -1)
                    variant = frozen(variant_context)
                    variant = {key: variant[key][0].cpu() for key in
                               ('features', 'subscores', 'pcs_scores')}
                    item = inputs_from_source(selected, spatial, rank_scores.cpu().numpy(),
                                              trajectories, variant)
                    predicted = model({key: value[None].to(device) for key, value in item.items()})
                    timing_index = int(choose(predicted, meta['policy']['threshold'])[0])
                    spatial_index = int(choose(predicted, meta['policy']['threshold'],
                                               SPATIAL_ONLY_INDICES)[0])
                    rank_index, pcs_index = int(spatial[0]), int(compact['pcs_scores'].argmax(-1)[0])
                    base_index = int(gpu['base_logits'].argmax(-1)[0])
                proposals = np.stack((trajectories[timing_index], trajectories[spatial_index],
                                      context['proposals'][rank_index].numpy(),
                                      context['proposals'][pcs_index].numpy(),
                                      context['proposals'][base_index].numpy()))
                future = pool.submit(score_candidates, paths[record['token']], proposals, index == 0)
                pending.append((future, record, timing_index, spatial_index,
                                rank_index, pcs_index, base_index))
                if index == 0 or len(pending) >= 2*args.score_workers:
                    finish(pending.pop(0))
                if args.max_scenes and index+1 >= args.max_scenes:
                    break
            for job in pending:
                finish(job)
    finally:
        stream.close()
    if len(rows) != len(source):
        raise ValueError('Incomplete navtest timing evaluation')
    summary = dict(scenes=len(rows), completed=True, timing_policy=meta['policy'],
                   rank_policy=rank_meta['policy'],
                   timing_changed=sum(r['timing_index'] != 0 for r in rows))
    for prefix in prefixes:
        standard = [dict(token=row['token'], valid=True, score=row[prefix+'_score'],
                         driving_direction_compliance=row[prefix+'_direction'],
                         **{name: row[prefix+'_'+name] for name in METRIC_NAMES}) for row in rows]
        average = {name: float(np.mean([r[name] for r in standard])) for name in
                   (*METRIC_NAMES, 'driving_direction_compliance', 'score')}
        write_csv(run/f'{prefix}.csv', standard+[dict(token='average', valid=True, **average)])
        summary[prefix] = average
    for reference in ('spatial', 'rank', 'pcs', 'base'):
        delta = np.asarray([r['timing_score']-r[reference+'_score'] for r in rows])
        summary['vs_'+reference] = dict(gain_points=float(delta.mean()*100),
                                        beneficial=int((delta > 1e-6).sum()),
                                        harmful=int((delta < -1e-6).sum()),
                                        severe_losses=int((delta <= -.2).sum()))
    write_new_json(run/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print('Paired timing-mode navtest:', run)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for command in ('init-cache', 'prepare-shard', 'complete-cache', 'diagnose',
                    'train', 'validate', 'evaluate'):
        q = commands.add_parser(command)
        if command != 'evaluate':
            q.add_argument('--features', required=True)
        if command in ('init-cache', 'prepare-shard', 'complete-cache'):
            q.add_argument('--ranker', required=True)
            q.add_argument('--scorer', required=True)
            q.add_argument('--candidate-cache', required=True)
            q.add_argument('--output', required=True)
            q.add_argument('--smoke', action='store_true')
        if command == 'prepare-shard':
            q.add_argument('--split', choices=('train', 'val'), required=True)
            q.add_argument('--metric-cache', required=True)
            q.add_argument('--num-shards', type=int, default=1)
            q.add_argument('--shard-index', type=int, default=0)
            q.add_argument('--score-workers', type=int, default=2)
        if command == 'train':
            q.add_argument('--cache', required=True)
            q.add_argument('--output', required=True)
            q.add_argument('--epochs', type=int, default=20)
            q.add_argument('--devices', type=int, default=4)
            q.add_argument('--batch-size', type=int, default=32)
            q.add_argument('--workers', type=int, default=0)
            q.add_argument('--lr', type=float, default=1e-4)
            q.add_argument('--seed', type=int, default=0)
            q.add_argument('--resume')
            q.add_argument('--smoke', action='store_true')
        if command == 'diagnose':
            q.add_argument('--cache', required=True)
            q.add_argument('--output', required=True)
        if command == 'validate':
            q.add_argument('--cache', required=True)
            q.add_argument('--timing', required=True)
            q.add_argument('--output', required=True)
            q.add_argument('--batch-size', type=int, default=32)
            q.add_argument('--workers', type=int, default=0)
        if command == 'evaluate':
            for key in ('baseline', 'backbone', 'anchor', 'scorer', 'ranker',
                        'timing', 'cache', 'metric-cache', 'data-root', 'output'):
                q.add_argument('--'+key, required=True)
            q.add_argument('--max-scenes', type=int, default=0)
            q.add_argument('--seed', type=int, default=0)
            q.add_argument('--workers', type=int, default=0)
            q.add_argument('--score-workers', type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    for key in ('epochs', 'devices', 'batch_size', 'score_workers', 'num_shards'):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(key+' must be positive')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')
    {'init-cache': init_cache, 'prepare-shard': prepare_shard,
     'complete-cache': complete_cache, 'diagnose': diagnose, 'train': train,
     'validate': validate, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
