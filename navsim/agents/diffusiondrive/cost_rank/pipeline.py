"""Reusable compact features and honest fold predictions for train logs only."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from navsim.agents.diffusiondrive.pcs.common import load_scorer, load_torch, sha256, write_new_json
from .data import FIELDS, CompactDataset, log_partitions, source_hashes, validate_records
from .model import FrozenPCS


def loader(dataset, workers=0, **kwargs):
    params = dict(num_workers=workers, **kwargs)
    if workers:
        params.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=1)
    return DataLoader(dataset, **params)


def prepare(args):
    from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
    root = Path(args.output)
    source = {s: CandidateDataset(args.cache, s) for s in ('train', 'val')}
    manifest = source['train'].manifest
    records = json.loads((Path(args.cache)/'records.json').read_text())
    validate_records(records)
    if manifest['num_candidates'] != 67:
        raise ValueError('This controlled experiment requires the original K67 bank')
    scorer, meta = load_scorer(args.scorer, 'cuda:0', manifest['provenance'])
    frozen = FrozenPCS(scorer).eval()
    identity = dict(schema='cost_rank_features_v1', candidate_root=str(Path(args.cache).resolve()),
                    candidate_manifest_sha256=sha256(Path(args.cache)/'manifest.json'),
                    candidate_records_sha256=sha256(Path(args.cache)/'records.json'),
                    provenance=manifest['provenance'], pcs_settings=meta['settings'],
                    pcs_sha256=sha256(args.scorer), feature_source_sha256=source_hashes()['model.py'],
                    limit=args.limit)
    write_new_json(root/'manifest.json', identity)
    first_checked = False
    for split, dataset in source.items():
        n = min(len(dataset), args.limit) if args.limit else len(dataset)
        records = dataset.records[:n]
        write_new_json(root/f'{split}_records.json', records)
        directory = root/split
        directory.mkdir(exist_ok=True)
        maps = {}
        for key, (dtype, shape) in dict(FIELDS, done=('bool', ())).items():
            path = directory/f'{key}.npy'
            if path.exists():
                array = np.load(path, mmap_mode='r+')
                if array.shape != (n, *shape) or array.dtype != np.dtype(dtype):
                    raise ValueError(f'Incompatible partially prepared file: {path}')
            else:
                array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(n, *shape))
                if key == 'done':
                    array[:] = False
                    array.flush()
            maps[key] = array
        pending = np.flatnonzero(~maps['done']).tolist()
        print(f'{split}: {len(pending)} pending / {n}; stored feature size ~{n*67*512*2/2**30:.2f} GiB', flush=True)
        for batch in tqdm(loader(Subset(dataset, pending), args.workers, batch_size=args.batch_size),
                          desc='Frozen PCS features/'+split):
            context = {k: v.to('cuda:0') for k, v in batch['context'].items()}
            extracted = frozen(context)
            if not first_checked:
                with torch.no_grad():
                    original = scorer(context)
                torch.testing.assert_close(extracted['pcs_scores'], original['scores'], rtol=0, atol=0)
                from .model import CostRanker, select
                ranker = CostRanker().to('cuda:0').eval()
                with torch.no_grad():
                    residual = ranker(extracted)
                if torch.count_nonzero(residual) or not torch.equal(
                        select(extracted['pcs_scores'], residual), original['scores'].argmax(-1)):
                    raise ValueError('Zero-initialized ranking is not identity')
                print('PASS: frozen PCS scores and zero-initialized rank selections match exactly')
                first_checked = True
            indices = batch['index'].numpy()
            values = {**extracted, **{k: batch[k] for k in ('labels', 'scores', 'direction')}}
            for key, value in values.items():
                array = value.detach().cpu().numpy()
                if not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite {split}/{key}')
                maps[key][indices] = array
            # Flush data BEFORE marking those rows complete; interruption can safely resume.
            for key in FIELDS:
                maps[key].flush()
            maps['done'][indices] = True
            maps['done'].flush()
        if not maps['done'].all():
            raise ValueError('Preparation incomplete')
        del maps
    write_new_json(root/'complete.json', dict(manifest=identity, scenes={s: len(json.loads(
        (root/f'{s}_records.json').read_text())) for s in source}))
    print('Compact feature cache complete:', root)


def fold_identity(args, dataset):
    record_path = Path(args.cache)/'records.json'
    records = json.loads(record_path.read_text())
    validate_records(records)
    folds = log_partitions(dataset.records, args.folds)
    if not 0 <= args.fold_index < args.folds:
        raise ValueError('Invalid fold index')
    train_indices, held_indices = np.flatnonzero(folds != args.fold_index), np.flatnonzero(folds == args.fold_index)
    identity = dict(schema='cost_rank_oof_teacher_v1', fold=args.fold_index, folds=args.folds,
                    candidate_manifest_sha256=sha256(Path(args.cache)/'manifest.json'),
                    candidate_records_sha256=sha256(record_path), provenance=dataset.provenance,
                    initialization='random_not_full_data_pcs', seed=args.seed+args.fold_index,
                    train_logs=sorted({dataset.records[i]['log_name'] for i in train_indices}),
                    held_logs=sorted({dataset.records[i]['log_name'] for i in held_indices}),
                    train_scenes=len(train_indices), held_scenes=len(held_indices),
                    settings=dataset.manifest['settings'], epochs=args.epochs, lr=args.lr,
                    partition_source_sha256=source_hashes()['data.py'])
    if set(identity['train_logs']) & set(identity['held_logs']):
        raise ValueError('OOF teacher log leakage')
    return identity, train_indices.tolist(), held_indices.tolist()


def predict_fold(args):
    from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
    from navsim.agents.diffusiondrive.pcs.model import PDMCSHead
    data = CandidateDataset(args.cache, 'train')
    root = Path(args.oof_root)
    result = json.loads((root/f'fold_{args.fold_index}.json').read_text())
    checkpoint = load_torch(result['checkpoint'])
    meta = checkpoint['fold_metadata']
    if (meta['candidate_manifest_sha256'] != sha256(Path(args.cache)/'manifest.json') or
            meta['candidate_records_sha256'] != sha256(Path(args.cache)/'records.json') or
            meta['initialization'] != 'random_not_full_data_pcs' or meta['fold'] != args.fold_index or
            meta['partition_source_sha256'] != source_hashes()['data.py']):
        raise ValueError('Teacher checkpoint/source mismatch')
    folds = log_partitions(data.records, meta['folds'])
    held = np.flatnonzero(folds == meta['fold']).tolist()
    expected_train = sorted({r['log_name'] for i, r in enumerate(data.records) if folds[i] != meta['fold']})
    expected_held = sorted({data.records[i]['log_name'] for i in held})
    if meta['train_logs'] != expected_train or meta['held_logs'] != expected_held:
        raise ValueError('Teacher heldout split mismatch')
    path = root/f'fold_{args.fold_index}_predictions.pt'
    checkpoint_hash = sha256(result['checkpoint'])
    if path.exists():
        old = load_torch(path)
        if (old['identity'] != meta or old['checkpoint_sha256'] != checkpoint_hash or
                old['indices'].tolist() != held or old['scores'].shape != (len(held), 67) or
                not torch.isfinite(old['scores']).all()):
            raise ValueError('Existing OOF prediction is incompatible/incomplete')
        print('Validated completed OOF predictions:', path)
        return
    model = PDMCSHead(**meta['settings']).eval().to('cuda:0')
    model.load_state_dict({k[5:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('head.')})
    indices, predictions = [], []
    with torch.no_grad():
        for batch in tqdm(loader(Subset(data, held), args.workers, batch_size=args.batch_size), desc='OOF predictions'):
            scores = model({k: v.to('cuda:0') for k, v in batch['context'].items()})['scores']
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite OOF prediction')
            indices.extend(batch['index'].tolist())
            predictions.append(scores.cpu())
    payload = dict(identity=meta, checkpoint_sha256=checkpoint_hash,
                   indices=torch.tensor(indices), scores=torch.cat(predictions))
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    print('Fold prediction complete:', path)


def assemble_oof(args):
    data = CompactDataset(args.features, 'train')
    folds = log_partitions(data.records, args.folds)
    root = Path(args.oof_root)
    merged = np.full((len(data), 67), np.nan, dtype=np.float32)
    seen = np.zeros(len(data), dtype=np.int64)
    teacher_hashes = []
    for fold in range(args.folds):
        path = root/f'fold_{fold}_predictions.pt'
        item = load_torch(path)
        meta = item['identity']
        expected_logs = sorted({r['log_name'] for i, r in enumerate(data.records) if folds[i] == fold})
        train_logs = sorted({r['log_name'] for i, r in enumerate(data.records) if folds[i] != fold})
        if (meta['fold'] != fold or meta['folds'] != args.folds or
                meta['candidate_records_sha256'] != data.manifest['candidate_records_sha256'] or
                meta['candidate_manifest_sha256'] != data.manifest['candidate_manifest_sha256'] or
                meta['held_logs'] != expected_logs or meta['train_logs'] != train_logs or
                meta['initialization'] != 'random_not_full_data_pcs' or
                meta['partition_source_sha256'] != source_hashes()['data.py']):
            raise ValueError('OOF prediction identity/leakage check failed')
        indices = item['indices'].numpy()
        values = item['scores'].numpy()
        if not np.array_equal(np.sort(indices), np.flatnonzero(folds == fold)):
            raise ValueError('Missing/duplicate/wrong-fold OOF indices')
        if values.shape != (len(indices), 67) or not np.isfinite(values).all():
            raise ValueError('Invalid OOF prediction shape/values')
        merged[indices] = values
        seen[indices] += 1
        teacher_hashes.append(dict(predictions_sha256=sha256(path), checkpoint_sha256=item['checkpoint_sha256']))
    if not (seen == 1).all() or not np.isfinite(merged).all():
        raise ValueError('OOF must cover every TRAIN scene exactly once')
    metadata = dict(schema='cost_rank_oof_v1', feature_manifest_sha256=sha256(Path(args.features)/'manifest.json'),
                    train_records_sha256=sha256(Path(args.features)/'train_records.json'), folds=args.folds,
                    teachers=teacher_hashes, source_sha256=source_hashes()['data.py'])
    write_new_json(root/'manifest.json', metadata)
    path = root/'oof_scores.npy'
    if path.exists():
        if not np.array_equal(np.load(path, mmap_mode='r'), merged):
            raise ValueError('Existing OOF scores differ')
    else:
        temporary = root/'oof_scores.tmp'
        with temporary.open('wb') as stream:
            np.save(stream, merged)
        temporary.replace(path)
    metadata['scores_sha256'] = sha256(path)
    write_new_json(root/'complete.json', metadata)
    scores = np.load(Path(args.features)/'train/scores.npy', mmap_mode='r')
    pcs = np.load(Path(args.features)/'train/pcs_scores.npy', mmap_mode='r')
    row = np.arange(len(data))
    actual_oof, actual_pcs = scores[row, merged.argmax(-1)], scores[row, pcs.argmax(-1)]
    summary = dict(scenes=len(data), folds=args.folds, oof_pdm=float(actual_oof.mean()),
                   original_pcs_train_pdm=float(actual_pcs.mean()),
                   oof_zero=int((actual_oof == 0).sum()), original_pcs_zero=int((actual_pcs == 0).sum()),
                   note='OOF applies to mining teachers only; generator and original PCS features are not cross-fitted.')
    write_new_json(root/'summary.json', summary)
    print(json.dumps(summary, indent=2))


def validate_oof(features, oof_root):
    root = Path(oof_root)
    m = json.loads((root/'complete.json').read_text())
    if (m['schema'] != 'cost_rank_oof_v1' or
            m['feature_manifest_sha256'] != sha256(Path(features)/'manifest.json') or
            m['train_records_sha256'] != sha256(Path(features)/'train_records.json') or
            m['source_sha256'] != source_hashes()['data.py'] or
            m['scores_sha256'] != sha256(root/'oof_scores.npy')):
        raise ValueError('OOF cache incomplete/stale/misaligned')
    return m
