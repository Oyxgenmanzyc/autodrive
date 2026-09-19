"""Read-only compact memmaps, deterministic log folds, and immutable provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from navsim.agents.diffusiondrive.pcs.common import sha256

FIELDS = {
    'features': ('float16', (67, 512)), 'subscores': ('float32', (67, 5)),
    'pcs_scores': ('float32', (67,)), 'proposals': ('float32', (67, 8, 3)),
    'base_logits': ('float32', (67,)), 'labels': ('float32', (67, 5)),
    'scores': ('float32', (67,)), 'direction': ('float32', (67,)),
}


def source_hashes():
    root = Path(__file__).parent
    return {name: sha256(root/name) for name in ('model.py', 'data.py', 'metrics.py')}


def log_partitions(records, folds=3, seed=31056):
    """Whole recording logs stay together; deterministic across record order."""
    if folds < 2:
        raise ValueError('At least two folds required')
    logs = sorted({r['log_name'] for r in records},
                  key=lambda log: hashlib.sha256(f'{seed}:{log}'.encode()).hexdigest())
    if len(logs) < folds:
        raise ValueError('Insufficient distinct logs')
    assignment = {log: i % folds for i, log in enumerate(logs)}
    return np.asarray([assignment[r['log_name']] for r in records], dtype=np.int64)


def validate_records(records):
    if len({r['token'] for r in records}) != len(records):
        raise ValueError('Duplicate candidate tokens')
    train = {r['log_name'] for r in records if r['split'] == 'train'}
    val = {r['log_name'] for r in records if r['split'] == 'val'}
    if train & val or any(r['split'] not in ('train', 'val') for r in records):
        raise ValueError('Train/val log overlap or unexpected split')


class CompactDataset(Dataset):
    def __init__(self, root, split, oof=None, smoke=False):
        self.root, self.split = Path(root), split
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['schema'] != 'cost_rank_features_v1':
            raise ValueError('Unexpected feature cache schema')
        if self.manifest['limit'] and not smoke:
            raise ValueError('Pilot features are for smoke only')
        if self.manifest['feature_source_sha256'] != source_hashes()['model.py']:
            raise ValueError('Frozen-feature/model implementation changed')
        self.records = json.loads((self.root/f'{split}_records.json').read_text())
        self.maps = None
        self.oof = str(oof) if oof else None
        self.calibration = log_partitions(self.records, 2) == 0 if split == 'val' else None
        done = np.load(self.root/split/'done.npy', mmap_mode='r')
        if done.shape != (len(self.records),) or not done.all():
            raise ValueError(f'Incomplete compact {split} cache')
        if not (self.root/'complete.json').is_file():
            raise ValueError('Preparation has no successful completion record')
        for key, (dtype, shape) in FIELDS.items():
            array = np.load(self.root/split/f'{key}.npy', mmap_mode='r')
            if array.shape != (len(self.records), *shape) or array.dtype != np.dtype(dtype):
                raise ValueError(f'Invalid compact field: {split}/{key}')

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['maps'] = None
        return state

    def __getitem__(self, index):
        if self.maps is None:
            self.maps = {k: np.load(self.root/self.split/f'{k}.npy', mmap_mode='r') for k in FIELDS}
            if self.oof:
                self.maps['oof_scores'] = np.load(self.oof, mmap_mode='r')
        item = {k: torch.from_numpy(np.array(v[index], copy=True)) for k, v in self.maps.items()}
        item['index'] = index
        if self.calibration is not None:
            item['calibration'] = bool(self.calibration[index])
        return item
