"""Join immutable K67 features with newly scored timing-mode candidates."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from navsim.agents.diffusiondrive.cost_rank.data import CompactDataset
from navsim.agents.diffusiondrive.pcs.common import load_torch, sha256
from .geometry import MODE_NAMES, TOP_SPATIAL
from . import geometry, model

BLOCK = 128
VARIANTS = TOP_SPATIAL*len(MODE_NAMES)


def implementation_hashes():
    return {name: sha256(path) for name, path in
            (('geometry', geometry.__file__), ('model', model.__file__), ('data', __file__))}


def inputs_from_source(source, modes, rank_scores, trajectories, variant=None):
    """Exactly the same deployable input assembly in train and navtest."""
    modes = torch.as_tensor(modes, dtype=torch.long)
    trajectories = torch.as_tensor(np.asarray(trajectories).copy(), dtype=torch.float32)
    if modes.shape != (TOP_SPATIAL,) or trajectories.shape != (VARIANTS, 8, 3):
        raise ValueError('Invalid timing candidate dimensions')
    probabilities = source['base_logits'].float().softmax(-1)
    repeat = lambda key: source[key][modes].float().repeat_interleave(len(MODE_NAMES), 0)
    features = repeat('features') if variant is None else variant['features'].float()
    subscores = repeat('subscores') if variant is None else variant['subscores'].float()
    pcs_scores = repeat('pcs_scores') if variant is None else variant['pcs_scores'].float()
    if features.shape != (VARIANTS, 512) or subscores.shape != (VARIANTS, 5) or \
            pcs_scores.shape != (VARIANTS,):
        raise ValueError('Invalid frozen variant PCS features')
    return dict(features=features, subscores=subscores,
                pcs_scores=pcs_scores, original=repeat('proposals'),
                base_probability=probabilities[modes].repeat_interleave(len(MODE_NAMES)),
                rank_scores=torch.as_tensor(np.asarray(rank_scores).copy(), dtype=torch.float32)[modes]
                .repeat_interleave(len(MODE_NAMES)), trajectories=trajectories)


class TimingModeDataset(Dataset):
    def __init__(self, features, cache, split, smoke=False):
        self.compact = CompactDataset(features, split, smoke=smoke)
        self.root, self.split = Path(cache), split
        manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['schema'] != 'timing_modes_v1' or manifest['modes'] != list(MODE_NAMES):
            raise ValueError('Timing-mode cache schema mismatch')
        if manifest['implementation_sha256'] != implementation_hashes():
            raise ValueError('Timing-mode source implementation changed')
        if manifest['features'] != self.compact.manifest:
            raise ValueError('Timing-mode cache does not match frozen K67 features')
        if manifest['limit'] and not smoke:
            raise ValueError('Pilot timing labels cannot train a formal model')
        records = json.loads((self.root/f'{split}_records.json').read_text(encoding='utf-8'))
        if records != self.compact.records:
            raise ValueError('Timing-mode cache record order mismatch')
        complete = json.loads((self.root/'complete.json').read_text(encoding='utf-8'))
        if complete['counts'][split] != len(records):
            raise ValueError('Timing-mode cache incomplete')
        pieces = []
        for start in range(0, len(records), BLOCK):
            path = self.root/split/f'block_{start//BLOCK:05d}.pt'
            if not path.is_file():
                raise ValueError(f'Missing timing-mode block: {path}')
            piece = load_torch(path)
            if piece['tokens'] != [r['token'] for r in records[start:start+BLOCK]]:
                raise ValueError(f'Timing-mode block token mismatch: {path}')
            pieces.append(piece)
        self.cached = {key: torch.cat([p[key] for p in pieces]) for key in
                       ('modes', 'rank_scores', 'trajectories', 'labels', 'scores', 'direction',
                        'variant_features', 'variant_subscores', 'variant_pcs_scores')}
        if len(self.cached['scores']) != len(self.compact):
            raise ValueError('Timing-mode cache count mismatch')

    def __len__(self):
        return len(self.compact)

    def __getitem__(self, index):
        source = self.compact[index]
        item = inputs_from_source(source, self.cached['modes'][index],
                                  self.cached['rank_scores'][index],
                                  self.cached['trajectories'][index],
                                  dict(features=self.cached['variant_features'][index],
                                       subscores=self.cached['variant_subscores'][index],
                                       pcs_scores=self.cached['variant_pcs_scores'][index]))
        item.update(labels=self.cached['labels'][index].float(),
                    scores=self.cached['scores'][index].float(),
                    direction=self.cached['direction'][index].float(),
                    calibration=bool(self.compact.calibration[index]) if self.split == 'val' else False,
                    index=index)
        return item
