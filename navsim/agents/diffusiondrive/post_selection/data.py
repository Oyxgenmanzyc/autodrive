"""Compact, resumable official-PDM labels for edits of frozen TRV selections."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from navsim.agents.diffusiondrive.pcs.common import load_torch, sha256, CONFIG_ROOT
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
from .geometry import brake_bank, safe_oracle


def source_identity():
    root = Path(__file__).parent
    result = {name: sha256(root/name) for name in ('geometry.py', 'model.py', 'data.py', 'training.py')}
    result['run_post_selection.py'] = sha256(root.parents[2]/'planning/script/run_post_selection.py')
    return result


def cache_identity(candidate, veto, metric_root, limit):
    return dict(schema='post_selection_brake_bank_v1', sources=source_identity(),
                provenance=candidate.provenance, veto_sha256=sha256(veto),
                candidate_manifest_sha256=sha256(candidate.root/'manifest.json'),
                metric_root=str(Path(metric_root).resolve()), limit=limit,
                scoring_sha256=sha256(CONFIG_ROOT/'pdm_scoring/default_scoring_parameters.yaml'))


def read_blocks(root, split, records):
    pieces = []
    for start in range(0, len(records), 128):
        path = Path(root)/split/f'block_{start//128:05d}.pt'
        item = load_torch(path)
        expected = [r['token'] for r in records[start:start+128]]
        if item['tokens'] != expected:
            raise ValueError(f'Token alignment mismatch: {path}')
        for key in ('selected', 'labels', 'scores', 'direction'):
            if len(item[key]) != len(expected) or not torch.isfinite(item[key]).all():
                raise ValueError(f'Invalid {key} in {path}')
        shapes = {'selected': (len(expected), 8, 3), 'labels': (len(expected), 25, 5),
                  'scores': (len(expected), 25), 'direction': (len(expected), 25),
                  'mode': (len(expected),)}
        for key, shape in shapes.items():
            if tuple(item[key].shape) != shape:
                raise ValueError(f'Wrong shape for {key} in {path}')
        pieces.append(item)
    return {key: torch.cat([p[key] for p in pieces]) for key in
            ('selected', 'labels', 'scores', 'direction', 'mode')}


class RefinementDataset(Dataset):
    def __init__(self, candidate_root, edit_root, split, smoke=False):
        self.source = CandidateDataset(candidate_root, split, allow_partial=smoke)
        self.manifest = json.loads((Path(edit_root)/'manifest.json').read_text())
        m = self.manifest
        if m['schema'] != 'post_selection_brake_bank_v1' or m['sources'] != source_identity():
            raise ValueError('Refinement source differs; use the matching release/cache')
        if m['provenance'] != self.source.provenance:
            raise ValueError('Frozen generator provenance differs')
        if m['candidate_manifest_sha256'] != sha256(self.source.root/'manifest.json'):
            raise ValueError('Candidate manifest differs')
        if m['limit'] and not smoke:
            raise ValueError('Pilot edit cache is not allowed for full training')
        self.records = self.source.records[:m['limit']] if m['limit'] else self.source.records
        record_path = Path(edit_root)/f'{split}_records.json'
        if json.loads(record_path.read_text()) != self.records:
            raise ValueError('Frozen source records changed')
        self.data = read_blocks(edit_root, split, self.records)
        self.teacher = torch.from_numpy(safe_oracle(
            self.data['labels'].numpy(), self.data['scores'].numpy(), self.data['direction'].numpy())).long()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        context = self.source[index]['context']
        selected = self.data['selected'][index]
        if not torch.equal(context['proposals'][int(self.data['mode'][index])], selected):
            raise ValueError('Saved selected trajectory no longer matches source')
        # Original 67 trajectories/logits are not inputs to the new refiner.
        return {'context': {k: context[k] for k in ('bev', 'agents', 'ego')},
                'variants': torch.from_numpy(brake_bank(selected.numpy())),
                **{k: self.data[k][index] for k in ('labels', 'scores', 'direction')},
                'teacher': self.teacher[index], 'index': index}
