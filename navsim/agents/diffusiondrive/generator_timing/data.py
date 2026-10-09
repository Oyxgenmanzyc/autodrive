"""Immutable, mode-labelled teachers from the original K67 navtrain cache."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from navsim.agents.diffusiondrive.pcs.common import load_torch, sha256, write_new_json
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset, entry_path, metric_paths
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from .geometry import FAMILIES, STRENGTHS, select_teachers, teacher_bank

BLOCK = 32


def identity(candidate_root, metric_root, baseline, anchor):
    root = Path(__file__).parent
    return dict(schema='generator_timing_teachers_v1',
                candidate_manifest_sha256=sha256(Path(candidate_root)/'manifest.json'),
                candidate_records_sha256=sha256(Path(candidate_root)/'records.json'),
                baseline_sha256=sha256(baseline), anchor_sha256=sha256(anchor),
                metric_root=str(Path(metric_root).resolve()),
                families=list(FAMILIES), strengths=list(STRENGTHS), block=BLOCK,
                source_sha256={p.name: sha256(p) for p in
                               (root/'geometry.py', root/'model.py', root/'data.py')})


def init_cache(candidate_root, metric_root, baseline, anchor, output):
    manifest = identity(candidate_root, metric_root, baseline, anchor)
    root = Path(output)
    for split in ('train', 'val'):
        data = CandidateDataset(candidate_root, split)
        write_new_json(root/f'{split}_records.json', data.records)
        (root/split).mkdir(parents=True, exist_ok=True)
    write_new_json(root/'manifest.json', manifest)
    print('Timing teacher cache initialized:', root)


def check_cache(candidate_root, metric_root, baseline, anchor, output, split):
    root = Path(output)
    expected = identity(candidate_root, metric_root, baseline, anchor)
    if json.loads((root/'manifest.json').read_text()) != expected:
        raise ValueError('Timing teacher provenance mismatch')
    data = CandidateDataset(candidate_root, split)
    if json.loads((root/f'{split}_records.json').read_text()) != data.records:
        raise ValueError('Timing teacher record mismatch')
    return data


def make_scene(entry, metric_path):
    original = entry['context']['proposals'].numpy().astype(np.float32)
    bank = teacher_bank(original)
    candidates = np.concatenate((original, bank.reshape(-1, 8, 3)))
    labels, scores, direction = score_candidates(metric_path, candidates)
    targets, valid, weight = select_teachers(original, bank, labels, scores, direction)
    return dict(targets=torch.from_numpy(targets).half(),
                valid=torch.from_numpy(valid), weight=torch.from_numpy(weight).half(),
                original_scores=torch.from_numpy(scores[:67]),
                bank_scores=torch.from_numpy(scores[67:].reshape(67, 2, len(STRENGTHS))),
                safe_count=int(valid.sum()))


def prepare_shard(candidate_root, metric_root, baseline, anchor, output,
                  split, shard_index, num_shards):
    if not 0 <= shard_index < num_shards:
        raise ValueError('Invalid shard index')
    data = check_cache(candidate_root, metric_root, baseline, anchor, output, split)
    paths = metric_paths(metric_root)
    missing = [r['token'] for r in data.records if r['token'] not in paths]
    if missing:
        raise ValueError(f'Missing metric caches: {len(missing)}; first={missing[:3]}')
    root = Path(output)/split
    for start in range(0, len(data), BLOCK):
        if (start//BLOCK) % num_shards != shard_index:
            continue
        target = root/f'block_{start//BLOCK:05d}.pt'
        if target.is_file():
            continue
        rows = []
        for record in data.records[start:start+BLOCK]:
            entry = load_torch(entry_path(candidate_root, record))
            if entry['token'] != record['token'] or entry['provenance'] != data.provenance:
                raise ValueError('K67 scene identity mismatch')
            rows.append(make_scene(entry, paths[record['token']]))
        piece = {'tokens': [r['token'] for r in data.records[start:start+BLOCK]]}
        for key in ('targets', 'valid', 'weight', 'original_scores', 'bank_scores'):
            piece[key] = torch.stack([row[key] for row in rows])
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(piece, temp)
        os.replace(temp, target)
        print(f'{split} shard {shard_index}: {start+len(rows)}/{len(data)}, '
              f'active scenes={sum(row["safe_count"] > 0 for row in rows)}', flush=True)


def complete_cache(candidate_root, metric_root, baseline, anchor, output):
    root = Path(output)
    counts = {}
    active = {}
    eligible_modes = {}
    for split in ('train', 'val'):
        data = check_cache(candidate_root, metric_root, baseline, anchor, output, split)
        indices = []
        families = np.zeros(2, dtype=np.int64)
        for start in range(0, len(data), BLOCK):
            path = root/split/f'block_{start//BLOCK:05d}.pt'
            if not path.is_file():
                raise ValueError(f'Missing teacher block: {path}')
            piece = load_torch(path)
            if piece['tokens'] != [r['token'] for r in data.records[start:start+BLOCK]]:
                raise ValueError(f'Teacher token order mismatch: {path}')
            if piece['targets'].shape != (len(piece['tokens']), 67, 2, 8, 3):
                raise ValueError(f'Teacher shape mismatch: {path}')
            indices.extend((start+i for i, row in enumerate(piece['valid']) if row.any()))
            families += piece['valid'].sum((0, 1)).numpy()
        counts[split] = len(data)
        active[split] = len(indices)
        eligible_modes[split] = dict(zip(FAMILIES, families.tolist()))
        np.save(root/f'{split}_active.npy', np.asarray(indices, dtype=np.int64))
    write_new_json(root/'complete.json', dict(manifest_sha256=sha256(root/'manifest.json'),
                                              scenes=counts, active_scenes=active,
                                              eligible_modes=eligible_modes))
    print('Complete generator timing teachers:', counts, active, eligible_modes)


class TimingTeacherDataset(Dataset):
    def __init__(self, candidate_root, teacher_root, split, active_only=False):
        self.candidates = CandidateDataset(candidate_root, split)
        self.root, self.split = Path(teacher_root), split
        complete = json.loads((self.root/'complete.json').read_text())
        if complete['manifest_sha256'] != sha256(self.root/'manifest.json'):
            raise ValueError('Timing teacher manifest changed')
        if complete['scenes'][split] != len(self.candidates):
            raise ValueError('Incomplete timing teacher cache')
        self.indices = (np.load(self.root/f'{split}_active.npy').tolist() if active_only
                        else list(range(len(self.candidates))))
        self._block_index, self._block = None, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        block = index // BLOCK
        if self._block_index != block:
            self._block = load_torch(self.root/self.split/f'block_{block:05d}.pt')
            self._block_index = block
        record = self.candidates.records[index]
        entry = load_torch(entry_path(self.candidates.root, record))
        if entry['token'] != record['token'] or self._block['tokens'][index % BLOCK] != record['token']:
            raise ValueError('Candidate/teacher scene mismatch')
        context = {key: entry['context'][key] for key in ('bev', 'agents', 'ego')}
        result = dict(context=context, token=record['token'], log_name=record['log_name'],
                      original=entry['context']['proposals'], index=index)
        result.update({key: self._block[key][index % BLOCK] for key in
                       ('targets', 'valid', 'weight', 'original_scores', 'bank_scores')})
        return result
