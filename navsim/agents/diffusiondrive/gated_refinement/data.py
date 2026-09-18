"""Reuse immutable PTR v4 scoring caches; add labels and train-only pairs."""
import csv
import json
from pathlib import Path
import torch
from navsim.agents.diffusiondrive.pcs.common import load_torch, sha256, write_new_json, CONFIG_ROOT
from navsim.agents.diffusiondrive.post_selection.data import RefinementDataset, read_blocks, source_identity
from navsim.agents.diffusiondrive.post_selection.geometry import ACTIONS
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset
from .labels import build_labels, match_negatives, calibration_mask, REASONS


def code_identity():
    root = Path(__file__).parent
    result = {p.name: sha256(p) for p in sorted(root.glob('*.py'))}
    result['runner'] = sha256(root.parents[2]/'planning/script/run_gated_refinement.py')
    return result


def compatible_sources(cached):
    """Permit runtime-only fixes without invalidating immutable supervision.

    labels.py defines the offline targets and model.py defines checkpoint tensor
    semantics. data/training/runner fixes do not change already materialized
    labels, whose records and source block hashes are checked independently.
    """
    current = code_identity()
    return all(cached.get(name) == current[name] for name in ('labels.py', 'model.py'))


def validate_edit_schema(manifest):
    # Training-head/runner changes do not invalidate already scored trajectories.
    # Exact brake geometry must still match; selector SHA is checked at evaluation,
    # and every saved selection is matched bit-exactly to its source mode on load.
    if manifest['schema'] != 'post_selection_brake_bank_v1':
        raise ValueError('Unsupported edit cache schema')
    if manifest['sources']['geometry.py'] != source_identity()['geometry.py']:
        raise ValueError('Scored brake action geometry differs')
    if manifest['scoring_sha256'] != sha256(CONFIG_ROOT/'pdm_scoring/default_scoring_parameters.yaml'):
        raise ValueError('PDM scoring configuration differs')


def prepare(edit_root, output, min_gain=.02):
    root, out = Path(edit_root), Path(output)
    manifest = json.loads((root/'manifest.json').read_text())
    validate_edit_schema(manifest)
    if manifest['limit']:
        raise ValueError('Pilot had zero positive train cases; use FULL edit cache for gate preparation')
    records = {s: json.loads((root/f'{s}_records.json').read_text()) for s in ('train', 'val')}
    train_logs = {r['log_name'] for r in records['train']}
    val_logs = {r['log_name'] for r in records['val']}
    if train_logs & val_logs:
        raise ValueError('Train/validation logs overlap')
    identity = dict(schema='gated_refinement_v1', sources=code_identity(), edit_manifest=manifest,
                    min_gain=min_gain, pair_features='selected_path_kinematics_train_standardization',
                    records_sha256={s: sha256(root/f'{s}_records.json') for s in records})
    write_new_json(out/'manifest.json', identity)
    summary = {}
    for split in ('train', 'val'):
        data = read_blocks(root, split, records[split])
        # Track content too: stale labels cannot survive an edit-cache overwrite.
        blocks = {p.name: sha256(p) for p in sorted((root/split).glob('block_*.pt'))}
        target = build_labels(data['labels'], data['scores'], data['direction'], min_gain)
        if split == 'train':
            target['pairs'] = match_negatives(data['selected'], target['need'], records[split])
        else:
            target['calibration'] = calibration_mask(records[split])
        artifact = dict(target=target, blocks=blocks, tokens=[r['token'] for r in records[split]])
        path = out/f'{split}.pt'
        if path.exists():
            old = load_torch(path)
            if old['blocks'] != blocks or old['tokens'] != artifact['tokens']:
                raise ValueError('Existing gate cache has different source content; use a new directory')
            if old['target'].keys() != target.keys() or any(not torch.equal(old['target'][k], v) for k, v in target.items()):
                raise ValueError('Existing gate targets differ from reproducible labels')
        else:
            temporary = path.with_suffix('.tmp')
            torch.save(artifact, temporary)
            temporary.replace(path)
        export = out/f'{split}_labels.csv'
        if not export.exists():
            with export.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['token', 'log_name', 'need_modify', 'unresolved_by_bank', 'oracle_gain',
                                 'teacher_action', 'extra_brake_onset_s', 'extra_decel_mps2', 'ramp_s', *REASONS])
                for i, record in enumerate(records[split]):
                    action = int(target['teacher'][i])
                    writer.writerow([record['token'], record['log_name'], int(target['need'][i]),
                                     int(target['unresolved'][i]), float(target['regret'][i]), action,
                                     *ACTIONS[action], *target['reasons'][i].int().tolist()])
        summary[split] = dict(scenes=len(records[split]), positives=int(target['need'].sum()),
                              unresolved=int(target['unresolved'].sum()),
                              reason_counts=dict(zip(REASONS, target['reasons'].sum(0).int().tolist())))
        if split == 'val':
            for name, mask in [('calibration', target['calibration']), ('audit', ~target['calibration'])]:
                summary[split][name] = dict(scenes=int(mask.sum()), positives=int(target['need'][mask].sum()))
                if not target['need'][mask].any() or target['need'][mask].all():
                    raise ValueError('Validation partition needs both classes')
    write_new_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


class GatedDataset(RefinementDataset):
    def __init__(self, candidate_root, edit_root, gate_root, split):
        self.source = CandidateDataset(candidate_root, split, allow_partial=False)
        self.manifest = json.loads((Path(edit_root)/'manifest.json').read_text())
        validate_edit_schema(self.manifest)
        if self.manifest['limit'] or self.manifest['provenance'] != self.source.provenance:
            raise ValueError('Full matching generator cache required')
        if self.manifest['candidate_manifest_sha256'] != sha256(self.source.root/'manifest.json'):
            raise ValueError('Candidate manifest changed')
        self.records = self.source.records
        if json.loads((Path(edit_root)/f'{split}_records.json').read_text()) != self.records:
            raise ValueError('Edit/source records mismatch')
        self.data = read_blocks(edit_root, split, self.records)
        self.teacher = torch.zeros(len(self.records), dtype=torch.long)
        root = Path(gate_root)
        summary = json.loads((root/'summary.json').read_text())
        if summary[split]['scenes'] != len(self.records):
            raise ValueError('Incomplete gate preparation')
        self.gate_manifest = json.loads((root/'manifest.json').read_text())
        m = self.gate_manifest
        if m['schema'] != 'gated_refinement_v1' or not compatible_sources(m['sources']):
            raise ValueError('Gate code/cache mismatch')
        if m['edit_manifest'] != self.manifest:
            raise ValueError('Gate/edit manifest mismatch')
        if m['records_sha256'][split] != sha256(Path(edit_root)/f'{split}_records.json'):
            raise ValueError('Gate record hash mismatch')
        artifact = load_torch(root/f'{split}.pt')
        if artifact['tokens'] != [r['token'] for r in self.records]:
            raise ValueError('Gate token alignment mismatch')
        blocks = {p.name: sha256(p) for p in sorted((Path(edit_root)/split).glob('block_*.pt'))}
        if artifact['blocks'] != blocks:
            raise ValueError('Edit block content changed; regenerate labels in a new directory')
        self.targets = artifact['target']

    def __getitem__(self, index):
        item = super().__getitem__(index)
        for key, value in self.targets.items():
            if key != 'pairs':
                item[key] = value[index]
        return item
