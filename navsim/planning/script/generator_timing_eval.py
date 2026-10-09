"""Locked generator-mode evaluation with matched 201-candidate controls."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from navsim.agents.diffusiondrive.cost_rank.data import log_partitions
from navsim.agents.diffusiondrive.cost_rank.model import FrozenPCS
from navsim.agents.diffusiondrive.generator_timing import data as teachers
from navsim.agents.diffusiondrive.generator_timing.geometry import teacher_bank
from navsim.agents.diffusiondrive.pcs.common import (
    build_generator, load_scorer, load_torch, provenance,
    sha256, token_seed, write_new_json,
)
from navsim.agents.diffusiondrive.pcs.data import CandidateDataset, entry_path, metric_paths
from navsim.agents.diffusiondrive.pcs.scoring import score_candidates
from navsim.planning.script.run_cost_rank import load_ranker
from navsim.planning.script.run_generator_timing import load_model

MARGINS = (0., .01, .02, .05, .1)
SAFETY_COLUMNS = (0, 1, 3, 4)


def _rank(context, frozen, ranker, alpha):
    features = frozen(context)
    residual = ranker(features)
    return (features['pcs_scores'] + alpha*residual).squeeze(0)


def _candidate_context(source, proposals, logits, device):
    result = {key: source[key].unsqueeze(0).to(device) for key in ('bev', 'agents', 'ego')}
    result['proposals'] = proposals.unsqueeze(0).to(device)
    result['base_logits'] = logits.unsqueeze(0).to(device)
    return result


def _generate(model, context, token, device):
    devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(token_seed(token, 310511))
        return model(context).squeeze(0).cpu().float()


def _random_control(model, context, token):
    trajectories, logits = [], []
    devices = [torch.cuda.current_device()]
    for seed in (310512, 310513):
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(token_seed(token, seed))
            generated = model.head.forward_test(context['ego'], context['agents'],
                                                context['bev'], context['bev'].shape[-2:], None, None)
        trajectories.append(generated['proposal_trajectory'].squeeze(0).cpu().float())
        logits.append(generated['proposal_logits'].squeeze(0).cpu().float())
    return torch.cat(trajectories), torch.cat(logits)


def _inputs(source, model, frozen, ranker, alpha, token, device, controls):
    original, base_logits = source['proposals'].float(), source['base_logits'].float()
    original_context = _candidate_context(source, original, base_logits, device)
    with torch.no_grad():
        original_scores = _rank(original_context, frozen, ranker, alpha)
    old = int(original_scores.argmax())
    new = _generate(model, original_context, token, device)
    if new.shape != (134, 8, 3):
        raise ValueError('Expected 134 decoder-generated timing candidates')
    banks = {'generated': (new, base_logits.repeat_interleave(2))}
    if controls:
        fixed = teacher_bank(original.numpy())[:, :, 0].reshape(134, 8, 3)
        banks['fixed_warp'] = (torch.from_numpy(fixed), base_logits.repeat_interleave(2))
        banks['extra_samples'] = _random_control(model, original_context, token)
    output = {}
    with torch.no_grad():
        for family, (additional, additional_logits) in banks.items():
            proposals = torch.cat((original, additional))
            logits = torch.cat((base_logits, additional_logits))
            context = _candidate_context(source, proposals, logits, device)
            predicted = _rank(context, frozen, ranker, alpha).cpu().numpy()
            output[family] = dict(proposals=proposals.numpy(), predicted=predicted,
                                  best=int(predicted.argmax()), original=old)
    distance = torch.linalg.vector_norm(new[..., :2] - original.repeat_interleave(2, 0)[..., :2], dim=-1)
    return output, int((distance.max(-1).values >= .2).sum())


def _aggregate(rows, family, margin):
    selected = np.asarray([r[family]['chosen'][margin] for r in rows])
    original = np.asarray([r['original_score'] for r in rows])
    delta = selected-original
    selected_metrics = np.asarray([r[family]['metrics'][margin] for r in rows])
    original_metrics = np.asarray([r['original_metrics'] for r in rows])
    safety = bool((selected_metrics[:, SAFETY_COLUMNS].mean(0)+1e-8 >=
                   original_metrics[:, SAFETY_COLUMNS].mean(0)).all())
    direction = np.asarray([r[family]['direction'][margin] for r in rows])
    old_direction = np.asarray([r['original_direction'] for r in rows])
    safety &= bool(direction.mean()+1e-8 >= old_direction.mean())
    return dict(scenes=len(rows), pdm=float(selected.mean()),
                baseline_pdm=float(original.mean()), gain_points=float(delta.mean()*100),
                changed=int(sum(r[family]['indices'][margin] >= 67 for r in rows)),
                beneficial=int((delta > 1e-6).sum()), harmful=int((delta < -1e-6).sum()),
                severe_losses=int((delta <= -.2).sum()),
                new_nc_failure=int(((original_metrics[:, 0] == 1) & (selected_metrics[:, 0] < 1)).sum()),
                new_dac_failure=int(((original_metrics[:, 1] == 1) & (selected_metrics[:, 1] < 1)).sum()),
                new_ttc_failure=int(((original_metrics[:, 3] == 1) & (selected_metrics[:, 3] < 1)).sum()),
                safety_pass=safety)


def _finish(record, families, futures, divergence):
    row = dict(token=record['token'], log_name=record['log_name'], divergence=divergence)
    for family, info in families.items():
        labels, scores, direction = futures[family].result()
        baseline_index = info['original']
        if family == 'generated':
            row['original_score'] = float(scores[baseline_index])
            row['original_metrics'] = labels[baseline_index].tolist()
            row['original_direction'] = float(direction[baseline_index])
            row['original_oracle'] = float(scores[:67].max())
        else:
            if abs(float(scores[baseline_index])-row['original_score']) > 1e-5:
                raise ValueError(f'Original PDM changed across matched controls: {record["token"]}')
        best = info['best']
        result = dict(oracle=float(scores.max()),
                      chosen={}, metrics={}, direction={}, indices={})
        for margin in MARGINS:
            index = best if best >= 67 and info['predicted'][best] > \
                info['predicted'][baseline_index] + margin else baseline_index
            result['indices'][margin] = index
            result['chosen'][margin] = float(scores[index])
            result['metrics'][margin] = labels[index].tolist()
            result['direction'][margin] = float(direction[index])
        row[family] = result
    return row


def _calibration(rows):
    calibration = [r for r in rows if r['calibration']]
    audit = [r for r in rows if not r['calibration']]
    def policy(family):
        grid = {m: _aggregate(calibration, family, m) for m in MARGINS}
        eligible = [m for m in MARGINS if grid[m]['safety_pass'] and grid[m]['gain_points'] > 0]
        selected = max(eligible, key=lambda m: (grid[m]['gain_points'], -m)) if eligible else None
        return grid, selected

    grid, selected = policy('generated')
    controls = {}
    for family in ('generated', 'fixed_warp', 'extra_samples'):
        family_grid, family_margin = policy(family)
        controls[family] = dict(calibration_oracle_points=float(np.mean(
            [r[family]['oracle']-r['original_oracle'] for r in calibration])*100),
            audit_oracle_points=float(np.mean(
                [r[family]['oracle']-r['original_oracle'] for r in audit])*100),
            selected_margin=family_margin,
            audit_locked=(_aggregate(audit, family, family_margin)
                          if family_margin is not None else None),
            audit_raw=_aggregate(audit, family, 0.))
    locked = _aggregate(audit, 'generated', selected) if selected is not None else None
    control_gain = max((controls[name]['audit_locked']['gain_points']
                        if controls[name]['audit_locked'] is not None else 0.)
                       for name in ('fixed_warp', 'extra_samples'))
    pass_test = bool(selected is not None and locked['gain_points'] > 0 and
                     locked['safety_pass'] and locked['severe_losses'] == 0 and
                     locked['gain_points'] > control_gain + .05 and
                     controls['generated']['audit_oracle_points'] >
                     max(controls['fixed_warp']['audit_oracle_points'],
                         controls['extra_samples']['audit_oracle_points']) + .1)
    return dict(schema='generator_timing_decision_v1', selected_margin=selected,
                calibration_grid=grid, audit_locked=locked, controls=controls,
                mode_divergence_fraction=float(sum(r['divergence'] for r in rows)/(134*len(rows))),
                pass_for_navtest=pass_test,
                note='Only calibration selects the policy; independent audit and equal-count controls decide continuation.')


def evaluate(args):
    if args.max_scenes < 0 or args.score_workers < 1:
        raise ValueError('Invalid evaluation settings')
    trained = load_torch(args.generator_checkpoint)
    if args.max_scenes == 0 and trained['epoch'] != 19:
        raise ValueError('Formal validation/navtest requires the locked 20-epoch adapter')
    device = 'cuda:0'
    teacher_manifest = json.loads((Path(args.teachers)/'manifest.json').read_text())
    teachers.check_cache(args.candidates, teacher_manifest['metric_root'],
                         args.baseline, args.anchor, args.teachers, 'val')
    model = load_model(args, device, args.generator_checkpoint).eval()
    frozen_scorer, _ = load_scorer(args.scorer, device,
                                   provenance(args.baseline, args.anchor, 0))
    frozen = FrozenPCS(frozen_scorer).eval()
    ranker, rank_meta = load_ranker(args.ranker, device)
    if rank_meta['identity']['features']['provenance'] != provenance(args.baseline, args.anchor, 0):
        raise ValueError('Locked Ranker differs from original generator')
    alpha = rank_meta['policy']['alpha']
    if rank_meta['identity']['features']['candidate_manifest_sha256'] != \
            sha256(Path(args.candidates)/'manifest.json'):
        raise ValueError('Ranker and candidate cache mismatch')
    if not rank_meta['policy']['enabled']:
        raise ValueError('Original locked Ranker must be enabled')
    controls = args.command == 'validate'
    identity = dict(generator_checkpoint_sha256=sha256(args.generator_checkpoint),
                    scorer_sha256=sha256(args.scorer), ranker_sha256=sha256(args.ranker),
                    candidate_manifest_sha256=sha256(Path(args.candidates)/'manifest.json'),
                    evaluation_source_sha256=sha256(__file__))
    decision = None
    if not controls:
        if not args.decision:
            raise ValueError('Navtest needs the locked validation decision')
        decision = json.loads(Path(args.decision).read_text())
        if not decision['pass_for_navtest'] or decision['selected_margin'] is None:
            raise ValueError('Independent validation did not authorize navtest')
        if decision['identity'] != identity:
            raise ValueError('Navtest inputs differ from locked validation')
    if controls:
        source = CandidateDataset(args.candidates, 'val')
        records = source.records
        partitions = log_partitions(records, 2) == 0
    else:
        from navsim.agents.diffusiondrive.pcs.data import FeatureSource
        generator, config = build_generator(args.baseline, args.backbone, args.anchor, device)
        source = FeatureSource(config, 'navtest', data_root=args.data_root)
        records = source.records
        partitions = np.zeros(len(records), dtype=bool)
    paths = metric_paths(args.metric_cache)
    records = records[:args.max_scenes] if args.max_scenes else records
    run = Path(args.output)/datetime.now().strftime('%Y.%m.%d.%H.%M.%S.%f')
    run.mkdir(parents=True, exist_ok=False)
    write_new_json(run/'run.json', dict(arguments=vars(args), rank_policy=rank_meta['policy'],
                                        generator_sha256=sha256(args.generator_checkpoint)))
    rows = []
    with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context('spawn')) as pool:
        pending = []
        for index, record in enumerate(tqdm(records, desc='Generator timing evaluation')):
            if record['token'] not in paths:
                raise ValueError(f'Missing metric cache: {record["token"]}')
            if controls:
                entry = load_torch(entry_path(args.candidates, record))
                if entry['token'] != record['token'] or entry['provenance'] != source.provenance:
                    raise ValueError('Val candidate provenance mismatch')
                context = entry['context']
            else:
                from navsim.agents.diffusiondrive.pcs.common import generate_context
                _, features = source[index]
                context = generate_context(generator, features, record['token'], 0, device)
            families, divergence = _inputs(context, model, frozen, ranker, alpha,
                                           record['token'], device, controls)
            futures = {family: pool.submit(score_candidates, paths[record['token']],
                                           info['proposals'], index == 0)
                       for family, info in families.items()}
            pending.append((record, families, futures, divergence, bool(partitions[index])))
            if len(pending) >= 2*args.score_workers:
                job = pending.pop(0)
                row = _finish(*job[:4])
                row['calibration'] = job[4]
                rows.append(row)
        for job in pending:
            row = _finish(*job[:4])
            row['calibration'] = job[4]
            rows.append(row)
    if len(rows) != len(records):
        raise ValueError('Incomplete generator timing evaluation')
    if controls:
        if args.max_scenes:
            # Pilot does not make a locked decision: it may lack both log partitions.
            report = dict(schema='generator_timing_pilot_v1', scenes=len(rows),
                          original_pdm=float(np.mean([r['original_score'] for r in rows])),
                          generated_oracle_pdm=float(np.mean([r['generated']['oracle'] for r in rows])))
            write_new_json(run/'pilot.json', report)
        else:
            report = _calibration(rows)
            report['identity'] = identity
            write_new_json(run/'decision.json', report)
    else:
        margin = decision['selected_margin']
        report = dict(schema='generator_timing_navtest_v1', scenes=len(rows),
                      result=_aggregate(rows, 'generated', margin),
                      selected_margin=margin, decision_sha256=sha256(args.decision))
        write_new_json(run/'summary.json', report)
    with (run/'paired_results.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('token', 'log_name', 'calibration',
            'original_score', 'original_oracle', 'generated_oracle', 'generated_selected',
            'generated_mode_divergence'))
        writer.writeheader()
        for row in rows:
            margin = report.get('selected_margin')
            writer.writerow(dict(token=row['token'], log_name=row['log_name'],
                                 calibration=row['calibration'], original_score=row['original_score'],
                                 original_oracle=row['original_oracle'],
                                 generated_oracle=row['generated']['oracle'],
                                 generated_selected=(row['generated']['chosen'][margin]
                                                     if margin is not None else ''),
                                 generated_mode_divergence=row['divergence']))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Generator timing result:', run)
