"""Calibration uses only its logs. Audit never selects alpha or checkpoint."""
import torch
from .model import select

ALPHAS = (0., .25, .5, 1.)
METRICS = ('nc', 'dac', 'ep', 'ttc', 'comfort')


def outcomes(modes, pcs_modes, labels, scores, direction):
    rows = torch.arange(len(modes), device=modes.device)
    chosen, old = labels[rows, modes].double(), labels[rows, pcs_modes].double()
    actual, baseline = scores[rows, modes].double(), scores[rows, pcs_modes].double()
    delta = actual-baseline
    result = dict(scenes=len(modes), pdm=float(actual.mean()), pcs_pdm=float(baseline.mean()),
                  gain_points=float(delta.mean()*100), changed=int((modes != pcs_modes).sum()),
                  beneficial=int((delta > 1e-6).sum()), harmful=int((delta < -1e-6).sum()),
                  new_zero=int(((baseline > 0) & (actual == 0)).sum()),
                  rescued_zero=int(((baseline == 0) & (actual > 0)).sum()),
                  severe_losses=int((delta <= -.2).sum()),
                  lost_pdm_sum=float((-delta).clamp_min(0).sum()),
                  gained_pdm_sum=float(delta.clamp_min(0).sum()),
                  selection_regret=float((scores.double().max(-1).values-actual).mean()))
    result.update({key: float(chosen[:, i].mean()) for i, key in enumerate(METRICS)})
    for i, key in ((0, 'nc'), (1, 'dac'), (3, 'ttc')):
        result['new_'+key+'_failure'] = int(((old[:, i] == 1) & (chosen[:, i] < 1)).sum())
        result[key+'_rescued'] = int(((old[:, i] < 1) & (chosen[:, i] == 1)).sum())
    # Aggregate constraint, NOT a per-scene guarantee; new failures are reported.
    result['safety_pass'] = bool((chosen[:, [0, 1, 3, 4]].mean(0)+1e-8 >=
                                  old[:, [0, 1, 3, 4]].mean(0)).all())
    result['safety_pass'] &= bool(direction[rows, modes].double().mean()+1e-8 >=
                                  direction[rows, pcs_modes].double().mean())
    return result


def alpha_grid(data):
    pcs_modes = data['pcs_scores'].argmax(-1)
    return [dict(alpha=a, **outcomes(select(data['pcs_scores'], data['residual'], a), pcs_modes,
                                   data['labels'], data['scores'], data['direction'])) for a in ALPHAS]


def calibrate(data):
    grid = alpha_grid(data)
    baseline = grid[0]['pdm']
    eligible = [r for r in grid if r['safety_pass'] and
                (r['alpha'] == 0 or r['pdm'] > baseline+1e-6)]
    best = max(eligible, key=lambda r: (r['pdm'], -r['changed'], -r['alpha']))
    return dict(enabled=best['alpha'] > 0, alpha=best['alpha']), grid


def reports(data, policy=None):
    mask = data['calibration'].bool()
    if not mask.any() or mask.all():
        raise ValueError('Both log-separated validation partitions must be populated')
    subsets = {name: {k: v[m] for k, v in data.items()} for name, m in
               (('calibration', mask), ('audit', ~mask))}
    if policy is None:
        policy, _ = calibrate(subsets['calibration'])
    output = {}
    for name, subset in subsets.items():
        grid = alpha_grid(subset)
        chosen = outcomes(select(subset['pcs_scores'], subset['residual'], policy['alpha']),
                          subset['pcs_scores'].argmax(-1), subset['labels'],
                          subset['scores'], subset['direction'])
        output[name] = dict(policy=chosen, grid=grid)
    return dict(policy=policy, reports=output)
