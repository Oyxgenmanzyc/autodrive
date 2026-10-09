"""Calibrate on one set of logs; audit the frozen decision on other logs."""
import torch

from .geometry import MODE_NAMES, SPATIAL_ONLY_INDICES
from .model import choose

THRESHOLDS = (0., .01, .02, .05, .1, .2, .4)


def outcome(predicted, labels, scores, direction, threshold, allowed=None):
    chosen = choose(predicted, threshold, allowed)
    row = torch.arange(len(chosen))
    actual, original = scores[row, chosen].double(), scores[:, 0].double()
    metric, base_metric = labels[row, chosen].double(), labels[:, 0].double()
    delta = actual-original
    result = dict(scenes=len(chosen), rank_pdm=float(original.mean()), pdm=float(actual.mean()),
                  gain_points=float(delta.mean()*100.), changed=int((chosen != 0).sum()),
                  beneficial=int((delta > 1e-6).sum()), harmful=int((delta < -1e-6).sum()),
                  severe_losses=int((delta <= -.2).sum()),
                  mean_oracle_regret=float((scores.double().max(-1).values-actual).mean()),
                  action_family_counts={name: int(((chosen != 0) & (chosen % len(MODE_NAMES) == i)).sum())
                                        for i, name in enumerate(MODE_NAMES)})
    for i, name in ((0, 'nc'), (1, 'dac'), (3, 'ttc')):
        result['new_'+name+'_failure'] = int(((base_metric[:, i] >= 1.-1e-6) &
                                              (metric[:, i] < 1.-1e-6)).sum())
        result[name+'_rescued'] = int(((base_metric[:, i] < 1.-1e-6) &
                                       (metric[:, i] >= 1.-1e-6)).sum())
    guarded = (0, 1, 3, 4)
    result['safety_pass'] = bool((metric[:, guarded].mean(0)+1e-8 >=
                                  base_metric[:, guarded].mean(0)).all())
    result['safety_pass'] &= bool(direction[row, chosen].double().mean()+1e-8 >=
                                  direction[:, 0].double().mean())
    return result


def reports(data, policy=None):
    mask = data['calibration'].bool()
    if not mask.any() or mask.all():
        raise ValueError('Independent log-separated calibration/audit required')
    parts = {name: {key: value[indices] for key, value in data.items()}
             for name, indices in (('calibration', mask), ('audit', ~mask))}

    def grid(part):
        return [dict(threshold=t, **outcome(part['predicted'], part['labels'], part['scores'],
                                            part['direction'], t)) for t in THRESHOLDS]

    if policy is None:
        calibration = grid(parts['calibration'])
        eligible = [r for r in calibration if r['gain_points'] > 0 and r['safety_pass'] and
                    r['severe_losses'] == 0]
        best = max(eligible, key=lambda r: (r['gain_points'], -r['changed'])) if eligible else None
        policy = dict(enabled=best is not None,
                      threshold=best['threshold'] if best else max(THRESHOLDS))
    result = {}
    for name, part in parts.items():
        selected = outcome(part['predicted'], part['labels'], part['scores'], part['direction'],
                           policy['threshold']) if policy['enabled'] else outcome(
                               part['predicted'].new_zeros(part['predicted'].shape),
                               part['labels'], part['scores'], part['direction'], max(THRESHOLDS))
        spatial = outcome(part['predicted'], part['labels'], part['scores'], part['direction'],
                          policy['threshold'], SPATIAL_ONLY_INDICES) if policy['enabled'] else selected
        result[name] = dict(policy=selected, spatial_only_control=spatial,
                            timing_incremental_gain_points=(selected['pdm']-spatial['pdm'])*100.,
                            grid=grid(part))
    audit = result['audit']['policy']
    timing_incremental = result['audit']['timing_incremental_gain_points']
    passed = (policy['enabled'] and audit['gain_points'] > 0 and audit['safety_pass'] and
              audit['severe_losses'] == 0 and timing_incremental > 0 and
              all(audit['new_'+name+'_failure'] == 0 for name in ('nc', 'dac', 'ttc')))
    return dict(policy=policy, reports=result, pass_for_navtest=bool(passed),
                note='Calibration alone chooses the threshold. Audit is never used to tune.')
