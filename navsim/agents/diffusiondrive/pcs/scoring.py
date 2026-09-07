"""Batch simulator + independent reference/candidate PDMS normalization.

Adapted from DiffusionDriveV2 _pairwise_subscores (MIT, commit 1cd12a1).
The reference at index zero is the cached PDM reference, NOT expert GT input
to the learned scorer. Avoid group-dependent progress normalization.
"""
import lzma
import pickle
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array, pdm_score
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex as M, WeightedMetricIndex as W,
)
from .model import METRIC_NAMES

_ENGINE = None


def pairwise_subscores(scorer):
    multi = scorer._multi_metrics
    weighted = scorer._weighted_metrics.copy()
    gate = multi.prod(axis=0)
    progress = scorer._progress_raw * gate
    denominator = np.maximum(progress[0], progress[1:])
    moving = denominator > scorer._config.progress_distance_threshold
    normalized = np.where(gate[1:] == 0, 0.0, 1.0)
    # No epsilon perturbation; divide only when denominator passes the threshold.
    np.divide(progress[1:], denominator, out=normalized, where=moving)
    weighted[W.PROGRESS, 1:] = normalized
    coef = scorer._config.weighted_metrics_array
    score = gate[1:] * (weighted[:, 1:] * coef[:, None]).sum(0) / coef.sum()
    sub = np.stack([
        multi[M.NO_COLLISION, 1:], multi[M.DRIVABLE_AREA, 1:],
        weighted[W.PROGRESS, 1:], weighted[W.TTC, 1:], weighted[W.COMFORTABLE, 1:],
    ], axis=-1)
    return sub, score, weighted[W.DRIVING_DIRECTION, 1:]


def scoring_engine():
    global _ENGINE
    if _ENGINE is None:
        config_path = Path(__file__).resolve().parents[3] / "planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
        config = OmegaConf.load(config_path)
        _ENGINE = instantiate(config.simulator), instantiate(config.scorer)
    return _ENGINE


def score_candidates(metric_path, proposals, verify=False):
    """CPU worker; verification compares all five subscores + total to official calls."""
    simulator, scorer = scoring_engine()
    with lzma.open(metric_path, "rb") as stream:
        cache = pickle.load(stream)
    initial = cache.ego_state
    sampling = simulator.proposal_sampling
    states = [get_trajectory_as_array(cache.trajectory, sampling, initial.time_point)]
    for proposal in proposals:
        trajectory = transform_trajectory(Trajectory(proposal), initial)
        states.append(get_trajectory_as_array(trajectory, sampling, initial.time_point))
    simulated = simulator.simulate_proposals(np.stack(states), initial)
    scorer.score_proposals(
        simulated, cache.observation, cache.centerline,
        cache.route_lane_ids, cache.drivable_area_map,
    )
    labels, scores, direction = pairwise_subscores(scorer)
    if not all(np.isfinite(x).all() for x in (labels, scores, direction)):
        raise ValueError("Nonfinite PDM label; do not silently cache a failed scene")
    if verify:
        for i in sorted({0, len(proposals) // 2, len(proposals) - 1}):
            result = pdm_score(cache, Trajectory(proposals[i]), sampling, simulator, scorer)
            expected = np.array([getattr(result, key) for key in METRIC_NAMES])
            np.testing.assert_allclose(labels[i], expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(scores[i], result.score, rtol=0, atol=1e-6)
            np.testing.assert_allclose(direction[i], result.driving_direction_compliance, rtol=0, atol=1e-6)
    return labels.astype(np.float32), scores.astype(np.float32), direction.astype(np.float32)

