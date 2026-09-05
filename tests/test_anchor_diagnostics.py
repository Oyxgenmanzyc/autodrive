import copy

import numpy as np
import pytest

from navsim.agents.diffusiondrive.anchors.diagnostics import (
    audit_anchor_bank,
    empty_layer_stats,
    merge_rank_reports,
    summarize_layer,
    utilization_summary,
)


def test_utilization_empty_single_and_balanced():
    assert utilization_summary([0, 0])["entropy_nats"] == 0
    assert utilization_summary([4])["normalized_entropy"] == 0
    assert utilization_summary([1, 1])["entropy_nats"] == pytest.approx(np.log(2))
    assert utilization_summary([1, 1])["normalized_entropy"] == pytest.approx(1)
    assert utilization_summary([999, 1])["active_mode_rate"] == 0.5


@pytest.mark.parametrize("counts", [[], [-1, 1], [np.nan], [0.5]])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        utilization_summary(counts)


def test_audit_uses_static_ade_and_does_not_modify_anchors():
    anchors = np.stack([np.zeros((8, 2)), np.ones((8, 2)), np.ones((8, 2))])
    original = anchors.copy()
    trajectories = anchors[[0, 0, 1]]
    report = audit_anchor_bank(trajectories, anchors, batch_size=1)
    np.testing.assert_array_equal(anchors, original)
    assert report["static_assignment"]["counts"] == [2, 1, 0]
    assert report["coverage"]["mean_nearest_ADE"] == 0
    assert report["coverage"]["coverage@0.5m"] == 1
    assert report["diversity"]["nearest_neighbor_ADE_m"] == pytest.approx([np.sqrt(2), 0, 0])
    assert report["diversity"]["pair_fraction_ADE_below_m"]["0.15"] == pytest.approx(1 / 3)


def test_single_anchor_has_no_other_neighbor():
    anchors = np.zeros((1, 8, 2))
    report = audit_anchor_bank(anchors, anchors)
    assert report["diversity"]["mean_nearest_neighbor_ADE_m"] is None
    assert report["diversity"]["pair_count"] == 0


def _rank_report(rank, counts):
    stats = empty_layer_stats(len(counts))
    stats.update(winner_counts=counts, selected_counts=counts, finite_samples=sum(counts), samples_seen=sum(counts))
    return {
        "schema_version": 1, "run_root": "/run", "epoch": 0, "world_size": 2,
        "rank": rank, "bank_sha256": "abc", "mode_count": len(counts),
        "layers": {"0": summarize_layer(stats)},
    }


def test_offline_merge_recomputes_global_entropy_not_rank_mean():
    a, b = _rank_report(0, [3, 0]), _rank_report(1, [0, 1])
    merged = merge_rank_reports([a, b])["layers"]["0"]
    assert merged["winner_counts"] == [3, 1]
    assert merged["static_winner_utilization"]["entropy_nats"] > 0
    assert merged["static_winner_utilization"]["frequency"] == [0.75, 0.25]


def test_offline_merge_rejects_missing_duplicate_and_incompatible_ranks():
    a, b = _rank_report(0, [1, 0]), _rank_report(1, [0, 1])
    for reports in ([a], [a, a]):
        with pytest.raises(ValueError):
            merge_rank_reports(reports)
    bad = copy.deepcopy(b)
    bad["bank_sha256"] = "other"
    with pytest.raises(ValueError):
        merge_rank_reports([a, bad])


def test_merge_uses_sample_weighted_losses_and_preserves_invalid_counts():
    a, b = _rank_report(0, [3, 0]), _rank_report(1, [0, 1])
    a["layers"]["0"].update(weighted_cls_sum=6.0, nonfinite_batches=1, samples_seen=5)
    b["layers"]["0"].update(weighted_cls_sum=8.0)
    merged = merge_rank_reports([a, b])["layers"]["0"]
    assert merged["trajectory_cls_loss"] == 3.5
    assert merged["finite_samples"] == 4
    assert merged["samples_seen"] == 6
    assert merged["nonfinite_batches"] == 1
