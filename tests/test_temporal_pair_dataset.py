import pickle

import torch

from navsim.common.dataclasses import SceneFilter
from navsim.planning.training.dataset import (
    TemporalPairCacheOnlyDataset,
    _previous_ego_delta_in_current_frame,
    dump_feature_target_to_pickle,
)
from navsim.planning.training.agent_lightning_module import TemporalPairAgentLightningModule


class _Builder:
    def __init__(self, name):
        self._name = name

    def get_unique_name(self):
        return self._name


def _frame(token, x):
    return {
        "token": token,
        "log_name": "log",
        "roadblock_ids": ["route"],
        "ego2global_translation": [x, 0.0, 0.0],
        "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
    }


def _write_cache(cache_path, token, value):
    token_path = cache_path / "log" / token
    token_path.mkdir(parents=True)
    dump_feature_target_to_pickle(token_path / "feature.gz", {"x": torch.tensor([value], dtype=torch.float32)})
    dump_feature_target_to_pickle(token_path / "target.gz", {"y": torch.tensor([value], dtype=torch.float32)})


def test_previous_ego_delta_is_current_frame_displacement():
    delta = _previous_ego_delta_in_current_frame(_frame("prev", 0.0), _frame("curr", 1.0))

    assert torch.allclose(torch.tensor(delta), torch.tensor([1.0, 0.0]))


def test_temporal_pair_index_requires_adjacent_filtered_samples():
    dataset = TemporalPairCacheOnlyDataset.__new__(TemporalPairCacheOnlyDataset)
    dataset._scene_filter = SceneFilter(
        num_history_frames=1,
        num_future_frames=0,
        frame_interval=1,
        has_route=True,
        tokens=["t0", "t1", "t3"],
    )
    pairs = dataset._build_pairs_for_log(
        scene_dict_list=[_frame("t0", 0.0), _frame("t1", 1.0), _frame("t2", 2.0), _frame("t3", 3.0)],
        valid_tokens={"t0", "t1", "t2", "t3"},
    )

    assert [(pair["prev_token"], pair["curr_token"]) for pair in pairs] == [("t0", "t1")]


def test_temporal_pair_dataset_returns_prev_and_current_cache(tmp_path):
    cache_path = tmp_path / "cache"
    data_path = tmp_path / "logs"
    data_path.mkdir()
    for idx, token in enumerate(["t0", "t1", "t2"]):
        _write_cache(cache_path, token, idx)
    with open(data_path / "log.pkl", "wb") as f:
        pickle.dump([_frame("t0", 0.0), _frame("t1", 1.0), _frame("t2", 2.0)], f)

    dataset = TemporalPairCacheOnlyDataset(
        cache_path=str(cache_path),
        data_path=str(data_path),
        scene_filter=SceneFilter(num_history_frames=1, num_future_frames=0, frame_interval=1, has_route=True),
        feature_builders=[_Builder("feature")],
        target_builders=[_Builder("target")],
        log_names=["log"],
        pair_index_cache_path=str(tmp_path / "pair_index"),
        split_name="train",
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["prev_features"]["x"].shape == sample["curr_features"]["x"].shape
    assert sample["pair_metadata"]["prev_token"] == "t0"
    assert sample["pair_metadata"]["curr_token"] == "t1"
    assert torch.allclose(sample["pair_metadata"]["previous_ego_delta"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(sample["pair_metadata"]["previous_ego_pose"], torch.tensor([-1.0, 0.0, 0.0]))


def test_temporal_pair_reference_keeps_connection_points_p0_to_p2():
    previous_trajectory = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ]
    )
    previous_ego_pose = torch.tensor([[-1.0, 0.0, 0.0]])

    temporal_reference = TemporalPairAgentLightningModule._build_temporal_reference(
        previous_trajectory,
        previous_ego_pose,
    )

    assert torch.allclose(
        temporal_reference,
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
    )
