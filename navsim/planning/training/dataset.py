from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os
import hashlib

import numpy as np
import torch
from tqdm import tqdm
from pyquaternion import Quaternion

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)


def _yaw_from_scene_frame(scene_frame: Dict[str, Any]) -> float:
    """Extracts ego yaw from a raw NAVSIM scene frame."""
    ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
    return ego_quaternion.yaw_pitch_roll[0]


def _previous_ego_delta_in_current_frame(previous_frame: Dict[str, Any], current_frame: Dict[str, Any]) -> np.ndarray:
    """Returns previous ego -> current ego displacement in the current ego frame."""
    previous_pose = _previous_ego_pose_in_current_frame(previous_frame, current_frame)
    return (-previous_pose[:2]).astype(np.float32)


def _previous_ego_pose_in_current_frame(previous_frame: Dict[str, Any], current_frame: Dict[str, Any]) -> np.ndarray:
    """Returns previous ego pose as (x, y, heading) in the current ego frame."""
    previous_translation = np.asarray(previous_frame["ego2global_translation"][:2], dtype=np.float32)
    current_translation = np.asarray(current_frame["ego2global_translation"][:2], dtype=np.float32)
    previous_yaw = _yaw_from_scene_frame(previous_frame)
    current_yaw = _yaw_from_scene_frame(current_frame)
    cos_h = np.cos(current_yaw)
    sin_h = np.sin(current_yaw)
    world_to_current = np.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float32)
    previous_xy_in_current = world_to_current @ (previous_translation - current_translation)
    previous_heading_in_current = np.arctan2(
        np.sin(previous_yaw - current_yaw),
        np.cos(previous_yaw - current_yaw),
    )
    return np.array(
        [previous_xy_in_current[0], previous_xy_in_current[1], previous_heading_in_current],
        dtype=np.float32,
    )


class TemporalPairCacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper that returns adjacent cached samples as prev -> current pairs."""

    def __init__(
        self,
        cache_path: str,
        data_path: str,
        scene_filter: SceneFilter,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        pair_index_cache_path: Optional[str] = None,
        split_name: str = "train",
    ):
        super().__init__()
        self._base_dataset = CacheOnlyDataset(
            cache_path=cache_path,
            feature_builders=feature_builders,
            target_builders=target_builders,
            log_names=log_names,
        )
        self._cache_path = Path(cache_path)
        self._data_path = Path(data_path)
        self._scene_filter = scene_filter
        self._pair_index_cache_path = Path(pair_index_cache_path) if pair_index_cache_path else None
        self._split_name = split_name
        self._pairs = self._load_or_build_pair_index(log_names=log_names)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self._pairs[idx]
        prev_features, prev_targets = self._base_dataset._load_scene_with_token(pair["prev_token"])
        curr_features, curr_targets = self._base_dataset._load_scene_with_token(pair["curr_token"])
        return {
            "prev_features": prev_features,
            "prev_targets": prev_targets,
            "curr_features": curr_features,
            "curr_targets": curr_targets,
            "pair_metadata": {
                "log_name": pair["log_name"],
                "prev_token": pair["prev_token"],
                "curr_token": pair["curr_token"],
                "previous_ego_delta": torch.tensor(pair["previous_ego_delta"], dtype=torch.float32),
                "previous_ego_pose": torch.tensor(pair["previous_ego_pose"], dtype=torch.float32),
            },
        }

    @property
    def tokens(self) -> List[str]:
        return [pair["curr_token"] for pair in self._pairs]

    def _index_cache_file(self, log_names: Optional[List[str]]) -> Optional[Path]:
        if self._pair_index_cache_path is None:
            return None
        os.makedirs(self._pair_index_cache_path, exist_ok=True)
        log_key = ",".join(log_names or [])
        digest = hashlib.sha1(log_key.encode("utf-8")).hexdigest()[:12]
        return self._pair_index_cache_path / f"{self._split_name}_temporal_pairs_{digest}.pkl"

    def _metadata(self, log_names: Optional[List[str]]) -> Dict[str, Any]:
        return {
            "split_name": self._split_name,
            "cache_path": str(self._cache_path),
            "data_path": str(self._data_path),
            "log_names": list(log_names or []),
            "num_history_frames": self._scene_filter.num_history_frames,
            "num_future_frames": self._scene_filter.num_future_frames,
            "frame_interval": self._scene_filter.frame_interval,
            "has_route": self._scene_filter.has_route,
        }

    def _load_or_build_pair_index(self, log_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        index_cache_file = self._index_cache_file(log_names)
        expected_metadata = self._metadata(log_names)
        if index_cache_file is not None and index_cache_file.is_file():
            with open(index_cache_file, "rb") as f:
                payload = pickle.load(f)
            if payload.get("metadata") == expected_metadata:
                pairs = self._filter_pairs_with_valid_cache(payload["pairs"])
                logger.info("Loaded %d temporal pairs from %s", len(pairs), index_cache_file)
                return pairs

        pairs = self._build_pair_index(log_names=log_names)
        if index_cache_file is not None:
            with open(index_cache_file, "wb") as f:
                pickle.dump({"metadata": expected_metadata, "pairs": pairs}, f)
            logger.info("Saved %d temporal pairs to %s", len(pairs), index_cache_file)
        return pairs

    def _filter_pairs_with_valid_cache(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_tokens = set(self._base_dataset.tokens)
        return [pair for pair in pairs if pair["prev_token"] in valid_tokens and pair["curr_token"] in valid_tokens]

    def _build_pair_index(self, log_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        valid_tokens = set(self._base_dataset.tokens)
        selected_log_names = set(log_names or [])
        pairs: List[Dict[str, Any]] = []
        log_files = sorted(self._data_path.iterdir())
        if selected_log_names:
            log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in selected_log_names]

        for log_pickle_path in tqdm(log_files, desc=f"Building {self._split_name} temporal pairs"):
            scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
            log_pairs = self._build_pairs_for_log(scene_dict_list=scene_dict_list, valid_tokens=valid_tokens)
            pairs.extend(log_pairs)

        logger.info("Built %d %s temporal pairs", len(pairs), self._split_name)
        return pairs

    def _build_pairs_for_log(
        self,
        scene_dict_list: List[Dict[str, Any]],
        valid_tokens: set,
    ) -> List[Dict[str, Any]]:
        sample_records: List[Dict[str, Any]] = []
        num_frames = self._scene_filter.num_frames
        step = self._scene_filter.frame_interval
        history_idx = self._scene_filter.num_history_frames - 1
        filter_tokens = self._scene_filter.tokens is not None
        allowed_tokens = set(self._scene_filter.tokens or [])

        for start_idx in range(0, len(scene_dict_list), step):
            frame_list = scene_dict_list[start_idx : start_idx + num_frames]
            if len(frame_list) < num_frames:
                continue
            current_frame = frame_list[history_idx]
            if self._scene_filter.has_route and len(current_frame["roadblock_ids"]) == 0:
                continue
            current_token = current_frame["token"]
            if filter_tokens and current_token not in allowed_tokens:
                continue
            if current_token not in valid_tokens:
                continue
            sample_records.append(
                {
                    "token": current_token,
                    "frame": current_frame,
                    "log_name": current_frame["log_name"],
                    "start_idx": start_idx,
                }
            )

        pairs: List[Dict[str, Any]] = []
        for prev_record, curr_record in zip(sample_records[:-1], sample_records[1:]):
            if curr_record["start_idx"] - prev_record["start_idx"] != step:
                continue
            prev_token = prev_record["token"]
            curr_token = curr_record["token"]
            if prev_token not in valid_tokens or curr_token not in valid_tokens:
                continue
            pairs.append(
                {
                    "log_name": curr_record["log_name"],
                    "prev_token": prev_token,
                    "curr_token": curr_token,
                    "previous_ego_delta": _previous_ego_delta_in_current_frame(
                        prev_record["frame"],
                        curr_record["frame"],
                    ),
                    "previous_ego_pose": _previous_ego_pose_in_current_frame(
                        prev_record["frame"],
                        curr_record["frame"],
                    ),
                }
            )
        return pairs


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)
