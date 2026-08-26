"""Extract model-aligned eight-step GT trajectories from the NAVSIM navtrain split."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from tqdm import tqdm

from navsim.agents.diffusiondrive.anchors.trajectory_distance import (
    NAVSIM_ONE_HOT_COMMANDS,
    TrajectoryCommand,
    navsim_command_from_one_hot,
)


def _local_future_trajectory(frames: list[dict[str, Any]], current_idx: int, num_poses: int) -> np.ndarray:
    current = frames[current_idx]
    current_xy = np.asarray(current["ego2global_translation"][:2], dtype=np.float64)
    current_yaw = Quaternion(*current["ego2global_rotation"]).yaw_pitch_roll[0]
    future_xy = np.asarray(
        [frames[idx]["ego2global_translation"][:2] for idx in range(current_idx + 1, current_idx + num_poses + 1)],
        dtype=np.float64,
    )
    delta = future_xy - current_xy
    cosine, sine = np.cos(current_yaw), np.sin(current_yaw)
    global_to_ego = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return (delta @ global_to_ego.T).astype(np.float32)


def _iter_navtrain_samples(
    data_path: Path,
    filter_config: Any,
    num_poses: int,
) -> Iterable[tuple[np.ndarray, TrajectoryCommand]]:
    selected_logs = set(filter_config.get("log_names") or [])
    selected_tokens = set(filter_config.get("tokens") or [])
    history_frames = int(filter_config.get("num_history_frames", 4))
    future_frames = int(filter_config.get("num_future_frames", 10))
    frame_interval = int(filter_config.get("frame_interval", history_frames + future_frames))
    has_route = bool(filter_config.get("has_route", True))
    max_scenes = filter_config.get("max_scenes")
    required_frames = history_frames + max(future_frames, num_poses)
    emitted = 0

    log_paths = sorted(path for path in data_path.iterdir() if path.is_file() and path.suffix == ".pkl")
    if selected_logs:
        log_paths = [path for path in log_paths if path.stem in selected_logs]
    for log_path in tqdm(log_paths, desc="Extracting navtrain trajectories"):
        with log_path.open("rb") as file:
            frames = pickle.load(file)
        for start_idx in range(0, len(frames), frame_interval):
            scene_frames = frames[start_idx : start_idx + required_frames]
            if len(scene_frames) < required_frames:
                continue
            current_idx = history_frames - 1
            current = scene_frames[current_idx]
            if has_route and not current["roadblock_ids"]:
                continue
            if selected_tokens and current["token"] not in selected_tokens:
                continue
            command = navsim_command_from_one_hot(current["driving_command"])
            if command is None:
                continue
            yield _local_future_trajectory(scene_frames, current_idx, num_poses), command
            emitted += 1
            if max_scenes is not None and emitted >= int(max_scenes):
                return


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_data = None
    if os.environ.get("OPENSCENE_DATA_ROOT"):
        default_data = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / "trainval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=default_data)
    parser.add_argument(
        "--navtrain-filter-config",
        type=Path,
        default=root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-poses", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_path is None:
        raise ValueError("Set --data-path or OPENSCENE_DATA_ROOT")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"NAVSIM log directory does not exist: {args.data_path}")
    filter_config = OmegaConf.load(args.navtrain_filter_config)
    samples = list(_iter_navtrain_samples(args.data_path, filter_config, args.num_poses))
    if not samples:
        raise RuntimeError("No navtrain trajectories were extracted; check paths and split filters")
    trajectories, commands = zip(*samples)
    array = np.stack(trajectories).astype(np.float32)
    command_array = np.asarray([command.value for command in commands], dtype="<U8")
    command_path = args.output.with_name(f"{args.output.stem}_commands.npy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)
    np.save(command_path, command_array)
    metadata = {
        "split": "navtrain",
        "data_path": str(args.data_path),
        "filter_config": str(args.navtrain_filter_config),
        "num_trajectories": int(len(array)),
        "shape": list(array.shape),
        "commands_path": str(command_path),
        "command_counts": {
            command.value: int(np.sum(command_array == command.value))
            for command in NAVSIM_ONE_HOT_COMMANDS
            if command is not None
        },
        "unknown_command_filtered": True,
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(array)} navtrain trajectories to {args.output}")
    print(f"Saved semantic commands to {command_path}")


if __name__ == "__main__":
    main()
