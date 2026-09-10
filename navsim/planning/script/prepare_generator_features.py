"""Optional recovery of deleted raw features; never overwrite existing files."""
import argparse
import json
import os
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from navsim.agents.diffusiondrive.pcs.common import CONFIG_ROOT
from navsim.agents.diffusiondrive.generator_timing.data import feature_path
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.dataset import dump_feature_target_to_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "feature-cache", "data-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    missing = [r for r in records if not feature_path(args.feature_cache, r).is_file()]
    print(f"Missing raw features: {len(missing)} of {len(records)}")
    if not missing:
        return
    config = instantiate(OmegaConf.load(CONFIG_ROOT / "common/train_test_split/scene_filter/navtrain.yaml"))
    tokens = set(config.tokens)
    if any(r["token"] not in tokens for r in missing):
        raise ValueError("Non-navtrain token in records")
    config.tokens = [r["token"] for r in missing]
    config.log_names = sorted({r["log_name"] for r in missing})
    sensors = SensorConfig.build_all_sensors(include=[3])
    blobs = Path(args.data_root) / "sensor_blobs/trainval"
    scenes = SceneLoader(Path(args.data_root) / "navsim_logs/trainval", blobs, config, sensors)
    if set(config.tokens) - set(scenes.scene_frames_dicts):
        raise ValueError("Required raw logs are missing")
    builder = TransfuserFeatureBuilder(TransfuserConfig())
    for r in tqdm(missing, desc="Recovering raw image/LiDAR features"):
        path = feature_path(args.feature_cache, r)
        agent_input = AgentInput.from_scene_dict_list(scenes.scene_frames_dicts[r["token"]], blobs, config.num_history_frames, sensors)
        features = builder.compute_features(agent_input)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        dump_feature_target_to_pickle(temporary, features)
        # Hard link publishes a complete file and refuses to overwrite a peer.
        os.link(temporary, path)
        temporary.unlink()
    print("Raw feature cache complete; now run prepare-pilot or prepare.")


if __name__ == "__main__":
    main()
