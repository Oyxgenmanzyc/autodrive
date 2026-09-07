"""Sources, token manifests, and disk caches for frozen PCS training."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from hydra.utils import instantiate

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.agents.diffusiondrive.transfuser_features import TransfuserFeatureBuilder
from navsim.planning.training.dataset import CacheOnlyDataset
from .common import CONFIG_ROOT, CONTEXT_KEYS, load_torch, token_seed, implementation_hashes


class FeatureSource(Dataset):
    def __init__(self, config, split, feature_cache=None, data_root=None, limit=0, seed=0):
        self.builder = TransfuserFeatureBuilder(config=config)
        self.cache = None
        self.scene_loader = None
        self.records = []
        if split == "navtrain":
            logs = OmegaConf.load(CONFIG_ROOT / "training/default_train_val_test_log_split.yaml")
            train_logs, val_logs = set(logs.train_logs), set(logs.val_logs)
            if train_logs & val_logs:
                raise ValueError("Train/validation log lists overlap")
            self.cache = CacheOnlyDataset(
                str(feature_cache), [self.builder], [], sorted(train_logs | val_logs),
            )
            for token, path in self.cache._valid_cache_paths.items():
                self.records.append({
                    "token": token, "log_name": path.parent.name,
                    "split": "train" if path.parent.name in train_logs else "val",
                })
        else:
            scene_filter = instantiate(OmegaConf.load(
                CONFIG_ROOT / "common/train_test_split/scene_filter/navtest.yaml"
            ))
            self.scene_loader = SceneLoader(
                data_path=Path(data_root) / "navsim_logs/test",
                sensor_blobs_path=Path(data_root) / "sensor_blobs/test",
                scene_filter=scene_filter,
                sensor_config=SensorConfig.build_all_sensors(include=[3]),
            )
            for log, tokens in self.scene_loader.get_tokens_list_per_log().items():
                self.records.extend({"token": token, "log_name": log, "split": "test"} for token in tokens)
        self.records.sort(key=lambda r: (r["split"], token_seed(r["token"], seed), r["token"]))
        self.available_counts = {
            group: sum(r["split"] == group for r in self.records)
            for group in sorted({r["split"] for r in self.records})
        }
        if limit:
            # Limit independently per split, so smoke cache contains train AND val.
            groups = sorted({r["split"] for r in self.records})
            self.records = [r for group in groups for r in [x for x in self.records if x["split"] == group][:limit]]
        if not self.records:
            raise ValueError("No input scenes found")
        if len({r["token"] for r in self.records}) != len(self.records):
            raise ValueError("Duplicate source tokens")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        token = record["token"]
        if self.cache is not None:
            features, _ = self.cache._load_scene_with_token(token)
        else:
            features = self.builder.compute_features(self.scene_loader.get_agent_input_from_token(token))
        return record, features


def metric_paths(root):
    # The old MetricCacheLoader only reads the first metadata CSV.
    # Read the actual cache files so caches from multiple jobs remain complete.
    result = {}
    for path in Path(root).rglob("metric_cache.pkl"):
        token = path.parent.name
        if token in result:
            raise ValueError(f"Duplicate metric cache token: {token}")
        result[token] = str(path)
    if not result:
        raise ValueError(f"No metric_cache.pkl under {root}")
    return result


def entry_path(root, record):
    return Path(root) / record["split"] / (record["token"] + ".pt")


class CandidateDataset(Dataset):
    def __init__(self, root, split, allow_partial=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["dataset"] != "navtrain":
            raise ValueError("Training must use navtrain caches, never navtest labels")
        if self.manifest["limit_per_split"] and not allow_partial:
            raise ValueError("Pilot cache cannot be used for full training; use --smoke or build the full cache")
        records = json.loads((self.root / "records.json").read_text(encoding="utf-8"))
        if len({r["token"] for r in records}) != len(records):
            raise ValueError("Duplicate cache tokens, possibly crossing train/val")
        if self.manifest["provenance"]["implementation_sha256"] != implementation_hashes():
            raise ValueError("Current implementation differs from the candidate cache")
        self.records = [r for r in records if r["split"] == split]
        missing = [r["token"] for r in self.records if not entry_path(root, r).is_file()]
        if missing:
            raise ValueError(f"Candidate cache incomplete: {len(missing)} missing {split} scenes")
        if not self.records:
            raise ValueError(f"No {split} samples")
        self.provenance = self.manifest["provenance"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        entry = load_torch(entry_path(self.root, record))
        if entry["token"] != record["token"] or entry["provenance"] != self.provenance:
            raise ValueError("Cache entry identity/provenance mismatch")
        return {
            "context": {k: entry["context"][k] for k in CONTEXT_KEYS},
            "labels": entry["labels"], "scores": entry["scores"],
            "index": index,
        }
