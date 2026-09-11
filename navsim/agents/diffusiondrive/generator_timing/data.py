"""Small GT target sidecar paired with frozen K67 contexts from PCS cache."""
from pathlib import Path
from types import SimpleNamespace
import json
import os

import torch
from torch.utils.data import Dataset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from navsim.agents.diffusiondrive.pcs.common import (
    CONFIG_ROOT, CONTEXT_KEYS, implementation_hashes, load_torch, sha256, token_seed,
)
from navsim.agents.diffusiondrive.pcs.data import entry_path
from navsim.common.dataclasses import Scene, SensorConfig
from navsim.common.dataloader import SceneLoader
from .risk_utils import build_gt_brake_timing_context
from .risk_brake_timing import _brake_timing_observations

SCHEMA = "generator_brake_timing_targets_v1"


def timing_config(weight=0.1):
    return SimpleNamespace(
        trajectory_sampling=SimpleNamespace(num_poses=8), risk_history_dt=0.5,
        risk_front_x_min=1.0, risk_front_x_max=32.0, risk_front_y_abs=1.8,
        risk_ego_front_offset=2.0, risk_ttc_max=10.0,
        brake_timing_accel_threshold=-0.5, brake_timing_temperature=0.35,
        brake_timing_profile_weight=0.25, brake_timing_preparation_time=1.0,
        brake_timing_loss_weight=weight,
    )


def target_identity():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in ("data.py", "risk_utils.py", "risk_brake_timing.py")}


def prepare_targets(args):
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    logs = OmegaConf.load(CONFIG_ROOT / "training/default_train_val_test_log_split.yaml")
    groups = {"train": set(logs.train_logs), "val": set(logs.val_logs)}
    if groups["train"] & groups["val"]:
        raise ValueError("Training/validation logs overlap")
    official = instantiate(OmegaConf.load(CONFIG_ROOT / "common/train_test_split/scene_filter/navtrain.yaml"))
    official_tokens = set(official.tokens)
    if len({r["token"] for r in records}) != len(records):
        raise ValueError("Duplicate record tokens")
    for r in records:
        if r["split"] not in groups or r["log_name"] not in groups[r["split"]] or r["token"] not in official_tokens:
            raise ValueError(f"Record is not in its official navtrain split: {r}")
    records.sort(key=lambda r: (r["split"], token_seed(r["token"], args.seed), r["token"]))
    if args.limit:
        records = [r for split in ("train", "val") for r in [x for x in records if x["split"] == split][:args.limit]]
    if not records or {r["split"] for r in records} != {"train", "val"}:
        raise ValueError("Both train and val records are required")
    identity = {
        "schema": SCHEMA, "implementation": target_identity(), "records": records,
        "data_root": str(Path(args.data_root).resolve()), "seed": args.seed,
    }
    path = Path(args.output)
    if path.exists():
        saved = load_torch(path)
        if saved["identity"] != identity:
            raise ValueError("Target sidecar conflicts; choose a new --output file")
        print(json.dumps(saved["summary"], indent=2))
        print(f"Reusing complete timing targets: {path}")
        return
    official.tokens = [r["token"] for r in records]
    official.log_names = sorted({r["log_name"] for r in records})
    raw = SceneLoader(
        Path(args.data_root) / "navsim_logs/trainval", Path(args.data_root) / "sensor_blobs/trainval",
        official, SensorConfig.build_no_sensors(),
    )
    absent = set(official.tokens) - set(raw.scene_frames_dicts)
    if absent:
        raise ValueError(f"Missing {len(absent)} raw log scenes; no silent filtering")
    contexts, trajectories = [], []
    config = timing_config()
    for r in tqdm(records, desc="GT brake-timing targets (no sensor loading)"):
        scene = Scene.from_scene_dict_list(
            raw.scene_frames_dicts[r["token"]], Path(args.data_root) / "sensor_blobs/trainval",
            official.num_history_frames, official.num_future_frames, SensorConfig.build_no_sensors(),
        )
        contexts.append(torch.as_tensor(build_gt_brake_timing_context(scene, config)).float())
        trajectories.append(torch.as_tensor(scene.get_future_trajectory(8).poses).float())
    contexts, trajectories = torch.stack(contexts), torch.stack(trajectories)
    if not torch.isfinite(contexts).all() or not torch.isfinite(trajectories).all():
        raise ValueError("Nonfinite timing targets")
    observations = _brake_timing_observations(trajectories, trajectories, contexts, config)
    summary = {}
    for split in ("train", "val"):
        mask = torch.tensor([r["split"] == split for r in records])
        summary[split] = {
            "scenes": int(mask.sum()), "active_scenes": int(observations["active"][mask].sum()),
            "active_rate": float(observations["active"][mask].float().mean()),
            "pre_risk_scenes": int(observations["pre_risk"][mask].sum()),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"identity": identity, "contexts": contexts, "trajectories": trajectories, "summary": summary}, temporary)
    # A concurrent writer is not supported: invoke prepare once, outside DDP.
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)
    print(json.dumps(summary, indent=2))
    print(f"Timing targets: {path} ({path.stat().st_size / 2**20:.2f} MiB)")


class TimingDataset(Dataset):
    def __init__(self, target_path, candidate_root, split, smoke=False):
        payload = load_torch(target_path)
        self.identity = payload["identity"]
        if self.identity["schema"] != SCHEMA or self.identity["implementation"] != target_identity():
            raise ValueError("Timing target implementation differs; regenerate into a new sidecar")
        self.records = self.identity["records"]
        self.indices = [i for i, r in enumerate(self.records) if r["split"] == split]
        self.contexts = payload["contexts"]
        self.trajectories = payload["trajectories"]
        self.candidate_root = Path(candidate_root)
        self.candidate_manifest = json.loads(
            (self.candidate_root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.candidate_manifest["dataset"] != "navtrain":
            raise ValueError("Generator training requires the navtrain candidate cache")
        if self.candidate_manifest["provenance"]["implementation_sha256"] != implementation_hashes():
            raise ValueError("Current generator/PCS implementation differs from the frozen context cache")
        cached_records = json.loads(
            (self.candidate_root / "records.json").read_text(encoding="utf-8")
        )
        cached = {(r["split"], r["token"]): r for r in cached_records}
        if len(cached) != len(cached_records):
            raise ValueError("Duplicate records in candidate cache")
        self.cache_records = {}
        for i in self.indices:
            key = (split, self.records[i]["token"])
            if key not in cached or cached[key]["log_name"] != self.records[i]["log_name"]:
                raise ValueError(f"Timing target is absent from candidate cache: {key}")
            self.cache_records[i] = cached[key]
        if not smoke and len(self.indices) < 1000:
            raise ValueError("Pilot targets cannot be used for full training")
        for i in self.indices:
            if not entry_path(self.candidate_root, self.cache_records[i]).is_file():
                raise FileNotFoundError(entry_path(self.candidate_root, self.cache_records[i]))
        if split == "train" and not smoke and payload["summary"]["train"]["active_scenes"] == 0:
            raise ValueError("No active timing supervision in the training split")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        entry = load_torch(entry_path(self.candidate_root, self.cache_records[i]))
        if entry["token"] != self.records[i]["token"]:
            raise ValueError("Candidate entry token mismatch")
        if entry["provenance"] != self.candidate_manifest["provenance"]:
            raise ValueError("Candidate entry provenance mismatch")
        context = {key: entry["context"][key] for key in CONTEXT_KEYS}
        if not all(torch.isfinite(value).all() for value in context.values()):
            raise ValueError(f"Nonfinite cached context: {entry['token']}")
        # Future annotations are only in targets, never features.
        return context, {"trajectory": self.trajectories[i], "brake_timing_context": self.contexts[i]}
