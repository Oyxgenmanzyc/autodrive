"""Reproducible frozen-generator loading and cache provenance."""
import hashlib
import json
import time
from pathlib import Path

import torch

SCHEMA = "pcs_candidates_v1"
CONTEXT_KEYS = ("proposals", "base_logits", "bev", "agents", "ego")
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "planning/script/config"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_torch(path):
    # Only load trusted experiment checkpoints/caches. torch>=2.6 changes the default.
    return torch.load(path, map_location="cpu", weights_only=False)


def token_seed(token, seed):
    digest = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def build_generator(checkpoint, backbone, anchor, device):
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
    config = TransfuserConfig(bkb_path=str(backbone), plan_anchor_path=str(anchor))
    model = V2TransfuserModel(config)
    raw = load_torch(checkpoint)["state_dict"]
    state = {}
    for key, value in raw.items():
        if key.startswith("agent."):
            key = key[len("agent."):]
        if key.startswith("_transfuser_model."):
            state[key[len("_transfuser_model."):]] = value
        else:
            raise ValueError(f"Unexpected baseline checkpoint key: {key}")
    anchor_key = "_trajectory_head.plan_anchor"
    if anchor_key not in state:
        raise ValueError("Baseline checkpoint must contain the trained anchor bank")
    expected = model.state_dict()[anchor_key]
    if state[anchor_key].shape != expected.shape or not torch.equal(state[anchor_key].float(), expected.float()):
        raise ValueError("Checkpoint anchor and configured anchor differ; use the original K67 bank")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model, config


def generate_context(model, features, token, seed, device):
    # Per-token noise is invariant to worker order, sharding, and cache resume.
    devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(token_seed(token, seed))
        features = {key: value.unsqueeze(0).to(device) for key, value in features.items()}
        context = model(features, return_candidates=True)["pcs_context"]
    result = {}
    for key in CONTEXT_KEYS:
        value = context[key][0].detach().cpu()
        result[key] = value.float() if key in ("proposals", "base_logits") else value.half()
        if not torch.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite frozen feature: {token}/{key}")
    return result


def implementation_hashes():
    pcs_root = Path(__file__).resolve().parent
    implementation = [
        pcs_root / "model.py", pcs_root / "v2_blocks.py", pcs_root / "scoring.py",
        pcs_root / "common.py", pcs_root.parent / "transfuser_model_v2.py",
        pcs_root.parent / "transfuser_config.py", pcs_root.parent / "transfuser_features.py",
        CONFIG_ROOT / "pdm_scoring/default_scoring_parameters.yaml",
    ]
    return {p.name: sha256(p) for p in implementation}


def provenance(checkpoint, anchor, seed):
    return {
        "schema": SCHEMA, "baseline_sha256": sha256(checkpoint),
        "anchor_sha256": sha256(anchor), "seed": seed,
        "generator_precision": "float32", "context_precision": "float16",
        "generator_sampling": "original_2step_ddim_no_augmentation",
        "implementation_sha256": implementation_hashes(),
    }


def write_new_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Resuming identical metadata is allowed; conflicting runs never overwrite it.
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        # Other cache shards may be finishing the same small manifest.
        for attempt in range(100):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                if attempt == 99:
                    raise ValueError(f"Incomplete manifest: {path}")
                time.sleep(0.1)
        if existing != data:
            raise ValueError(f"Conflicting manifest: {path}; use a new output directory")
        return
    with stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def load_scorer(path, device, expected=None):
    from .model import PDMCSHead
    checkpoint = load_torch(path)
    meta = checkpoint["pcs_metadata"]
    if expected is not None and meta["provenance"] != expected:
        raise ValueError("Scorer checkpoint does not match baseline/anchor/seed")
    scorer = PDMCSHead(**meta["settings"])
    prefix = "head."
    state = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
    scorer.load_state_dict(state, strict=True)
    return scorer.eval().to(device), meta
