"""Resumable small history sidecar. No GT enters timing query inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.agents.diffusiondrive.pcs.common import CONFIG_ROOT, sha256, load_torch, write_new_json
from navsim.agents.diffusiondrive.pcs.data import FeatureSource
from navsim.agents.diffusiondrive.generator_timing.data import TimingDataset, target_identity
from .history import build_history_risk_tokens

SCHEMA = 'timing_query_inputs_v1'


def input_sources():
    return {name: sha256(Path(__file__).parent / name) for name in ('history.py', 'data.py')}


def risk_input(agent_input):
    # Four past/current LiDAR frames ONLY, with the same builder in train/test.
    config = SimpleNamespace(risk_history_num_frames=4, risk_history_dt=.5,
                             risk_front_x_min=1., risk_front_x_max=32., risk_front_y_abs=1.8,
                             risk_lidar_min_z=.2, risk_lidar_max_z=3., risk_lidar_min_points=3,
                             risk_lidar_gap_percentile=10., risk_ego_front_offset=2.,
                             risk_ttc_max=10., risk_drac_max=6.)
    tokens = build_history_risk_tokens(agent_input, config)
    speed = float(max(0., agent_input.ego_statuses[-1].ego_velocity[0]))
    if not np.isfinite(tokens).all() or not np.isfinite(speed):
        raise ValueError('Nonfinite timing input')
    return torch.from_numpy(tokens), torch.tensor(speed, dtype=torch.float32)


class HistorySource(Dataset):
    def __init__(self, records, data_root):
        self.records = records
        cfg = instantiate(OmegaConf.load(CONFIG_ROOT / 'common/train_test_split/scene_filter/navtrain.yaml'))
        cfg.tokens = [r['token'] for r in records]
        cfg.log_names = sorted({r['log_name'] for r in records})
        sensors = SensorConfig.build_no_sensors()
        sensors.lidar_pc = [0, 1, 2, 3]
        self.scene_loader = SceneLoader(Path(data_root) / 'navsim_logs/trainval',
                                        Path(data_root) / 'sensor_blobs/trainval', cfg, sensors)
        missing = set(cfg.tokens) - set(self.scene_loader.scene_frames_dicts)
        if missing:
            raise ValueError(f'Missing raw history for {len(missing)} scenes')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        token = self.records[i]['token']
        history, speed = risk_input(self.scene_loader.get_agent_input_from_token(token))
        return {'token': token, 'history': history, 'ego_speed': speed}


def prepare_inputs(args):
    from navsim.planning.script.run_pcs import loader, save_entry
    target = load_torch(args.targets)
    if target['identity']['implementation'] != target_identity():
        raise ValueError('GT timing targets differ from their source implementation')
    records = target['identity']['records']
    root = Path(args.output)
    identity = dict(schema=SCHEMA, sources=input_sources(), target_sha256=sha256(args.targets),
                    records=records, block_size=512)
    write_new_json(root / 'manifest.json', identity)
    paths = [root / f'block_{i//512:04d}.pt' for i in range(0, len(records), 512)]
    for i, path in enumerate(paths):
        if path.exists():
            saved = load_torch(path)
            if saved['tokens'] != [r['token'] for r in records[i*512:(i+1)*512]]:
                raise ValueError(f'Conflicting history block: {path}')
    pending_records = [r for i, r in enumerate(records) if not paths[i//512].exists()]
    if pending_records:
        source = HistorySource(pending_records, args.data_root)
        block, offset = [], {r['token']: i for i, r in enumerate(records)}
        for item in tqdm(loader(source, args.workers, batch_size=None), desc='Timing query inputs (history LiDAR only)'):
            block.append(item)
            i = offset[item['token']]
            if (i+1) % 512 == 0 or i+1 == len(records):
                path = paths[i//512]
                if path.exists():
                    raise FileExistsError(path)
                save_entry(path, {'tokens': [x['token'] for x in block],
                                 'history': torch.stack([x['history'] for x in block]),
                                 'ego_speed': torch.stack([x['ego_speed'] for x in block])})
                block = []
    history, speeds = read_inputs(root, records, identity['target_sha256'])
    valid = (history[..., -1] > .5).all(-1)
    enabled = (history[:, -2:, -1] > .5).all(-1)
    summary = dict(scenes=len(records), bytes=sum(p.stat().st_size for p in paths),
                   enabled_rate=float(enabled.float().mean()), all_four_valid_rate=float(valid.float().mean()))
    for split in ('train', 'val'):
        mask = torch.tensor([r['split'] == split for r in records])
        # Intersection with original GT supervision, for diagnosing sparse gradients.
        from navsim.agents.diffusiondrive.generator_timing.risk_brake_timing import _brake_timing_observations
        from navsim.agents.diffusiondrive.generator_timing.data import timing_config
        obs = _brake_timing_observations(target['trajectories'], target['trajectories'], target['contexts'], timing_config())
        summary[split] = dict(scenes=int(mask.sum()), enabled=int((enabled & mask).sum()),
                              timing_supervised=int((enabled & obs['active'] & mask).sum()))
    write_new_json(root / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def read_inputs(root, records, target_sha):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema'] != SCHEMA or manifest['sources'] != input_sources()
            or manifest['records'] != records or manifest['target_sha256'] != target_sha):
        raise ValueError('History inputs/GT token alignment or provenance differs')
    history, speeds = [], []
    for i in range(0, len(records), 512):
        part = load_torch(root / f'block_{i//512:04d}.pt')
        if part['tokens'] != [r['token'] for r in records[i:i+512]]:
            raise ValueError('History block token order differs')
        if part['history'].shape != (len(part['tokens']), 4, 12) or part['ego_speed'].shape != (len(part['tokens']),):
            raise ValueError('History block tensor shape differs')
        history.append(part['history'])
        speeds.append(part['ego_speed'])
    history, speeds = torch.cat(history), torch.cat(speeds)
    if not torch.isfinite(history).all() or not torch.isfinite(speeds).all():
        raise ValueError('Nonfinite saved timing query inputs')
    return history, speeds


class AttentionDataset(TimingDataset):
    def __init__(self, targets, candidate_cache, input_root, split, smoke=False):
        super().__init__(targets, candidate_cache, split, smoke)
        self.history, self.ego_speed = read_inputs(input_root, self.records, sha256(targets))

    def __getitem__(self, index):
        context, targets = super().__getitem__(index)
        i = self.indices[index]
        context['timing_history'] = self.history[i]
        context['timing_ego_speed'] = self.ego_speed[i]
        return context, targets


class AttentionFeatureSource(FeatureSource):
    def __init__(self, config, data_root, limit, seed):
        super().__init__(config, 'navtest', None, data_root, limit, seed)
        self.scene_loader._sensor_config.lidar_pc = [0, 1, 2, 3]

    def __getitem__(self, index):
        record = self.records[index]
        agent_input = self.scene_loader.get_agent_input_from_token(record['token'])
        features = self.builder.compute_features(agent_input)
        history, speed = risk_input(agent_input)
        features['timing_history'], features['timing_ego_speed'] = history, speed
        return record, features
