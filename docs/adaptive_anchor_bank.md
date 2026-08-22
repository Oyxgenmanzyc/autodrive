# NAVSIM Adaptive Anchor Bank

本实现把 anchor 学习与 DiffusionDrive 训练彻底解耦。第一版固定使用 `navtrain` 的 GT 轨迹离线生成 anchor，训练和推理期间不更新 anchor。

本分支以原版 DiffusionDrive 为模型基线，只修改 anchor 的离线生成、加载和候选数量 K。它不包含 temporal-pair 数据集、GT-protected gradient、temporal energy/ranking loss、跨帧历史轨迹或 temporal rescore。

## 1. 提取 navtrain GT

在已配置 `OPENSCENE_DATA_ROOT` 的 NAVSIM 环境中执行：

```bash
python scripts/anchors/extract_navsim_trajectories.py \
  --output ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories.npy
```

也可以显式指定日志目录：

```bash
python scripts/anchors/extract_navsim_trajectories.py \
  --data-path /path/to/openscene/navsim_logs/trainval \
  --output /path/to/navtrain_trajectories.npy
```

脚本严格复用 `scene_filter/navtrain.yaml` 的 log、token、route 和滑窗过滤逻辑，输出 shape 为 `[N, 8, 2]` 的 ego-local XY 轨迹。

## 2. 构建 anchor bank

先准备 DiffusionDrive 官方的 `kmeans_navsim_traj_20.npy`，再执行：

```bash
python scripts/anchors/build_anchor_bank.py \
  --trajectories ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories.npy \
  --initial-anchors /path/to/kmeans_navsim_traj_20.npy \
  --output-dir ${NAVSIM_EXP_ROOT}/anchors/adaptive_v1 \
  --max-base-anchors 64 \
  --residual-modes 8
```

默认流程依次执行：

1. 使用 ADE 做全局 nearest-anchor assignment；
2. 根据 P95 max error、cluster variance、support 和 uncovered count 选择 split；
3. 对候选 cluster 做 local K-Means(K=2)，children 继承原始 `root_id`；
4. 按 root 学习 tangent-normal residual codebook，并显式保留 zero residual；
5. 使用 per-base support mask 过滤组合，最后按 ADE 去重。

`residual_modes` 表示每个 root 的总 residual mode 上限，包含 zero residual。所有阈值均可通过 `--help` 查看和覆盖。

输出文件包括：

- `adaptive_anchor_bank.npy`：模型直接加载的 `[K_final, 8, 2]` anchor；
- `adaptive_anchor_bank_base.npy`：只包含 Expanded Base 的消融实验 anchor；
- `adaptive_anchor_bank.npz`：base tree、root、residual、support 和 composition metadata；
- `adaptive_anchor_bank_report.json`：三阶段 coverage/ADE、split history 和 distortion curve；
- `adaptive_anchor_bank_base_expansion.png`：coverage/ADE/P95 error 随 base K 的变化。

## 3. 先做离线对比

```bash
python scripts/anchors/analyze_anchor_bank.py \
  --trajectories ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories.npy \
  --anchor-bank Original=/path/to/kmeans_navsim_traj_20.npy \
  --anchor-bank Base=${NAVSIM_EXP_ROOT}/anchors/adaptive_v1/adaptive_anchor_bank_base.npy \
  --anchor-bank Adaptive=${NAVSIM_EXP_ROOT}/anchors/adaptive_v1/adaptive_anchor_bank.npy \
  --output ${NAVSIM_EXP_ROOT}/anchors/adaptive_v1/comparison.json
```

确认 adaptive bank 的 Coverage@1m、mean nearest ADE 和 P95 max error 确有改善后，再启动训练。

## 4. 训练使用新 anchor

通过 Hydra 覆盖配置中的路径：

```bash
python navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  agent.config.plan_anchor_path=${NAVSIM_EXP_ROOT}/anchors/adaptive_v1/adaptive_anchor_bank.npy
```

模型会从文件 shape 自动推导 K。加载旧 checkpoint 时不会恢复 checkpoint 内的 20-anchor tensor，因此不会覆盖当前配置选择的新 bank。计算量和显存仍近似随 K 线性增长，第一轮建议依次验证 K=40/64、80/128，而不是直接使用数百个候选。

训练使用原版 `AgentLightningModule` 和普通 `CacheOnlyDataset`。只要数据集版本、split、trajectory sampling 和 Transfuser feature/target 配置不变，原版或 3.1.01 已生成的 `transfuser_feature` / `transfuser_target` 缓存都可复用；不要在本分支训练命令中设置 `temporal_pair_training=true`。
