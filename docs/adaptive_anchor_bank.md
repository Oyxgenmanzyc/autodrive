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

脚本严格复用 `scene_filter/navtrain.yaml` 的 log、token、route 和滑窗过滤逻辑，输出 shape 为 `[N, 8, 2]` 的 ego-local XY 轨迹。同时输出同顺序的 `navtrain_trajectories_commands.npy`，保存 `left / straight / right` 语义 command；NAVSIM 的 `unknown` command 在离线标定数据中直接过滤。

## 2. 标定统一轨迹距离

在正式 adaptive anchor split 之前，先离线建立约 60 个 Calibration Medoid，并保存每个 command 的固定 Delta/FDE P95 尺度：

```bash
python scripts/anchors/calibrate_trajectory_distance.py \
  --trajectories ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories.npy \
  --commands ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories_commands.npy \
  --output-dir ${NAVSIM_EXP_ROOT}/anchors/calibration_v1
```

默认实现严格遵循以下流程：

1. 总计分配 60 个 Calibration Anchor，每个 command 至少 10 个；
2. 在每个 command 内固定种子抽样，最多 10,000 条轨迹；
3. 只用 Raw Delta Distance 构造距离矩阵并运行 FasterPAM K-Medoids；
4. Calibration Medoid 必须是真实 GT 轨迹；
5. 全量 GT 按 Raw Delta Distance 找最近 medoid；
6. Delta 与 FDE 对同一个最近 medoid 计算误差，并分别统计 P95；
7. 将 6 个尺度保存到 `trajectory_distance_calibration.json`，后续保持固定。

`trajectory_distance_calibration.npz` 保存 medoid、command id、原始 GT 索引和 support，供审计与可视化使用。Calibration Bank 仅用于定义距离尺度，不是最终 Adaptive Anchor Bank。

## 3. 构建并去重 Command-conditioned Anchor Bank

正式初始 Anchor 不再沿用无法追溯 command 来源的旧全局 20 Anchor。构建脚本默认将 navtrain GT 按 Left / Straight / Right 分组，在各组内部使用固定尺度 `D_traj` 做 K-Medoids，并按 command 样本量比例分配总计 20 个初始 Anchor（每组至少 1 个）。每个 medoid 的 command 直接继承其来源 GT 分组，不做几何猜测，也不使用 Calibration Bank 充当正式 Bank。

```bash
python scripts/anchors/build_anchor_bank.py \
  --trajectories ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories.npy \
  --commands ${NAVSIM_EXP_ROOT}/anchors/navtrain_trajectories_commands.npy \
  --calibration ${NAVSIM_EXP_ROOT}/anchors/calibration_v1/trajectory_distance_calibration.json \
  --output-dir ${NAVSIM_EXP_ROOT}/anchors/adaptive_v1
```

默认流程依次执行：

1. 先按 command mask，仅在当前 GT command 对应的 Anchor 子集内用正式 `D_traj` assignment；
2. 每个 Base 统计所负责 GT 的 `P95(D_traj)`，用优先队列选择最差 cluster；
3. 仅当 `P95(D_traj) > tau_split` 时尝试 `K-Medoids(k=2, distance=D_traj)`；
4. parent 超过 4096 条时固定种子抽样 4096 条求 medoid，再将完整 parent 回分配给两个真实 GT medoid；
5. 两个 child 都达到当前 command 全部样本的 3% 才接受 split，否则保留 parent；
6. 没有 cluster 超阈值时停止，并用 `max_total_anchors=72` 防止异常增长。
7. 只在相同 command 内按 `D_traj < tau_dedup` 去重，优先保留 support 更大、其次 local mean `D_traj` 更低的 Anchor，随后重新 assignment 全量 GT。

每次接受 Split 后都会在当前全部 Anchor 上重新进行同 command global assignment。被 3% 规则拒绝的 node 仅在其 GT membership 完全不变时保持 blocked；其他 Split 导致其 membership 变化后，该 node 会自动重新进入候选评估。

`tau_split` 对应论文符号 `τ_D`，默认 1.0；命令行只开放计划规定的 0.8/1.0/1.2。`tau_dedup` 默认 0.15，开放 0.10/0.15/0.20/0.25 消融。variance、P95 max waypoint error 和 Coverage@1m 继续写入 diagnostics，但不参与 split 或 dedup。

输出文件包括：

- `command_anchor_bank.npy`：便于离线可视化的去重后 flat `[K, 8, 2]` tensor，不包含训练所需 metadata；
- `command_anchor_bank_initial.npy / _initial_commands.npy / _initial.npz`：可追溯的初始 20 Anchor、command metadata、真实 GT source index 与全量 support；
- `command_anchor_bank.npz`：模型使用的完整 artifact，包含 Anchor、command metadata、固定 Delta/FDE scale、Dedup 来源、GT assignment 和最近 `D_traj`；
- `command_anchor_bank_report.json`：配置、停止原因、逐次 Accept/Reject 记录和最终 cluster diagnostics；
- `command_anchor_bank_base_expansion.png`：`P95 D_traj / ADE / P95 max error` 随 K 的变化。
- `command_anchor_bank_diversity.png`：最终 Bank 的同 command pairwise `D_traj` 直方图。

## 4. 审核离线结果

优先检查 `command_anchor_bank_report.json`：每次 split 都记录 parent command、`P95(D_traj)`、K-Medoids 抽样数、完整 child support、3% 下限以及 Accept/Reject 原因。报告同时给出 `total_k / k_left / k_straight / k_right`、每个 Anchor support、Mean/P50/P95 nearest `D_traj`、ADE/FDE、P95 max point error 和 Coverage@0.5/1.0/1.5m。

`diversity_metrics` 只比较相同 command 的 Anchor，记录 Mean/Median nearest-neighbor `D_traj`、pairwise 分布及 `D_traj < 0.10/0.15/0.20/0.25` 的比例。不同 command 的几何近邻不会混入 Diversity 统计。

旧 `analyze_anchor_bank.py` 使用全局 ADE 最近邻，不理解 command metadata，因此不能作为本阶段 command-conditioned coverage 的正式结论；后续统一 metrics 时再接入。

## 5. 训练使用新 anchor

模型现在要求 `plan_anchor_path` 指向完整 NPZ，普通 NPY 会因缺少 command metadata 和固定尺度而被拒绝：

```bash
python navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  agent.config.plan_anchor_path=${NAVSIM_EXP_ROOT}/anchors/adaptive_v1/command_anchor_bank.npz
```

训练 winner 使用 Current Prediction 与 GT 的 `D_traj` 最近邻，并且只在当前 command modes 内竞争；argmin 在 `no_grad` 中执行，winner 后继续使用原版 L1 regression。

Classification 保留原 focal loss 的 logits、target、alpha 和 gamma，只将 reduction 改为：

```text
positive focal loss + 当前样本 valid negative focal loss mean
```

因此不同 command 的 Anchor 数量不会直接缩放分类损失。训练和推理始终保持 `[B,K,...]`，不创建动态 tensor shape；训练时 invalid command modes 不参与 classification/responsibility/regression，推理时在原 `argmax` 前把 invalid logits 置为 `-inf`。

训练数据入口没有保证过滤 `unknown`，因此 `LossComputer` 会按样本排除 unknown，使其不参与本版 trajectory supervision，也不会将其映射为 Straight；其他训练任务仍可使用这些样本。推理遇到 `unknown` 时保留全部 modes，回退到原始 logits `argmax`。

训练日志按 command 累积最终 decoder winner。每个 epoch 结束时只执行一次 `[K]` winner-count vector 的 distributed all-reduce，再从全局计数计算每个 mode 的 winner frequency、winner frequency 超过 0.1% 的 active mode rate，以及 winner entropy。日志键统一使用 `trajectory_utilization/<command>/...`；计数 buffer 不写入 checkpoint，记录完成后清零。

训练使用原版 `AgentLightningModule` 和普通 `CacheOnlyDataset`。只要数据集版本、split、trajectory sampling 和 Transfuser feature/target 配置不变，原版或 3.1.01 已生成的 `transfuser_feature` / `transfuser_target` 缓存都可复用；不要在本分支训练命令中设置 `temporal_pair_training=true`。
