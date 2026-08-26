# Command-conditioned Anchor Bank 与递归 Split 修改审核

对应实施计划：**十一、Command-conditioned 初始 Anchor Bank** 至 **十七、递归 Split 的执行策略**。

## 1. 本次修改结果

本次已将正式 Base Expansion 主链改为：

```text
flat Anchor Bank + command_ids
        ↓ command mask
同 command 内按 D_traj assignment
        ↓
每个 Base 统计 P95(D_traj)
        ↓ worst-first priority queue
K-Medoids(k=2, distance=D_traj)
        ↓
完整 parent 回分配
        ↓
两个 child 均满足 3% command support
        ↓
Accept / Reject
```

停止条件是没有未处理 cluster 满足 `P95(D_traj) > tau_split`；另保留 `max_total_anchors=72` safety cap。

## 2. 文件级修改

### `navsim/agents/diffusiondrive/anchors/command_anchor_bank.py`

- 定义正式 flat Bank 数据：`anchors / command_ids / support / parent_ids / root_ids / node_ids`；
- `command_conditioned_assignment()` 在计算 argmin 前先做 command mask；
- `P95(D_traj)` 是唯一 split trigger；
- 使用 priority queue 每次处理 P95 最大的 cluster；
- 每次 Accept 后会重新执行全 Bank、同 command global assignment；
- 被 3% 规则拒绝的 node 仅在 GT membership 不变时保持 blocked，membership 变化后重新开放；
- variance、P95 max waypoint error、Coverage@1m 只进入 diagnostics；
- 固定 `tau_split`（论文符号 `τ_D`）默认 1.0，目标数量不写成硬性 60。

### `navsim/agents/diffusiondrive/anchors/kmedoids_splitter.py`

- 所有距离调用统一的 `trajectory_distance()`，没有复制 Delta/FDE 公式；
- 用 `K-Medoids(k=2)` 取两条真实 GT medoid，不再生成 KMeans 均值中心；
- parent 超过 `max_split_medoid_samples=4096` 时固定种子抽样；
- medoid 产生后对完整 parent 做 `D_traj` 回分配，child support 不是抽样 support。

### `scripts/anchors/build_anchor_bank.py`

构建入口现在要求：

- GT trajectory；
- GT command；
- Calibration JSON 中的固定 P95 scale。

脚本会先从 GT 自动重建正式初始 20 Anchor：按 command 分组、按样本数比例分配名额（每组至少 1 个），再在各组内部使用正式 `D_traj` K-Medoids。初始 medoid 的 command 直接继承其来源 GT 分组，同时保存 source index 和全量 support。

命令行 `tau_split` 仅开放实施计划规定的 `0.8 / 1.0 / 1.2`，默认 1.0。当前审核批次只输出 Base Bank，不提前接入后续 Dedup、Residual 或 prediction responsibility。

### `navsim/agents/diffusiondrive/anchors/io.py`

新增本阶段专用输出：

```text
command_anchor_bank.npy
command_anchor_bank.npz
command_anchor_bank_report.json
command_anchor_bank_base_expansion.png
command_anchor_bank_initial.npy
command_anchor_bank_initial_commands.npy
command_anchor_bank_initial.npz
```

其中 NPZ 保存审核所需的全部 metadata、GT assignment 与最近 `D_traj`。

### `tests/test_command_anchor_bank.py`

新增以下逻辑测试：

1. command mask 在距离 argmin 前生效；
2. 初始 Anchor 在各 GT command 组内生成并继承可追溯标签；
3. membership 不变时保持 blocked，global assignment 改变 membership 后解除 blocked；
4. worst-first 优先拆分 P95 最大的 command cluster；
5. child medoid 必须来自真实 GT，且 tree metadata 正确继承；
6. 大 parent 的 medoid 输入受采样上限约束，但 support 来自完整 parent；
7. 任一 child 少于 command 总样本的 3% 时拒绝；
8. Anchor 总数达到 safety cap 时停止。

## 3. 针对审核意见的修正

### 3.1 初始 Anchor command 来源

不再要求人工提供 `initial_anchor_commands.npy`，也不再尝试给旧全局 20 Anchor 事后贴标签。默认从 navtrain GT 重新生成 command-conditioned 初始 20 Anchor，标签由 Left / Straight / Right GT 分组天然继承。Calibration Bank 仍然只提供固定尺度，不参与正式初始 Anchor 候选。

### 3.2 blocked 与 global reassignment

当前算法在每次 Accept 后确实重新进行全 Bank、同 command global assignment。因此 blocked 状态现在保存的是 Reject 当时该 node 的完整 GT index membership：

```text
membership 完全相同 → 保持 blocked，避免无限重试
membership 发生变化 → 清除 blocked，允许重新计算 P95 并尝试 Split
```

这样既不会永久错过 assignment 改变后的有效 Split，也不会在 Bank 未变化时反复运行同一个失败的 K-Medoids。

## 4. 建议审核点

- `command_conditioned_assignment()` 是否彻底阻断跨 command 竞争；
- 初始 medoid 的 `source_indices` 是否都来自相同 command GT；
- split 判断是否只读取 `p95_d_traj > tau_split`；
- `split_parent_cluster()` 是否只调用统一 `D_traj`，且 medoid 是真实样本；
- 4096 抽样后是否确实对完整 parent 回分配；
- 3% 分母是否为当前 command 的全量训练样本 `N_c`；
- Reject 后 parent 是否保留，membership 不变时是否阻止重试、变化后是否重新开放；
- 输出 NPZ 是否完整包含 `command_ids / support / parent_ids / root_ids`。

## 5. 验证状态

遵循本项目本地实验工作流，本次未在本机运行单元测试或训练。测试代码已补齐，待审核通过后在服务器环境执行。
