# Calibration Bank 与正式 D_traj 修改审核说明

日期：2026-08-26  
目标分支：`change/20260822-adaptive-multimodal-anchor-3.1.02`  
状态：代码未提交、未推送，等待审核

## 1. 本轮范围

本轮严格实现《anchor逻辑修改实施计划》第六至第十节：

- 约 60 个 Calibration Anchor；
- command 比例分配，每组至少 10 个；
- Raw Delta Distance K-Medoids；
- 每组最多固定种子抽样 10,000 条；
- 全量 GT 统计 Delta/FDE P95；
- Delta 与 FDE 必须参考同一个 delta-nearest medoid；
- 保存六个固定尺度；
- 使用固定尺度形成正式 `D_traj`。

本轮没有修改 Split Trigger、正式 Anchor Split、Dedup、训练 responsibility、classification loss、regression loss 或推理 selector。

## 2. 修改文件

| 文件 | 作用 |
|---|---|
| `navsim/agents/diffusiondrive/anchors/trajectory_distance.py` | 增加 NAVSIM command 映射、固定尺度对象和正式 `trajectory_distance()` 入口 |
| `navsim/agents/diffusiondrive/anchors/calibration.py` | Calibration Bank、分配、抽样、K-Medoids、P95 统计及 JSON/NPZ 读写 |
| `scripts/anchors/extract_navsim_trajectories.py` | 在轨迹之外同步保存语义 command，并过滤 unknown |
| `scripts/anchors/calibrate_trajectory_distance.py` | 独立离线标定入口 |
| `requirements.txt` | 增加 `kmedoids==0.5.5` |
| `tests/test_trajectory_distance.py` | 增加 one-hot command 与正式 D_traj 测试 |
| `tests/test_anchor_calibration.py` | 增加分配、同 medoid P95、真实 medoid、保存/加载测试 |
| `docs/adaptive_anchor_bank.md` | 增加实际标定流程说明 |
| `CHANGELOG.txt` | 记录本阶段实验修改 |

## 3. 关键源码片段

### 受 minimum 约束的 60-anchor 分配

```python
allocations = allocate_command_anchors(
    config.total_anchors,
    command_counts,
    config.min_per_command,
)
```

### Raw Delta Distance K-Medoids

```python
distance_matrix = pairwise_raw_delta_distance(
    sampled_trajectories,
    sampled_trajectories,
    command,
    batch_size=config.distance_batch_size,
)
kmedoids_result = kmedoids.fasterpam(
    distance_matrix,
    allocations[command],
    init="random",
    random_state=config.random_seed + command_id,
)
```

### Delta 与 FDE 复用同一个 medoid

```python
local_assignments = distances.argmin(axis=1)
delta_errors[start:end] = distances[np.arange(end - start), local_assignments]
fde_errors[start:end] = raw_fde_distance(
    trajectories[start:end],
    anchors[local_assignments],
    command,
)
```

### 正式 D_traj

```python
return (
    delta_distance / (delta_scale + 1e-6)
    + 0.2 * fde_distance / (fde_scale + 1e-6)
)
```

## 4. 数据流

```text
NAVSIM navtrain frame
        ↓
ego-local GT [N, 8, 2]
+ semantic command [N]
        ↓
过滤 unknown，分为 Left / Straight / Right
        ↓
按样本比例分配总计 60 个 medoid
每组至少 10 个
        ↓
每组固定种子抽样，最多 10k
        ↓
Raw Delta Distance 矩阵
        ↓
FasterPAM K-Medoids
        ↓
真实 GT Calibration Medoid
        ↓
该 command 全量 GT 找 delta-nearest medoid
        ↓
对同一个 medoid 记录 Raw Delta 与 Raw FDE
        ↓
分别取 P95，保存 6 个固定尺度
        ↓
正式 D_traj
```

## 5. 60 个 Anchor 的分配

`allocate_command_anchors()` 使用带下限约束的比例分配和 largest remainder rounding：

```text
总数固定为 60
Left / Straight / Right 每组至少 10
剩余数量按 command 样本量比例分配
小数余数从大到小补齐，最终严格等于 60
```

例如样本比例为 `15% / 70% / 15%` 时，输出为：

```text
Left      10
Straight  40
Right     10
```

## 6. Calibration K-Medoids

Calibration 阶段没有 normalization scale，因此只使用已经确定的主项：

```text
D_cal = Raw Delta Distance
```

其性质为：

- 8 个 Delta，包含 `p0=(0,0) → p1`；
- 8 个 timestep 等权；
- Straight 使用 `x:y=0.6:0.4`；
- Left/Right 使用 `x:y=0.4:0.6`；
- 不加 FDE；
- 不做 normalization；
- 不加入 heading、curvature、jerk、DTW 或 Hausdorff。

实现使用 `kmedoids.fasterpam()` 和预计算的 Raw Delta Distance 矩阵。最终 medoid 索引直接回到原始 GT，因此 Calibration Anchor 一定是真实轨迹，不是均值中心。

## 7. 计算固定 P95 Scale

对每条 GT，先确定唯一参考 medoid：

```text
j* = argmin RawDelta(GT, calibration_medoid_j)
```

然后对同一个 `j*` 同时计算：

```text
e_delta = RawDelta(GT, medoid_j*)
e_fde   = RawFDE(GT, medoid_j*)
```

禁止为 FDE 重新寻找 endpoint-nearest medoid。最后每个 command 分别统计：

```text
delta_scale = P95(e_delta)
fde_scale   = P95(e_fde)
```

最终得到 `Left / Straight / Right × Delta/FDE` 共 6 个固定值。

## 8. 正式 D_traj

正式入口为 `trajectory_distance()`，只接受离线加载的 command scale：

```text
D_traj
= RawDelta / (delta_scale(command) + 1e-6)
+ 0.2 × RawFDE / (fde_scale(command) + 1e-6)
```

Scale 从 `trajectory_distance_calibration.json` 加载，不在 batch、epoch、Split 后重新计算，也不是 learnable parameter。

## 9. 输出文件

标定脚本生成：

### `trajectory_distance_calibration.json`

保存：

- 三个 command 的 trajectory count；
- 分配的 Calibration Anchor 数量；
- 实际参与 K-Medoids 的抽样数；
- 6 个固定 P95 scale；
- 每个 medoid 的全量 GT support。

### `trajectory_distance_calibration.npz`

保存：

- `anchors [60,8,2]`；
- `command_ids [60]`；
- `source_indices [60]`；
- `support [60]`；
- command counts、anchor counts、sampled counts；
- Delta/FDE scale 数组。

## 10. 工程保护

- 输入轨迹必须为有限的 `[N,8,2]`；
- command 必须覆盖 Left、Straight、Right；
- 总 Anchor 数必须满足每组 minimum；
- 分配数量不能超过对应 command 的样本或抽样数量；
- P95 scale 必须有限且大于 0；
- pairwise 距离按行分块计算，避免额外产生完整四维广播张量；
- K-Medoids 只处理每组最多 10k 抽样，全量数据仅做 `O(N×K)` assignment。

注意：10k 样本的 float32 方阵本身约占 400 MB；这是实施计划允许的 Calibration 离线上限，不会出现在训练或推理阶段。

## 11. 测试覆盖

已新增单元测试，覆盖：

1. `15/70/15 → 10/40/10` 的 minimum-constrained 分配；
2. Delta 与 FDE 使用同一个 delta-nearest medoid；
3. Calibration Anchor 确实来自真实 GT；
4. JSON/NPZ 保存与固定尺度加载；
5. NAVSIM `[left, straight, right, unknown]` one-hot 映射；
6. 正式 `trajectory_distance()` 使用对应 command 的固定 scale。

按照本仓库本地实验工作流，本轮未运行本地测试，等待服务器 smoke test。

## 12. 外部实现依据

- NAVSIM 官方文档说明 driving command 包含 Left、Straight、Right 与可过滤的 Unknown：<https://github.com/autonomousvision/navsim/blob/main/docs/agents.md>
- FasterPAM 使用任意预计算 dissimilarity matrix，并返回真实 medoid 索引：<https://github.com/kno10/python-kmedoids>
