# 第十九至二十五节最终修改审核

修改日期：2026-08-26

## 1. 最终实现范围

本轮完成实施计划第十九至二十五节：

1. Current Prediction Responsibility；
2. 保持原版 Regression Loss；
3. K-invariant focal reduction；
4. 训练与推理 Command Mask；
5. Anchor Builder、Split、Coverage 日志；
6. Diversity 日志；
7. Training Utilization 日志。

同时按第二十八至三十一节约束模块边界，没有引入新 regression/heading/curvature loss、风险或进度 selector、时序模块、在线 Anchor growth、可学习阈值或动态 P95。

## 2. 训练损失闭环

### 2.1 Responsibility

正样本由当前 decoder 预测决定：

```text
Current Predictions
→ 仅保留当前 command modes
→ D_traj(prediction.detach(), GT)
→ no_grad argmin
→ winner
```

静态 `plan_anchor ↔ GT` 不再参与 positive mode 分配。

### 2.2 Regression

`D_traj` 只决定 winner。winner 确定后继续调用原版：

```python
F.l1_loss(best_reg, target_traj)
```

没有新增或修改 regression loss。

### 2.3 K-invariant Classification

保留原 focal loss 的 logits、one-hot target、`alpha=0.25` 和 `gamma=2.0`，只修改 reduction：

```text
L_cls = mean(L_positive)
      + mean_per_sample(mean(L_valid_negative))
```

negative 先在每个样本的 active command 内取均值，再在 batch 内取均值。因此 `1 positive + 10 negatives` 与 `1 positive + 30 negatives` 不会仅因 mode 数量不同而产生不同的 loss scale。

不同 command 的 invalid modes 完全不进入 focal loss。

## 3. Command Mask

tensor shape 始终保持 `[B,K,...]`，所有 Anchor 继续共同通过 decoder。

训练阶段：

- Responsibility 只比较 command-valid modes；
- Classification 只统计 command-valid positive/negative；
- Regression 只 gather winner；
- invalid modes 不产生 trajectory loss 梯度。

推理阶段：

```text
masked_logits = logits.masked_fill(~command_mask, -inf)
winner = masked_logits.argmax()
```

原 selector 没有重新设计，只增加 eligibility mask。

训练入口未保证过滤 `unknown`，因此 unknown 样本不参与 trajectory classification、responsibility 或 regression。推理遇到 unknown 时允许全部 modes，回退到原 selector；不会把 unknown 映射成 Straight。

## 4. Anchor Builder、Split 与 Coverage 日志

`command_anchor_bank_report.json` 现在包含：

- `total_k / k_left / k_straight / k_right`；
- 每个最终 Anchor 的 support；
- 每次 Split 的 parent id、command、parent support、P95 `D_traj`；
- child support、3% 下限、Accept/Reject 和 rejection reason；
- Mean/P50/P95 nearest `D_traj`；
- Mean nearest ADE/FDE；
- P95 max point error；
- Coverage@0.5m、Coverage@1m、Coverage@1.5m。

旧几何指标只作为 diagnostics，不参与 Split 或 Dedup decision。

## 5. Diversity 日志

新增 `anchors/anchor_metrics.py`，所有正式距离仍调用统一 `D_traj`，没有复制距离公式。

Diversity 仅在相同 command 内计算，报告：

- Mean nearest-neighbor Anchor `D_traj`；
- Median nearest-neighbor Anchor `D_traj`；
- pairwise `D_traj` 的 min/P25/P50/P75/P95/max 与直方图；
- `D_traj < 0.10 / 0.15 / 0.20 / 0.25` 的比例；
- 上述指标的全局汇总和 Left/Straight/Right 分组结果。

构建脚本额外生成 `command_anchor_bank_diversity.png`。

## 6. Training Utilization 日志

只使用最终 decoder 的 winner，按 command 累积：

- 每个全局 mode index 的 winner frequency；
- winner frequency `> 0.1%` 的 active mode rate；
- winner entropy。

日志键示例：

```text
train/trajectory_utilization/left/winner_mode_3_frequency
train/trajectory_utilization/left/active_mode_rate
train/trajectory_utilization/left/winner_entropy
```

计数器是运行期非持久 `[K]` buffer，不写入 checkpoint。每个 rank 在 epoch 内累计本地 winner counts；epoch end 时一次性 all-reduce 完整 `[K]` vector，再从严格的 global counts 计算 frequency、active mode rate 和 entropy。所有 utilization 指标只记录一次 epoch-level 全局结果，随后各 rank 同步清零计数器。

## 7. 模块边界

| 模块 | 单一职责 |
| --- | --- |
| `trajectory_distance.py` | 唯一正式轨迹距离、Torch 等价实现、command 解析 |
| `calibration.py` | Calibration Medoid 与固定 P95 scale |
| `kmedoids_splitter.py` | K-Medoids、采样上限和完整回分配 |
| `command_anchor_bank.py` | command metadata、priority queue、Split 更新和 cap |
| `anchor_dedup.py` | 同 command `D_traj` Dedup |
| `anchor_metrics.py` | Coverage 与 Diversity 审计指标 |
| `io.py` | NPY/NPZ/JSON artifact 序列化 |
| `multimodal_loss.py` | Responsibility、Command Mask、K-invariant focal、utilization |
| `agent_lightning_module.py` | epoch end 触发一次 `[K]` 全局归并并写出 utilization 日志 |

实施计划建议的 `anchor_io.py` 在本仓库沿用已有 `io.py` 文件名，避免无收益重命名；职责保持一致。

## 8. 第一版配置落点

实施计划第二十九节的默认值没有另建一套重复 YAML，而是落在对应模块的唯一配置入口：

- Calibration：`CalibrationConfig(60, min_per_command=10, max_kmedoids_samples=10000)`；
- Distance：`NUM_TIMESTEPS=8`、`FDE_WEIGHT=0.2` 与 command XY weights；
- Split：`CommandAnchorBankConfig(tau_split=1.0, child_min_ratio=0.03, max_split_medoid_samples=4096, max_total_anchors=72)`；
- Dedup：构建脚本默认 `tau_dedup=0.15`；
- Responsibility/Loss：Current Prediction、command-only、原版 regression、K-invariant focal 均为本分支固定第一版逻辑。

离线参数仍可通过两个 Anchor 脚本的显式命令行参数做计划规定的消融。没有增加未被训练入口消费的装饰性配置。

## 9. 明确没有实现

- 新 regression loss；
- heading loss、curvature loss；
- risk/progress score；
- collision-aware 或 hierarchical selector；
- temporal module；
- BEV-conditioned residual generator；
- learnable split threshold 或 normalization；
- online Anchor growth；
- 每 batch 动态统计 P95；
- 动态 command tensor shape。

## 10. 测试覆盖与状态

新增或更新的单元测试覆盖：

- K-invariant focal 不随 valid negative 数量变化；
- 单 valid mode、零 negative 时 focal loss 保持有限且可反向传播；
- invalid command logits 不产生分类梯度；
- 训练与推理 command selector；
- unknown mixed/all-unknown batch；
- winner histogram、active mode rate、entropy；
- DDP utilization 只 all-reduce 一次完整 `[K]` count vector；
- Coverage 与 Diversity 指标；
- variable-K NPZ 模型接口。

本地按实验工作流未执行测试。代码审核通过后，建议服务器执行：

```bash
pytest -q \
  tests/test_trajectory_distance.py \
  tests/test_anchor_calibration.py \
  tests/test_command_anchor_bank.py \
  tests/test_anchor_dedup.py \
  tests/test_anchor_metrics.py \
  tests/test_dynamic_responsibility.py \
  tests/test_diffusiondrive_variable_anchor.py
```

本审核文档随 `3.1.02_2` 代码版本一并提交。

## 11. `3.1.02_2` 最终审核修正

根据最终审核意见，Training Utilization 已从“各 rank 分别计算 entropy/active rate 后同步标量”改为：

```text
rank-local [K] winner_counts
→ epoch end
→ 一次 all_reduce(SUM) 整个 [K] vector
→ global frequency / active mode rate / entropy
→ epoch log
→ reset local counts
```

因此不再使用 `mean(H(rank_i))` 近似全局 entropy。约 60 维 count vector 每 epoch 只同步一次。

同时增加单 valid mode 边界测试，确认 `0 negative` 时 negative mean 明确定义为 0，不产生 `NaN`。
