# 第十八至十九节修改审核：Dedup 与训练 Responsibility

修改日期：2026-08-26

## 1. 本轮范围

本轮只实现实施计划中的两部分：

1. 第十八节：在同一 command 内用正式 `D_traj` 对递归 Split 结果去重；
2. 第十九节：训练正样本从“静态 Anchor 与 GT 最近”改为“当前预测轨迹与 GT 最近”。

本轮没有提前实现第二十一节的 K-invariant classification reduction，也没有实现第二十二节完整的训练/推理 command mask 和推理 selector。原版 winner 后的 L1 regression 保持不变。

## 2. 最终数据流

```text
navtrain GT + command
        │
        ├── 固定 Calibration Delta/FDE scale
        │
        └── command-conditioned 初始 Bank
                    │
                    └── 递归 D_traj Split
                              │
                              └── 同 command D_traj Dedup
                                         │
                                         ├── 全量 GT 重新 assignment 一次
                                         └── command_anchor_bank.npz
                                                      │
                                                      └── 训练时按当前预测选择 winner
```

训练所需的 Anchor、command id 和固定距离尺度现在来自同一个 NPZ artifact，避免模型使用的几何原型和 Responsibility 使用的尺度来自不同文件。

## 3. Dedup 实现

新增文件：`navsim/agents/diffusiondrive/anchors/anchor_dedup.py`

判定规则严格限制在相同 command 内：

```text
duplicate(A_i, A_j) ⇔ command_i = command_j
                       且 D_traj(A_i, A_j) < tau_dedup
```

默认 `tau_dedup=0.15`，命令行只开放实施计划规定的 `0.10 / 0.15 / 0.20 / 0.25`。

每个 command 内的保留优先级为：

1. support 更大；
2. support 相同时，local mean `D_traj` 更小；
3. 两者仍相同时，保留原 Bank 中索引更小的 Anchor，保证结果可复现。

去重采用上述固定优先级逐个扫描。一个候选只要与已保留的同 command Anchor 存在 `D_traj < tau_dedup`，就被删除。不同 command 即使轨迹几何完全相同也不会互相去重。

Dedup 完成后不会沿用 Split 前的旧 assignment，而是在最终 Bank 上对全量 GT 执行一次 command-conditioned reassignment，并重新计算：

- `support`；
- 每个 Anchor 的 local mean `D_traj`；
- 每条 GT 的最终 assignment；
- 每条 GT 的最近 `D_traj`。

`command_anchor_bank_report.json` 增加去重前后数量、阈值、删除数量、来源索引、最终 support 和 local mean `D_traj`。`command_anchor_bank.npz` 增加 `dedup_source_indices` 与 `dedup_removed_count`，便于追溯每个最终 Anchor 来自 Split Bank 的哪个位置。

## 4. 训练 Responsibility 实现

修改文件：`navsim/agents/diffusiondrive/modules/multimodal_loss.py`

原逻辑使用静态 `plan_anchor` 与 GT 的距离选择正样本。新逻辑使用当前 decoder 输出的 `poses_reg`：

```text
k* = argmin D_traj(stop_gradient(current_prediction_k), GT)
     k ∈ 当前样本 command 对应的 Anchor modes
```

实现中的关键约束：

- 距离输入为当前预测轨迹的 XY 与 GT XY；
- 只允许与当前 driving command 相同的 Anchor mode 竞争；
- 使用与离线 Split、Dedup 完全相同的 command-conditioned XY 权重、Raw Delta Distance、固定 P95 scale 和 `0.2 × FDE`；
- `poses_reg.detach()` 与 `torch.no_grad()` 共同保证离散 argmin 不进入反向传播；
- 不符合 command 的 mode 距离被置为 `inf`；
- winner 确定后，仍使用原版 `F.l1_loss(best_reg, target_traj)`，本轮没有修改 regression loss。

训练特征构建器会原样保留四维 command，数据集与 DataLoader 没有保证过滤 `unknown`。因此 `LossComputer` 在进入 Responsibility 前增加 sample-level valid mask：

- Left、Straight、Right 样本正常参与 trajectory classification、responsibility 与 regression；
- `unknown` 样本不参与本版全部 trajectory supervision，但仍可参与模型的其他训练任务；
- 一个 batch 全部为 `unknown` 时返回可反向传播的零 trajectory loss；
- `unknown` 不会被映射为 Straight，避免污染 Straight Anchor 的责任分配。

## 5. 模型 artifact 接入

修改文件：

- `navsim/agents/diffusiondrive/transfuser_model_v2.py`
- `navsim/agents/diffusiondrive/transfuser_config.py`
- `navsim/agents/diffusiondrive/transfuser_agent.py`
- `navsim/agents/diffusiondrive/anchors/io.py`

`TrajectoryHead` 现在要求 `plan_anchor_path` 指向 `command_anchor_bank.npz`，并在初始化时读取：

- `anchors: [K, 8, 2]`；
- `command_ids: [K]`；
- `delta_scales: [3]`；
- `fde_scales: [3]`。

普通 NPY 只保留为可视化输出，不再允许作为训练 artifact。模型会检查 command id、shape、有限值和正尺度。

顶层模型将原始 `status_feature[:, :4]` driving command 传入 `TrajectoryHead`，再传入 `LossComputer`。Checkpoint 加载同时过滤 Anchor、command metadata 和距离尺度 buffer，避免旧 checkpoint 覆盖当前配置选择的 Bank。

当前训练返回轨迹以及推理 `forward_test` 仍保持原选择方式；完整 command mask 和推理 selector 属于第二十二节，不在本轮范围。

## 6. 文件级修改

| 文件 | 本轮作用 |
| --- | --- |
| `anchors/anchor_dedup.py` | 同 command `D_traj` 去重、保留优先级、最终全量 reassignment |
| `anchors/trajectory_distance.py` | 增加与 NumPy 正式距离一致的 Torch `D_traj` 和 command 解析 |
| `anchors/io.py` | NPZ 保存最终 Bank、command metadata、固定尺度和 Dedup 审计字段 |
| `anchors/__init__.py` | 导出 Dedup API |
| `scripts/anchors/build_anchor_bank.py` | 在递归 Split 后接入 Dedup，并开放规定阈值 |
| `modules/multimodal_loss.py` | 用当前预测和 command-local `D_traj` 选择训练 winner |
| `transfuser_model_v2.py` | 加载完整 NPZ并把 driving command 传入 loss |
| `transfuser_config.py` | 默认 Anchor 路径改为 command-conditioned NPZ |
| `transfuser_agent.py` | 防止 checkpoint 覆盖当前 Anchor metadata 与固定尺度 |
| `tests/test_anchor_dedup.py` | 覆盖同 command 去重、保留优先级、跨 command 隔离和重新 assignment |
| `tests/test_dynamic_responsibility.py` | 覆盖 current prediction winner、command mask、unknown 样本过滤、全 unknown batch、无梯度 argmin 和原版 L1 |
| `tests/test_trajectory_distance.py` | 覆盖 Torch/NumPy 正式 `D_traj` 一致性和 unknown command 拒绝 |
| `tests/test_diffusiondrive_variable_anchor.py` | 更新 NPZ 加载、元数据校验和训练接口测试 |

## 7. 建议审核点

1. `anchor_dedup.py` 是否确实只比较相同 command；
2. support 优先、local mean `D_traj` 次优先是否符合实施计划；
3. Dedup 后是否对全量 GT 只重新 assignment 一次；
4. `current_prediction_responsibility()` 是否使用 `poses_reg.detach()`，而不是静态 Anchor；
5. winner 是否只能来自当前 command modes；
6. winner 后的 regression 是否仍为原版 L1；
7. 模型是否只从完整 NPZ 读取 Anchor、command 和固定 scale；
8. 本轮是否没有提前改动 classification reduction 与推理 selector。
9. mixed/all-unknown batch 是否不会报错，且 unknown 不产生 trajectory loss 梯度。

## 8. 验证状态

本地按当前实验工作流未运行测试。新增和更新的测试已写入仓库，待本轮代码审核通过后在服务器环境执行。建议服务器至少运行：

```bash
pytest -q \
  tests/test_trajectory_distance.py \
  tests/test_anchor_dedup.py \
  tests/test_dynamic_responsibility.py \
  tests/test_diffusiondrive_variable_anchor.py
```

本轮未提交、未推送到 GitHub；保留为工作区修改，等待代码审核。
