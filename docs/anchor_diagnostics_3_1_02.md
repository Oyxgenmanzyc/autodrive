# 新 3.1.02：原算法 + 只读诊断

基底：`0c18ceb`，原 `change/20260822-adaptive-multimodal-anchor-3.1.02`。
新分支：`change/20260905-adaptive-anchor-diagnostics-3.1.02`。
诊断来源：`3.1.02_2` 的 anchor utilization、coverage、diversity 设计，重新适配原版 ADE 与普通 NPY bank。

本版结束当前消融迁移工作，只带回观测工具。原版约 88.44 的 PDM 成绩属于已有实验，并非新分支已经完成评测的成绩。

## 迁移边界

| 内容 | 本版处理 |
| --- | --- |
| Base expansion + residual composition | 原 3.1.02 原样保留 |
| NPY K67 bank、静态 ADE 匹配 | 原样保留；仍使用原 adaptive_v1 bank |
| Focal mean、L1、loss weights | 原样保留 |
| 模型结构、DDIM、推理 top1 | 原样保留 |
| optimizer、scheduler、LR 倍率行为 | 原样保留，不夹带 E1 的修改 |
| 数据集、训练缓存、metric cache | 构造逻辑原样保留 |
| 使用统计、支持度、近重复率、可视化 | 迁回并适配原版；观测不反馈给模型 |
| command 分组限制、D_traj、动态 WTA、K-invariant、child ratio | 不迁回 |

训练统计默认关闭，只有 config 和 callback 注册入口有两处原文件修改。模型、loss、生成器源码保持基底原样。开启统计仍有额外计算、CPU 传输和文件 I/O 开销，不声称 wall time 完全不变。

## 使用已有 checkpoint 或继续原实验

继续使用**原 3.1.02** 的 checkpoint 和原 NPY bank：

```text
/home/hndx/navsim_workspace/exp/anchors/adaptive_v1/adaptive_anchor_bank.npy
```

不要传入 `3.1.02_2` 的 command NPZ bank，也不要携带其 `use_command_mask` / `trajectory_cls_reduction` 参数。原 checkpoint 结构保持兼容；诊断 callback 没有模型参数、buffer 或 checkpoint 状态。

不需要为了这些诊断功能重新生成原 3.1.02 训练缓存或 metric cache。不要把两版训练产物混用。本版默认不启动训练或评测。

## 可选训练统计

原训练命令额外添加：

```bash
+agent.config.anchor_diagnostics=true
```

默认 `false` 不注册 observer，也不导入其实现。启用后：

- 只观察当前训练 batch 的既有 LossComputer 输出；不额外运行模型，不增加随机采样。
- 用 `detach` / `no_grad` 重算诊断标量；hook 返回 `None`，不替换原 loss。
- 每个 decoder 层单独计数；层编号从 0 开始，最高编号为最终 decoder，不能将各层计数相加当作样本数。
- `winner_counts` 是 GT 对固定 anchor 的静态 ADE 匹配；`selected_counts` 是当前 logits top1。前者有覆盖、后者长期未被选，才是值得关注的选择问题，不能混为一谈。
- active rate 定义为频率严格大于 0.1% 的 anchor 比例；同时给出实际非零计数和归一化熵。
- `trajectory_cls_loss` / `trajectory_reg_loss` 已乘各自权重，但尚未乘外层 `trajectory_weight=12`；以有限 batch 样本数加权平均。
- 发现 NaN/Inf 时只增加 `nonfinite_batches`，原训练输出原样保留；该 batch 不参与诊断均值及频率，明确报告排除数量。
- epoch 文件记录当时实际 optimizer group LR，不修改 scheduler。
- 不调用 `self.log(sync_dist=...)`、all_reduce、barrier、all_gather、broadcast 等额外同步。训练进程只写自己的 JSON。

输出位置为 Trainer 的 `default_root_dir`（默认配置对应本次实验输出目录）下：

```text
anchor_diagnostics/
  epoch_000_rank_0.json
  epoch_000_rank_1.json
  epoch_000_rank_2.json
  epoch_000_rank_3.json
  ...
  epoch_099_rank_3.json
```

文件已存在时保留旧文件并打印警告；重新执行同一 epoch 请使用新的实验输出目录。日志文件无需从 tmux 滚动区复制。每张卡的文件是 rank-local，**不能直接称为全局频率或全局熵**。

训练结束后离线合并某个 epoch：

```bash
python scripts/anchors/merge_anchor_diagnostics.py \
  --input-dir "/实际实验时间目录/anchor_diagnostics" \
  --epoch 99 \
  --output "/实际实验时间目录/anchor_diagnostics/epoch_099_global.json"
```

合并要求同一 run、epoch、bank hash，且每个 rank 恰好一份文件；缺少任意 rank 即报错。先合并计数和加权和，再计算全局熵及均值，不能简单平均各卡熵。统计口径包含 DDP sampler 补齐的训练抽样，不等同去重后的 dataset token 总数。

## 离线 anchor 可视化与覆盖/重复诊断

不需要 GPU。在新代码目录中，确保 `PYTHONPATH` 包含新代码根目录，然后执行：

```bash
python scripts/anchors/diagnose_anchor_bank.py \
  --trajectories /home/hndx/navsim_workspace/exp/anchors/navtrain_trajectories.npy \
  --anchor-bank /home/hndx/navsim_workspace/exp/anchors/adaptive_v1/adaptive_anchor_bank.npy \
  --assignment-batch-size 4096 \
  --output-dir "/home/hndx/navsim_workspace/exp/anchors/diagnostics_3_1_02_$(date +%Y%m%d_%H%M%S)" \
  --plot
```

输出：`anchor_diagnostics.json`、`anchor_support.csv`、`anchor_overview.png`、`anchors_individual.png`、`anchor_support_diversity.png`。不加 `--plot` 时不导入 matplotlib。输出目录必须是新目录，避免覆盖已有实验结果。

全部距离以原版 ADE（米）计算，不是归一化 D_traj，不能直接与 3.1.02_2 的 D_traj 阈值横向等同。覆盖率沿用原 3.1.02 定义 `min_anchor(max_t(error)) <= epsilon`；另列 ADE winner 的 FDE / 最大点误差，二者不混名。近重复比例只是报告，不删除任何 anchor。图中逐条轨迹共用坐标尺度，避免把短轨迹放大后误判为长轨迹。

## 服务器验证（本地未运行）

先在服务器原 navsim 环境运行新增测试：

```bash
python -m pytest -q tests/test_anchor_diagnostics.py tests/test_anchor_diagnostics_callback.py
```

原 variable-K 测试会向 `sys.modules` 注入 stub，放到单独进程中运行：

```bash
python -m pytest -q tests/test_adaptive_anchor_bank.py tests/test_diffusiondrive_variable_anchor.py
```

服务器跨两轮 DDP smoke test：在**原 3.1.02 完整训练指令副本**中使用独立实验名，设置 `+agent.config.anchor_diagnostics=true`、`trainer.params.max_epochs=2`、`trainer.params.limit_train_batches=10`、`trainer.params.limit_val_batches=2`；保留原物理卡选择与原环境。结束后检查两个 epoch 都各有四个 rank 文件，运行离线合并。正式训练恢复原训练轮数和 batch 上限。

本地未运行单元测试、语法检查、smoke test 或 PDM。新增测试检查启用 observer 前后相同输入的 loss、梯度、RNG、state_dict 一致，以及无 collective、非有限值记录和多卡文件合并。启用统计后的服务器表现仍需以上验证。
