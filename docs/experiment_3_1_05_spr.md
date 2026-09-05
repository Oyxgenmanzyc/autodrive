# 3.1.05：场景条件候选重构（SPR）

基线：`change/20260905-adaptive-anchor-diagnostics-3.1.02`，
提交 `ddaafbc35785b5f4e2d2b03d5b56393b6c3e6096`。
实验分支：`change/20260905-scene-proposal-reconstruction-3.1.05`。

## 迁移内容

候选 anchor → 原有 DDIM / 两层 refinement → K 条候选 →
SPR 与 ego/agent query 交互 → 重构最终轨迹。

SPR 的全称为 Scene-conditioned Proposal Reconstruction。核心网络和坐标工具
迁移自 MeanFuser，来源、固定版本和修改边界见
[来源记录](../third_party/meanfuser/NOTICE.md)。
论文应表述为“借鉴并适配 MeanFuser 的重构模块”，不能将改名后的架构宣称为原创。

保留原作者的三层 TransformerDecoder、MLP 宽度、初始化、位移归一化统计，
以及 `embedded_vocab + norm(cross_attn_output)` 的顺序。
附件中的示意代码省略了归一化并改动了网络；本次以固定版本的真实源码为准。

本地适配仅包括动态 K、绝对轨迹转换、ego 与 agent memory、训练/推理接入、
对照指标和 checkpoint 检查。没有迁移 MeanFlow、GMN，也没有修改 anchor
构建算法、DDIM、候选分类/回归标签或感知任务。

## 配置与损失

基础 `diffusiondrive_agent` 默认关闭 SPR，保持旧预测路径。
选择 `agent=diffusiondrive_spr_agent` 才启用本实验。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| spr_enabled | true（SPR 配置） | 创建并训练重构头 |
| spr_loss_weight | 2.0 | 加入 trajectory_loss 内部的重构权重 |
| spr_detach_proposals | true | 候选坐标停止接收重构梯度；场景 memory 仍接收梯度 |
| spr_output | reconstruction | 最终输出；selector 可切换同权重的分类选择结果 |
| anchor_diagnostics | true（SPR 配置） | 沿用 3.1.02 的训练诊断 |

训练损失为：

`total = 原有总损失 + trajectory_weight × spr_loss_weight × L1(delta_pred, delta_gt)`

delta 包含作者原始归一化的 Δx、Δy 和 sin(yaw)、cos(yaw)。
默认外层 trajectory_weight=12，因此新增项的实际总权重为 24。
这与 MeanFuser 总损失中的独立权重 2 不同；该值是本次集成设置，未经调参验证。
日志中的 spr_loss 是外层乘 12 前的值。未添加额外 PDM 代理损失。

训练直接复用本次带噪训练 forward 的最后一层候选，推理复用最后一次 DDIM
输出，均只执行一次重构。训练和推理候选分布仍有差异；尚未为重构训练额外
运行完整 DDIM，这是减少自写代码与训练开销的取舍。

重构损失默认不回传到候选坐标；它会更新 SPR 和共享场景编码器，
因此并不是冻结整个 DiffusionDrive。原有分类与回归损失继续训练候选生成器。

## 服务器运行

服务器使用现有 NAVSIM / DiffusionDrive 环境和数据；本地未执行测试、训练或 PDM。
本项目是 Python 工程，通用 AGENTS.md 中的 Maven 命令不适用。

```bash
git fetch origin
git switch --track origin/change/20260905-scene-proposal-reconstruction-3.1.05
export NAVSIM_DEVKIT_ROOT="$PWD"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export ANCHOR_PATH=/absolute/path/to/the_baseline_anchor_bank.npy
export BKB_PATH=/absolute/path/to/pytorch_model.bin
# 继续使用现有 NAVSIM 数据、地图、缓存环境变量。
# 可选：使用 3.1.02 checkpoint 热启动；不是恢复 optimizer/epoch。
export INIT_CHECKPOINT=/absolute/path/to/baseline_3_1_02.ckpt

python -m pytest tests/test_scene_proposal_reconstruction.py tests/test_diffusiondrive_variable_anchor.py
bash scripts/training/run_diffusiondrive_spr_training.sh \
  trainer.params.fast_dev_run=true +trainer.params.devices=1 \
  trainer.params.strategy=auto dataloader.params.batch_size=2

# 正式训练的 epoch、batch size、学习率和数据划分需与基线对齐。
bash scripts/training/run_diffusiondrive_spr_training.sh

export CHECKPOINT=/absolute/path/to/trained_3_1_05.ckpt
bash scripts/evaluation/run_diffusiondrive_spr.sh
SPR_OUTPUT=selector bash scripts/evaluation/run_diffusiondrive_spr.sh
```

可将现有训练命令中的 cache_path、use_cache_without_dataset、devices 等参数追加到脚本后；
脚本没有硬编码个人服务器数据路径。INIT_CHECKPOINT 不设置时从头初始化。
新增头没有训练好的权重随代码发布。

评测时加载缺少 SPR 参数的旧 checkpoint 会报错；旧 checkpoint 仅允许用于训练热启动。
已含 SPR 的 checkpoint 在训练和评测时均要求 anchor 的数值与顺序一致。
该重构 MLP 对候选位置敏感，不能换一个相同 K 但重新排序的 bank。
若要换 bank，应从不含 SPR 的基线重新训练，或另行设计权重迁移实验。

## 如何判断是否改善 PDM

训练及验证会记录 `spr_ade`、`selector_ade` 和 `proposal_oracle_ade`，
单位为米，均为 detached 观测值，不参与反向传播。
oracle 是每个样本全部候选中 ADE 最小者，只用于诊断，推理不访问 GT。

应比较三组：原 3.1.02 权重 + 原选择器；3.1.05 权重 + SPR；
同一 3.1.05 权重 + selector。第三组可分析重构输出本身是否有用，
但不能替代未训练 SPR 的独立基线，因为共享编码器也发生了更新。

正式比较需使用同一评测 token 集、相同 anchor、相同训练预算，
同时报告 PDM 总分及碰撞、可行驶区域、TTC、舒适性和进度分项。
现有 PDM 脚本有随机扩散采样及无序 token 遍历，两次独立运行不保证
逐 token 使用相同噪声；本次未修改评测器，应重复评测并报告波动，
不能把小幅单次差异直接归因为 SPR。

ADE 下降不等于 PDM 上升。重构可能改善硬选择的误差，也可能生成越界或
不舒适的轨迹；是否提升需以上述 PDM 结果为准。当前无实验分数。

## 验证状态

已添加坐标往返/周期性、原始归一化、可变 K、梯度边界、损失权重、
训练/推理接线、checkpoint 完整性、bank 顺序和 CUDA AMP 检查。
集成单测沿用仓库的轻量 scheduler/decoder stubs，不能代替真实服务器 smoke。
按实验工作流，本地未执行 pytest、smoke、语法检查或完整 PDM；
服务器命令通过后再开始正式训练。
