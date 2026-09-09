# 3.1.05_2：原 PCS 的三因子风险否决器（TRV）

日期：2026-09-09。分支：
`change/20260909-hard-negative-tririsk-veto-3.1.05_2`。

本分支以原 3.1.05 PCS commit `303e9f1` 为基底，保留原 PCS epoch5 作为冻结
proposer。新增 Triple-Risk Veto（TRV），只决定接受 PCS 候选还是回退原 selector
的 base。原 PCS、上一版 GTRS+CAGS 分支和所有验证结果均保留。

## 实验依据

原 PCS 在同次 navtest 候选上达到 0.8907557，相对 base 提升 0.0058663，
但产生 231 个新增零分。上一版 CAGS 将新增零分降至 67、换选胜率提高至 84.3%，
同时将 PDM 降至 0.8868674。其 113 个严重损失中有 54 个仍被判断为高胜率、
低风险，说明全局 GTRS 负例没有充分覆盖原 PCS 真正会选择的困难候选。

TRV 直接把原 PCS 的历史决策作为训练分布，避免重新学习候选排序，也不再引入
GTRS 轨迹。目标是保留原 PCS 的主体收益，只拦截相对 base 的安全退化。

## 数据与标签

复用已有 K67 candidate cache。`prepare-pairs` 用冻结 epoch5 PCS 遍历 navtrain 的
train/val 候选，为每个场景记录：PCS mode、base mode、两者 PDM 和五项子指标。
该缓存只有索引和标签，不包含 BEV，不运行生成器或 PDM 仿真，可断点续存。

三个风险标签只表示 candidate 相对 base 的退化：

```text
r_nc  = clamp(NC_base  - NC_candidate,  0, 1)
r_dac = clamp(DAC_base - DAC_candidate, 0, 1)
r_ttc = clamp(TTC_base - TTC_candidate, 0, 1)
```

base 与 candidate 同时失败时标签为 0，因为回退不能改善该指标；base 安全而
candidate 归零时标签为 1。NC 从 1 降至 0.5 时保留 0.5 软标签。

## 模型与训练

冻结原 PCS encoder、decoder 和五项 metric heads。风险模块读取 PCS candidate、
base 及两者残差的 512 维特征，同时读取两条轨迹的预测子指标、原 selector 概率
和 PCS 分数。NC、DAC、TTC 使用三个相互独立的 MLP head。

损失采用逐因子 focal BCE。相对退化、指标从正数降为零，以及 PCS 总分输给 base
分别增加权重，使低风险误报的代价高于不必要回退。optimizer 只包含三个风险头，
原 PCS 权重不更新。

每个 epoch 在独立 val cache 上搜索三个风险阈值，以最终“接受 PCS 或回退 base”
后的真实 val PDM 选择组合。搜索集合包含 `[1,1,1]` 的完全不否决配置，因此风险头
无效时，val 策略可以退回原 PCS。checkpoint 按否决后的 `val/pdm` 选择，并保存
当轮阈值。

## 推理与诊断

1. 原 PCS 从 K67 中选择 candidate。
2. 三个风险头预测 candidate 相对 base 的 NC、DAC、TTC 退化概率。
3. 任一风险超过其 val 阈值时回退 base，否则保留 PCS candidate。

三个风险使用 OR，而不取平均，因为任一乘法安全项归零都能使 PDM 归零。
完整验证同时评分最终轨迹、原 PCS candidate 和 base，输出三份标准 CSV、逐场景
风险与否决记录，以及相对 PCS/base 的净增益。

## 复用范围与止损

本分支不修改旧 candidate provenance 涉及的文件，直接复用：

- `/home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05`
- 原 navtest metric cache；
- 原 PCS epoch5 checkpoint；
- 原 K67 baseline、anchor 和生成器。

不需要 GTRS、training feature cache 或新的 metric cache。新增 pair cache 预计只有
数 MB；正式训练每卡 BS32、物理 0–3 卡、10 epochs。

本实验的主要有效标准是：同次 `pcs_proposer.csv` 必须复现约 89.0756，且 TRV 的
最终 PDM 高于 PCS，同时减少新增零分。若完整 navtest 没有正增益，则按用户要求
作废该实验，不继续在 navtest 上反复调阈值。

服务器步骤见 [运行指令](server_3_1_05_2_triple_risk_veto.md)。
