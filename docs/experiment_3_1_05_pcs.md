# 新 3.1.05：PDM Candidate Scoring（PCS）

日期：2026-09-07。分支：`change/20260907-pdm-candidate-scoring-3.1.05`。

本分支替代 SPR 作为新的 3.1.05 实验方案。原 SPR 分支保留，结果不作覆盖。
基底为 `ddaafbc`（3.1.02 原算法及只读 anchor 诊断），基础模型使用原 3.1.02
epoch99、step266000 的 K67 checkpoint；不能使用 SPR 或 matched control checkpoint 替代。

## 已有证据和研究边界

历史 navtest 总分：3.1.02 为 88.4440；SPR 为 88.4172；
关闭 SPR 的匹配对照为 88.2635；SPR 训练模型的原 selector 为 88.2561。
这些单次结果不足以证明稳定因果收益。本分支尚无训练或 PDM 实测结果，
不承诺达到 88.6，也不把 V2 的整套模型成绩当作 selector 的单独收益。

PCS 目标是在冻结候选生成能力的条件下，提高实际所选轨迹的 PDM。
第一版仅检验子指标监督的单阶段评分器，不同时加入重构、RL、候选扩增、
刹车修正、精筛、多重风险门控或额外 progress 排序损失。

## 来源和取舍

主要来源：
- [DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2/tree/1cd12a1e155c34dcc471261835444c8d5587580b)
- [selector 源码](https://github.com/hustvl/DiffusionDriveV2/blob/1cd12a1e155c34dcc471261835444c8d5587580b/navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py)
- [BEV 采样](https://github.com/hustvl/DiffusionDriveV2/blob/1cd12a1e155c34dcc471261835444c8d5587580b/navsim/agents/diffusiondrivev2/modules/blocks.py)
- [3.20.10](https://github.com/Oyxgenmanzyc/autodrive/tree/233457e00121e35fcf65d0dd63ded4eb3de740b9)

`pcs/v2_blocks.py` 保留 V2 的航向编码、轨迹点 BEV 采样和单层 ScorerTransformerDecoderLayer。
保留原 MIT 许可于同目录的 `LICENSE_DiffusionDriveV2.txt`。PCS 是本项目实验名称，
不表示原始评分结构由本项目独立提出。论文应引用 DiffusionDriveV2，
不能将改名、K67 适配或常规工程封装当作原创结构贡献。

3.20.10 借鉴的是 detach 的隔离原则和同次推理配对诊断设计；
没有搬入它的历史风险特征、几何代理标签、timing head、path guard 或 shield。

## 数据流

1. 冻结生成器：原 backbone、transformer、原分类头和轨迹头全部冻结并处于 eval 模式。
2. 保留原两步去噪，生成 67 条最终 8×3 轨迹及原分类 logits；不改变候选坐标或航向。
3. 缓存生成器提供的 BEV、自车/周围车辆特征。场景特征存 FP16，
   候选几何与 logits 存 FP32；在线推理采用相同特征舍入方式。
4. 评分器编码轨迹坐标与航向，沿轨迹采样 BEV，与车辆、自车和其他候选交互。
5. 五个 head 预测 NC、DAC、EP、TTC、C；采用真实 PDM 子指标的 BCE 监督（允许软标签）。
6. 按 `NC * DAC * (5*EP + 5*TTC + 2*C) / 12` 选择最大预测分数候选。

使用 512 维、16 头的原 V2 scorer layer；候选数从数据维度读取，不写死 20。
训练仅包含评分器，无生成器 optimizer；子指标头不直接接收未来数据。

与上游的有意差异：
- 单阶段，没有 coarse-to-fine 筛选。
- 保留 NC=0.5 标签；不把它原地改为 0。
- 第一版没有额外 EP 排序损失，以免把 progress 优化误称为总 PDM 排序。
- 不裁剪或重算最终候选几何，确保评分标签与返回轨迹是同一条。
- 离线缓存真实推理候选，避免每个 epoch 重跑冻结生成器与 PDM 仿真。
- 初始化只接收严格匹配的基础模型权重，anchor 内容不一致直接报错。

## PDM 标签一致性

参考 V2 的 pairwise normalization：
一次批量模拟参考轨迹与所有候选，再把每条候选的 progress 单独与参考轨迹归一化。
禁止使用整个候选集合的最大 progress 直接生成训练标签。
每个缓存 shard 的首个新场景，对三条候选逐一调用原 `pdm_score()`，
要求五项子指标、总分及 direction 的误差不超过 1e-6，通过后才继续。
处理静止分母时不加扰动真实分数的 epsilon。

训练 cache 来自 navtrain；沿用原 `train_logs/val_logs`，
两者日志集合必须不相交。navtest cache 被 CandidateDataset 明确禁止用于训练。
缓存覆盖范围是原特征 cache 中指定日志的可用场景，程序打印数量；
已有特征 cache 本身是否完整应结合原训练记录核对。
任何请求场景缺少 metric cache，程序报错，不默默跳过。
训练前要求 manifest 中所有指定场景都已完成缓存。

## 可复现性和对照

- 每条 token 的扩散随机种子由 token 和 seed 派生，与进程顺序、shard 和断点续存无关。
- 缓存记录 baseline、anchor、关键实现和评分配置哈希，混用时拒绝训练/验证。
- 缓存 manifest 记录完整请求列表与数据集来源；pilot 和正式 cache 采用独立目录。
- 每次在线验证都在同一批候选中输出 PCS 和原 selector。
- 新采样顺序与历史评估不同，因此本次原 selector 不保证恰好重现历史 88.4440；
  必须优先使用同次 `base_selector.csv` 判定净增益。
- 训练按独立 val 日志的真实已缓存 PDM 选 best checkpoint，不按 BCE 或 navtest 选。
- DDP 验证按场景索引去除 sampler 补齐的重复样本后聚合。
- 训练中 val PDM 衡量的是固定缓存候选上的选择质量，不等同于最终 navtest 实测。
- 完整 eval 会输出 pcs.csv、base_selector.csv、paired_results.csv、summary.json 和 run.json。
  中途失败保留已完成的 paired_results.csv，但不生成 completed=true 的汇总。

## 成本和止损

复用原 K67 训练特征 cache；原 navtest metric cache 继续用于最终验证。
若训练集 metric cache 尚未生成，则须新增 navtrain metric cache。
另新增候选及场景特征缓存，不能直接用原目标 cache 代替。

先跑 pilot，测单场景缓存体积与评分耗时。缓存包含完整 BEV，
正式缓存可能需要较多磁盘空间，不能按几 KB 的轨迹文件估算。
四卡 cache 是四个独立 shard，CPU 评分默认每卡四进程。
训练每卡 BS32、四卡全局 BS128、20 epochs、3e-4、16-mixed。
这些是起始设置，未声称已测出最佳超参数或运行时间。

先看独立 val 的候选 oracle 和前 5/10 候选上限。
如果改选空间不足，就停止；上限高也不意味着学习器一定可达。
正式评估除总分外，检查救回零分、新增零分、DAC/NC/TTC 变化。
只有独立验证显示净收益，再进入完整 navtest；稳定收益需多种子复验。

## 验证记录

新增测试覆盖 K67、候选重排等变性、梯度隔离、checkpoint 往返、
逐参考归一化、静止/零分边界、软标签不变、禁止 navtest 训练和 manifest 冲突。
按照项目工作流，本地未运行单元测试、语法检查或 smoke；
需在 navhigh 运行 test、pilot、smoke 和 eval-smoke。
本分支当前是待服务器验证的实现，尚无新 PDM 结果。

执行步骤见 [服务器指令](server_3_1_05_pcs.md)。

