# 3.1.05_6：独立训练的高代价排序修正

日期：2026-09-19。基于独立工作树；不覆盖以前的实验。用户明确要求模块单独训练，避免多种损失相互干扰。

## 证据与边界

- 同一 K67 候选集合，navtest：base 88.488936 → PCS 89.075571 → PCS+TRV 89.181861。
- navtrain val：PCS 87.410486，候选 Oracle 96.175107；这是候选上限，不是可兑现收益，更不能当作 navtest 上限。
- 3.1.05_5 联合训练最佳 epoch16：calibration +0.027234 PDM分；audit -0.015178分，55次修改仅5次有益、41次有害。没有运行该版 navtest。现有实现未证实泛化收益，不等于证明所有后处理路线不可能有效。
- 原 TRV 的 NC 相对风险样本 train 56/85109、val 126/18179；后处理 Gate 正样本率 train 约1.17%、val约8.39%。泛化错误分布不同是已观察事实，不能仅凭这些数字断定其唯一原因是过拟合或缺失未来信息。
- 3.1.05_6 尚无服务器实验结果，不预报提升分数。

## 架构：只有新模块接收梯度

冻结原 K67 和原 PCS，从 PCS decoder 提取每条候选的512维特征、5个分项预测、合成分数，以及缓存中的轨迹与原始 base logits。独立的小排序器使用候选 self-attention 输出分数残差：

`s_new[i] = s_PCS[i] + alpha * 0.2 * tanh(h[i])`

- 原 PCS 与生成器不在新模块的 optimizer 中，绝不回传梯度；不修改旧权重文件。
- 残差末层零初始化；初始输出分数/argmax 与同一次原 PCS 推理完全相同，包括并列值。
- 同一组67条轨迹不变；允许选第三条等候选，不限于 PCS/base 二选一。
- 新模块没有 gate、risk、PDM回归、BCE、蒸馏、时机或力度辅助损失。
- 旧 TRV 是独立对照，不接到新排名后面。它的训练分布与 PCS hash 都绑定原 PCS，不应强行复用作新策略否决器。
- feature 用FP16存储再还原FP32；原 PCS 分数及轨迹保留FP32；零初始化和实时评估遵循同样规则。批大小改变仍可能导致旧 PCS 极近分数的 argmax 浮点翻转，不能宣称跨不同批大小 bitwise 等价。

## 唯一训练目标

每个场景内，真实 PDM `q_i-q_j >= 0.02` 才构成有明确质量差异的有序候选对，近似平局忽略。

`L_pair = w_ij * softplus(-(s_new[i]-s_new[j])/0.05)`

`w_ij = min((q_i-q_j)/0.1, 5) * (1 + 3 * hard_ij)`

`hard_ij` 表示较差的j进入折外教师 top5，且教师给j的分数不低于更好的i。保留所有正常有序对的基础权重，绝不只训练稀有危险场景。每场景按有效对数量归一化，再对有有效对的场景求平均；标签全部相同的场景贡献可微零损失。

这是一项损失内部的样本加权，不是多loss求和。单损失同样可能过拟合，残差限制和验证回退不能保证收益。

## 折外挖掘：与最终模块分开

1. **只把 navtrain train 的完整 recording logs 分成3折**，同log不跨折；验证日志不参与。
2. 每折新建一个随机初始化 PCS 教师，只在另外两折训练20轮。该准备阶段沿用原 PCS BCE，但它属于独立临时模型，不和新排序模块联合训练，也不覆盖原 PCS。
3. 教师只给它未训练过的一折预测分数，合并为85109个train场景的OOF分数；逐token/索引/日志/来源hash检查，要求每场景恰好覆盖一次。
4. 新排序器读取固定OOF分数挖掘高代价错序，正式训练不需要运行这些教师。

**准确限定：OOF仅针对挖掘教师。K67和原PCS特征提供者已经在原train上训练过，本版并非整条网络严格交叉拟合。** 教师只见2/3数据，其错误也未必完全代表实际部署PCS的错误，因此保存OOF与原PCS训练PDM/归零统计供判断。不能把更多挖出的错误直接当作改进证据。

## 缓存与成本

- 复用 `pcs_candidates_k67_3_1_05`，包括真实 PDM 分项/合成分数/方向和场景context。所有原有provenance文件保持Git字节不变。
- 不需要原始 `training_cache`、navtrain metric cache、GTRS、PTR edit_cache、Gate cache，不删除任何旧文件。
- 新增 `cost_ranked_pcs_3_1_05_6/features/navtrain`：约8GB量级的紧凑memmap，主要是67×512 FP16特征。一次提取，训练时不反复运行原PCS。
- 三折teacher训练仍需读取原205GB候选缓存，每折20轮、每轮约2/3 train。三折累计约40个完整train轮的数据读取量，**比最终小模块训练更耗时**。
- OOF分数本体约22MiB，另含分折预测、teacher checkpoints、报告。训练、重新resume不需要重生成候选/PDM标签。
- feature cache写入数据flush后才标记done，失败可原目录续跑。单个prepare进程写同一feature root，不要对同一目录并发prepare。
- memmap按字段打开一次，WORKERS默认0，避免之前Too many open files和数据进程开销。

## 校准、验证与停止条件

- navtrain val按log确定性分为calibration/audit。alpha只搜索固定的`0,.25,.5,1`。
- calibration选择PDM最佳且NC/DAC/TTC/comfort/direction均值不下降的alpha；没有大于1e-6原始PDM的收益时取0。
- checkpoint只监控calibration PDM；audit不能影响alpha、epoch或训练。每轮记录两边的全alpha表，以便解释失败。
- 指标包括PDM、有益/有害切换、损失总额、严重损失(Δ≤-0.2)、新增归零、NC/DAC/TTC救回和新增失败、selection regret。
- 均值安全约束不保证每场景安全，新增失败必须一并审阅。
- 这些navtrain val日志在旧实验中已经反复查看；本版audit只表示本次未用于选模型的分区，不能声称是历史上从未查看的独立数据。反复根据audit改模型会进一步削弱独立性。
- 若calibration禁用、audit负收益或严重失误增加，保留负结果，不自动运行navtest。
- 通过后才进行固定checkpoint/固定alpha的navtest，输出原base/原PCS/原TRV/新ranker四个对照。navtest不重调alpha。
- 先证明优于原PCS，再检查是否超过89.181861的旧TRV对照；前者不能代替后者。

## 代码与验证记录

新增 `cost_rank/{model,data,metrics,pipeline,training}.py`、runner、launcher、installer和专门单元测试。没有修改旧PCS/生成器/评分/provenance文件。

测试覆盖：原PCS冻结、零初始化及并列值、标签不能进入推理、唯一loss梯度方向/代价权重/OOF权重、无有效候选对、log划分、校准回退/安全约束/audit不能选策略。单卡与四卡Smoke均为完整训练前的必经步骤。

**本地不运行测试或语法检查；等待服务器navhigh执行单元测试、单卡/四卡Smoke和正式验证。** 静态代码审阅不代表服务器已通过。

## 论文表述

可以描述为“在冻结生成器和基础评分器上，独立训练的代价敏感候选排序残差，以及按日志分组的折外难例挖掘”。排序损失本身不是新概念；本实验是否有贡献要由受控结果支撑。参考DiffusionDriveV2 Table8中辅助rank loss的消融，但不能直接搬用其收益。

https://arxiv.org/html/2512.07745v1#S8
