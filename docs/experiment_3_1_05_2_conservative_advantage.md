# 3.1.05_2：保守优势门控选择器（CAGS）

日期：2026-09-08。分支：
`change/20260908-gtrs-conservative-advantage-selector-3.1.05_2`。

本实验保留已验证的 3.1.05 PCS 代码、checkpoint 与 205GB K67 候选缓存，
在独立分支中增加 **Conservative Advantage-Gated Selector（CAGS）**。
它针对 3.1.05 的主要失分机制：模型几乎总在换选，但实际负收益场景中
93.7% 的选择被 base 在所有 PDM 子项上支配，且新增零分主要来自 DAC/NC 失败。

## 核心机制

CAGS 把原 selector 的候选作为显式 base action，并为每条候选联合预测：

- 五项 PDM 子指标，用于继承原 PCS 的场景和轨迹表征；
- `predicted_delta`：`PDM(candidate) - PDM(base)`；
- `predicted_win_probability`：候选优于 base 的概率；
- `predicted_catastrophic_risk`：NC、DAC、TTC 为零或相对 base 严重退化的概率。

选择效用为 `predicted_delta - risk_penalty * predicted_catastrophic_risk`。
只有候选同时通过最小优势、最小胜率和最大灾难风险三个条件才替换 base，
否则直接保留 base。base 在决策空间中固定为零优势、零风险动作。

训练包含五项子指标 BCE、相对优势回归、胜负 BCE、灾难风险 BCE，
以及基于真实相对优势和灾难惩罚构造目标分布的 KL 损失。
最后一项属于保守的离线优势加权策略改进，不是在线强化学习：训练过程中
没有环境交互，也不使用 navtest 奖励。这样可直接利用完整候选的 PDM 标签，
同时避免策略梯度在稀疏乘法 PDM 奖励上的高方差。

## GTRS 的使用边界

数据来自 DiffusionDriveV2 文档要求单独下载的 `navtrain_16384.pkl`，
轨迹词表来自 `gtrs_traj/16384.npy`。参考其训练代码，将 40 帧词表按
`[:, ::5]` 降采样为与当前模型一致的 8×3 轨迹。GTRS 仅用于训练增强，
不会在推理时扩大 67 条候选，也不会改变最终轨迹生成器。

原始 pkl 约 29GB。预处理对每个 navtrain 场景按 token 读取 16384 条标签，
确定性选取默认 32 条：一半优先覆盖绝对安全失败或严重负优势，四分之一覆盖
与 base 接近的难例，八分之一覆盖明显正优势，其余随机补齐且不重复。
预处理结果只保存 `uint16` 词表索引和 `float16` 标签/分数，训练每轮不再读取
29GB pkl，也不重跑 PDM 仿真。预处理逐场景记录完成状态，同一配置中断后可续存。

该增强的主要价值是补足危险负例和 base 决策边界，而不是直接提高生成轨迹质量。
GTRS 轨迹不会进入 navtest 推理候选，因此它能否提高最终 PDM，取决于学到的风险
边界能否迁移到原 K67 候选。该结果必须通过同次 base 配对验证判断。

来源：

- [DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2)
- [官方训练与验证说明](https://github.com/hustvl/DiffusionDriveV2/blob/master/docs/train_eval.md)
- [原 3.1.05 PCS 设计](experiment_3_1_05_pcs.md)

仓库不包含 29GB pkl 或 7.6MB 词表；服务器脚本只引用用户已有文件。
V2 评分结构的 MIT 许可仍保存在 `pcs/LICENSE_DiffusionDriveV2.txt`。

## 缓存和权重兼容

本分支没有修改旧候选缓存 provenance 中参与哈希的文件，因此直接复用：

`/home/hndx/navsim_workspace/exp/pcs_candidates_k67_3_1_05`

无需重新生成 training feature cache、navtrain metric cache 或 K67 PCS cache。
新增的 compact GTRS cache 是训练索引，不是 metric cache，预期仅几十 MB。

训练默认从原 3.1.05 最佳 `epoch=05.ckpt` 继承 encoder、decoder 和五个 metric
heads；新增决策 head 单独初始化。初始化会严格检查旧 checkpoint 与 K67 cache 的
baseline、anchor、seed 和实现 provenance，不允许 `strict=False` 静默跳过共享权重。
原 checkpoint 文件不会被改写。

## 验证与止损

默认门控为：最小预测优势 0.01、最小胜率 0.55、最大灾难风险 0.10、
风险惩罚 0.50。验证时可以覆盖这些阈值而无需重训，便于在独立数据上检查
收益与失误的关系；阈值选择不能读取 navtest 总分后反复调优。

重点检查：

- `delta` 是否稳定高于原 PCS 的 +0.00587；
- `new_zero` 和 dominated selection 是否显著下降；
- change rate 是否从原 PCS 的 97.94% 降到有意义的范围；
- NC、DAC、TTC 的减少是否小于 EP 的收益；
- `predicted_delta`、胜率和风险是否有区分度，而非再次饱和。

GTRS 只改善选择器的风险识别，不提高 K67 oracle 上限。若独立验证仍无法识别
高代价误选，应停止继续调阈值，转向提高候选生成质量或加入更直接的场景风险特征。

执行步骤见 [服务器指令](server_3_1_05_2.md)。
