# 3.1.05_5：独立修改 Gate 与配对专精训练

修改日期：2026-09-18。涉及文件：`gated_refinement/{labels,data,model,training}.py`、`run_gated_refinement.py`、`run_3_1_05_5.sh`、`install_3_1_05_5.sh`、`test_gated_refinement.py`、服务器文档和 CHANGELOG。预期影响是检验可修正场景的可辨识性并减少identity主导的决策失效；论文中应作为独立Gate/动作头与配对训练的新增实验，不能把Oracle上界写成学习模型提升。

## 实验动机与证据

3.1.05_4 离线受约束 Oracle：训练 85109 场景中 1378 个可改善（1.62%），验证 18179 场景中 1557 个可改善（8.57%）。验证 Oracle 为 93.1259 PDM，原输出 87.5316；这是利用真实评分选择动作的上界，不是模型效果。epoch00 扩展阈值诊断在 risk_limit=1、margin=0 时 edits=0，说明仅放宽风险阈值无法解除未触发修改的问题。gain Pearson 0.241、risk Pearson 0.288 不能证明可部署的尾部识别能力。

## 架构和标签

- 冻结原 K67、PCS、TRV。新增模块接在 TRV 最终轨迹之后。
- Gate 读取原轨迹的几何特征、采样 BEV、agent/ego context。收益与风险标签不进入网络输入。
- 复用 Cross-Attention，为修改动作预测 gain/risk；Gate 单独输出是否值得修改，另设分项改善原因与 regret 辅助头。
- 不再让所有修改动作与固定为零的 identity gain 竞争。Gate 开启后在通过风险过滤的非 identity 动作中按预测收益排序；所有动作被过滤时输出 identity。
- 现有动作空间只有额外制动：4 个 onset × 3 个 additional deceleration × 2 个 ramp = 24 动作，另加 identity。无法加速、改变横向路径、直接恢复过度制动。此版不扩充动作空间，从而能复用原评分。
- 所有正例动作必须 NC/DAC/TTC/comfort/direction 不退化、PDM 增益 > 1e-6；同时满足增益 >= 0.02，或 NC/DAC/TTC 任一改善。0.02 是预先设定的实验超参数，不是已验证的最优值。
- 监督 need_modify、best_action、onset/deceleration/ramp、oracle_gain，以及 NC/DAC/TTC/comfort/progress 改善标签。它们说明“哪些指标改善”，不能证明事故根因是刹车过晚。
- 原轨迹存在安全失败但动作库无有效解时标记 unresolved_by_bank：对于本编辑器的 Gate 是负例，不代表真实场景安全。
- 正例动作采用有效动作上的 soft target。只有正例参与动作排序/收益回归；Gate/原因/regret 在配对全体样本上训练，联合阶段风险头对所有样本训练。

## 相似负例与训练步骤

1. 只用训练集 selected trajectory 的位置、速度、加速度、航向正余弦做标准化描述子；每个正例找 4 个不同 recording log 的最近负例。标准化统计只从训练集拟合。该距离是轨迹运动学相似代理，尚未覆盖完整的交互语义。
2. 每组 1 个正例、2 个配对负例、1 个随机负例。Gate 使用 BCE 加配对排序损失，不叠加巨大 pos_weight。随机负例和配对负例每轮重新抽选。
3. 一轮覆盖所有正例一次（仅 DDP 尾部补齐）；不按 85109 定义训练轮。此版 20 轮比旧版全数据 20 轮更新次数少，日志会保存正例数量。验证仍覆盖完整 18179 场景。
4. probe 阶段只训练小模块的 Gate、原因、regret；动作头冻结。主感知/生成器均使用原冻结缓存。它是小型上下文探针，不等同于仅训练线性分类器。
5. joint 阶段从最佳 probe 初始化，训练 Gate 和动作/风险分支。新增 branch 不直接加载旧失败 refiner 的收益头。
6. 每5轮在已访问的训练负例中收集Gate最高误报分数，更新最多1024个难负例池。后续第4个负例槽以50%概率从该池抽取，另外50%仍随机。配对正例/两个相似负例不变；不会使用val误报挖掘训练样本，也不会额外扫描205GB缓存。所有正例仍每轮至少覆盖一次。

## 校准与验证

- 官方 navtrain val 按完整 log 确定性分成 calibration/audit 两半；log 不跨分区。训练样本不从两部分检索或采样。
- probe 按 calibration AUPRC 选 checkpoint；joint 按 calibration PDM 选 checkpoint，阈值也仅用 calibration。
- audit 原始分布结果仅用于诊断；若据此继续修改模型，它就不再是完全未使用的最终测试集。navtest 不用于训练/选阈值。
- 输出 AUPRC、正例比例、Recall/Precision at 5%/10% budget（阈值并列时包括所有并列样本并报告实际触发率）。过采样 Gate sigmoid 不是已校准真实概率；阈值在自然分布上选择。
- 每轮保存原始验证预测、所有阈值组合、calibration/audit 报告。风险约束检查分项均值，属于经验约束，不构成单场景安全保证。
- 最终必须看 audit/navtest PDM 净增益、有益/有害修改、TTC 救回/新增失败，不能只看 Gate 召回率。

## 缓存兼容

复用 PCS 205GB、完整 edit cache 98MB；prepare 只生成 gate labels/pairs/CSV，不读取原图、LiDAR、GTRS 或重新算 PDM。正式训练需要 PCS+edit+gate cache。在线 navtest 评估仍需原 checkpoint、backbone、anchor、PCS/TRV checkpoint、navtest 数据和 metric cache。

评分缓存与新模型代码分开版本校验：严格检查动作 geometry hash、缓存 schema、冻结 generator provenance、candidate manifest、records、逐 block 内容 hash；加载时逐场景验证 selected 与原 proposals[mode] 精确相等。不会因为模型训练代码变化而要求重新评分。旧 sources 元数据完整保留；在线验证继续检查原 selector checkpoint SHA。新的 gate cache 与新增源文件 SHA 绑定。

## 验证状态

遵循本项目 skill：not run locally; pending server smoke test。新增服务器单元测试覆盖标签保护、噪声增益、软目标、匹配防泄漏、DDP 配对采样、Gate/identity 解耦、并列 AUPRC、校准和梯度；训练效果待服务器验证。未声称已获得 PDM 提升。

## 2026-09-18 DDP 运行修复

首轮四卡正式 probe 在 epoch 4 结束后的第一次难负例更新处停滞，约 26 分钟后由 NCCL watchdog 中止（退出码 134）。这与客户端 SSH 断网无关；训练由服务器 `nohup` 运行。根因是训练钩子中新增的 Python 字典 `all_gather_object` collective。修复后每个 rank 只从自身已经访问的训练负例维护最多 1024 个难例，不再为该启发式步骤创建额外 NCCL collective。验证集全量汇总逻辑未变，因为它在 epoch 0--4 已连续正常完成且用于去除 DDP padding。

该修复只改变训练期难负例池的跨卡通信，不改变标签、Gate/动作模型、损失、采样比例或部署决策。为允许从故障前 checkpoint 续训，cache 兼容性仍严格验证标签定义和模型张量语义，同时继续验证 records、全部 edit block 内容 hash、generator/candidate provenance；训练调度、runner 和 loader 的运行修复不再伪装成标签失配。修复后的服务器单元测试及四卡续训仍待执行。
