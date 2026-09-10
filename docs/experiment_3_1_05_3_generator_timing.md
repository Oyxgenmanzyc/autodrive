# 3.1.05_3：生成器时机监督，固定 PCS＋TRV

目的：在已有效的筛选器下验证用户原 3.22 的**轨迹生成时机监督**，不是继续修改筛选器。
上一个 timing-weighted-gain-selector 分支保留作历史；本分支从原 TRV `611e87b` 建立。

## 来源与取舍

原始来源是用户仓库的
[`change/20260713-brake-timing-supervision-3.22`](https://github.com/Oyxgenmanzyc/autodrive/tree/change/20260713-brake-timing-supervision-3.22)，
提交 `6f4168c20a106c7d2b7af36a0d04cea2308503ab`。

- `generator_timing/risk_brake_timing.py` 复用该提交同名文件的实现。
- `generator_timing/risk_utils.py` 提取该提交的 GT context 构建及其所需函数。
- 保留仓库 MIT LICENSE；迁移保留来源，不将已有方法宣称为新发明。
- 已核对 3.22.2、endpoint-transport、3.22.8 的设计。3.22.8 改成 all-mode risk-corridor；
  它增加安全距离、所需减速度和 jerk 等损失，属于另一个监督方案，未混入本次原 3.22 验证。
- 不迁移历史风险 attention、推理安全盾或新的筛选逻辑，以免改变本次问题。

## 训练到底改变什么

输入还是原 camera、LiDAR、车辆状态；现有 K67 anchor 和两步 DDIM 推理保留。
冻结感知编码器、场景上下文、原分类头参数，只微调轨迹头中的其余生成参数。
PCS epoch05 与 TRV epoch02 的参数、阈值完全固定，不进入训练模型和优化器。
冻结感知模块同时保持 eval 模式，避免 BatchNorm 缓冲区和 dropout 导致上下文漂移。

训练损失为原轨迹回归/分类损失加原 3.22 时机项。原分类头参数虽然冻结，
共享生成特征改变后，分类 logits 仍可能改变；“固定筛选器”指 PCS/TRV 权重与规则不变，
不意味着对变化后的候选强行保持相同编号、分数或回退轨迹。

原 3.22 时机项：

1. 用未来 GT 标注跟踪当前前车，要求连续至少两帧有效。
2. 当前或延迟 TTC 进入随速度变化的 T1 加 1 秒准备带，且 GT 持续刹车至少两步，才激活。
3. 用当前位置原点和当前车速计算候选速度、加速度，dt=0.5 秒。
4. `sigmoid((-0.5-acceleration)/0.35)` 给出软刹车轮廓。
5. GT anchor 匹配模式与 GT 的瞬时轮廓、累计最大轮廓分别做 SmoothL1，权重 0.25/0.75，
   总辅助权重 0.1；只作用于最后一个解码层，与原实现一致。

未来标注只进入 targets 和 loss，不进入生成器/筛选器的推理特征。
**原方法直接监督一个匹配模式，通过共享生成网络间接影响其他模式；不能预先声称所有轨迹都会更好。**

## 对照设计

| 组别 | 初始生成器 | 额外训练 | 时机损失 | 最终筛选器 |
|---|---|---|---|---|
| 原始参考 | K67 epoch99 | 无 | 无 | 原 PCS＋TRV |
| control | 同一 K67 epoch99 | 10 epochs | 0 | 同一 PCS＋TRV |
| timing | 同一 K67 epoch99 | 10 epochs | 0.1 | 同一 PCS＋TRV |

默认每卡 BS32、4 卡、AdamW LR=2e-5、10 epochs，均为本次微调设置，不冒称原 3.22 的完全复现。
两组必须使用相同数据、超参和种子。首轮验证可用单种子筛查，稳定结论需后续多种子复核。
主比较使用两组相同步数的 `last.ckpt`；自动保存的 `epoch=xx.ckpt` 按候选 oracle ADE 选择，
不是按 PDM 选择，不能当作最佳 PDM。不要用 navtest 调超参或选 epoch。

训练 val 日志中的 `brake_timing_selected` 是原分类头选中的轨迹诊断，
不是 PCS/TRV 选中轨迹的最终官方 TTC。DDP val sampler 可能补齐少量样本；
完整最终评测采用不重复的同一 token 集，输出数必须匹配。

## 缓存边界

| 数据 | 本版用途 |
|---|---|
| `training_cache_3_1_02_k67/*/*/transfuser_feature.gz` | 复用原始 image/LiDAR/status 特征，生成器训练必需 |
| 原 `transfuser_target.gz` | 不必复制；本版 sidecar 同时保存 GT trajectory 和 timing context |
| 新 `generator_brake_timing_3_1_05_3/targets/navtrain.pt` | 新建小型 GT 时机监督，约数十 MB，以实测为准 |
| 旧 PCS cache | 只读取 `records.json` 固定场景划分，不读其候选/上下文张量训练新生成器 |
| 旧 decision-pair/GTRS cache | 本版不用 |
| navtrain metric cache | 监督训练不需要 |
| navtest `metric_cache` | 复用；评测时实时生成和打分新候选，不用重做场景 metric cache |

如果原始特征已删除，`prepare-features` 可以只补缺失特征，读取原始 trainval sensor_blobs；
这可能重新占用约原 94 GB 空间。不会删除或覆盖已有缓存，不创建一份重复的 205 GB PCS cache。
准备 sidecar 后，其内部已保存 records；保留小型 `records.json` 方便重建。
已有原始特征假定来自未修改的 K67 FeatureBuilder；本分支不修改它。

## 评测如何定位瓶颈

同场景、同 token seed 分别用原生成器与新生成器生成全部 K67 候选，用同一冻结筛选器选轨迹。
每条候选采用已有官方等价、逐候选归一化的 PDM 打分，首场景做等价检查。
评测比只评分最终轨迹更慢，但只写紧凑诊断，不重建完整训练候选缓存。

输出 `paired_results.csv`：

- 原始/新生成器分别经 base、PCS、TRV 后的 PDM 及所有子指标。
- 全部候选平均 PDM、TTC 合格比例、NC/DAC/TTC 同时合格比例。
- 每场景 oracle PDM、实际选择距 oracle 的差距、是否存在安全候选。
- 新旧最终 PDM 的逐场景差值、否决情况。

输出 `candidate_diagnostics.npz`：`tokens [N]`、`proposals [N,2,K,8,3]`、
`labels [N,2,K,5]`、`scores [N,2,K]`。第二维顺序为原始/新生成器；
label 顺序为 NC、DAC、progress、TTC、comfort。

判断路径：

- timing 相对 control，候选 TTC、PDM 均值/oracle 与最终分一起改善：支持生成时机监督有收益。
- 候选改善但最终不改善：继续检查固定筛选器在新候选分布上的误选；这次不重训筛选器掩盖它。
- 候选本身没改善：检查训练激活覆盖、时机误差/晚刹率、过早刹车及 progress 损失。
- 只有 timing/control 相对原始都改善：可能主要是额外训练收益，不能直接归功于时机项。

## 验证状态

未在本地运行测试、语法检查或 smoke，待服务器 navhigh 验证。
旧 TRV 89.181861 是参考结果，不能当作本版成绩。
