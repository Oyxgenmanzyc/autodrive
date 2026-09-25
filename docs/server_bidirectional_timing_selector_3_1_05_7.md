# 3.1.05_7 单卡双向 Timing Selector

该模块不会修改或训练生成器、PCS Scorer、Cost Ranker。完整流程先为 train split 生成 31 动作教师标签，再单独训练 Timing Selector。

默认训练参数与此前实验一致：20 epochs、batch size 32、workers 0、learning rate 1e-4、FP32。设备固定为用户提供的物理 GPU UUID。

服务器顺序：

1. `check`
2. `test`
3. `prepare-train`
4. `train`

`prepare-train` 可重复执行，已经写完并校验过的 block 会跳过。训练结果写入 `bidirectional_timing_selector_3_1_05_7/train/<timestamp>`。每轮生成 calibration/audit 报告；checkpoint 只按 calibration PDM 选择。
