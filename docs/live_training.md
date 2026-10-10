# 桌面观看真实 DrQ-v2 训练

对应论文：**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**，Pawel Ladosz 等，IEEE RA-L，2024，DOI：10.1109/LRA.2024.3379837。

这是本项目自行实现的 CPU 复现训练入口。任务与论文的差异、奖励公式的解释和已有短训练结果，见本目录其他文档；此入口不代表已复现论文成功率。

## 启动

已经配置好的这台电脑，双击桌面 **打开真实DrQ训练.cmd**。或者在 Ubuntu-20.04 的终端运行：

```bash
cd /mnt/d/paper_reproductions/posture_control
bash start_live_training.sh
```

脚本会创建独立的 `runs/live_training_日期_时间/` 目录，执行 12000 步新训练；已有模型不会被覆盖。有同一入口启动的训练仍在运行时，不再启动第二份。

手动选择步数和输出目录：

```bash
bash run_train_cpu.sh --gui --realtime --steps 12000 --output runs/my_live_training
```

手动指定输出目录时请使用新目录，避免覆盖该目录原有日志和模型。

## 桌面上看什么

1. **Bullet Physics**：当前训练环境中的无人机、移动平台与真实物理运动。
2. **LIVE DrQ-v2 TRAINING**：同一训练过程的下视图像、当前动作、训练步数、参数更新次数、损失、每轮结果。
3. 源码 `landing_repro/train.py`：训练循环；`agent.py`：DrQ-v2；`environment.py`：平台、无人机、奖励与终止条件。

前 256 步标注 `RANDOM WARM-UP`，动作来自随机采样，模型尚未控制无人机。之后标注 `MODEL + EXPLORATION NOISE`，动作来自模型并加入训练探索噪声。达到条件时每 4 步执行一次参数更新，`Updates` 会增加。

这个入口没有调用使用平台真值的 `oracle_action`。三维显示与训练共享同一个环境对象，窗口不是预录视频，也不是另开一个 PID 演示。网络输出速度指令后仍由底层 PID 转成电机转速。

每轮结束显示 `SUCCESS`（成功接触）、`GROUND`（落地失败）、`PLATFORM_EDGE`（平台边缘碰撞）或 `TIMEOUT`（超时），短暂停留后自动开始下一轮。上方累计的是训练中的成功次数；训练带探索噪声，正式成功率仍需要独立评估。

## 暂停、结束与保存

- **Pause / Continue**：暂停或继续训练。
- **Stop and save model**：正常结束，保存当前模型和汇总。关闭监控窗口也执行该操作。
- 每 1000 步另保存一次 `latest.pt`。强制关闭物理窗口或结束进程可能只保留之前的检查点，优先使用监控窗口的停止按钮。
- `console.log`：过程日志；`episodes.csv`：每轮结果；`metrics.jsonl`：参数更新指标；`live_status.json`：最新监控状态；`summary.json`：正常结束时汇总；`latest.pt`：模型。

软件渲染和 CPU 更新使实际运行可能慢于仿真时间；12000 步不是训练收敛的保证。当前检查点不包含经验池及完整随机数状态，不能视为可精确续训的快照。
