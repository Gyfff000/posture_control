# posture_control

无人机控制与论文仿真复现笔记。

## 文献复现指南

[**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning** 的环境与实验步骤](docs/reproduction_ladosz_2024_vision_drqv2.md)

- 作者：Pawel Ladosz、Meraj Mammadov、Heejung Shin、Woojae Shin、Hyondong Oh。
- 来源：IEEE Robotics and Automation Letters，2024，9(5)：4575-4582。
- DOI：[10.1109/LRA.2024.3379837](https://doi.org/10.1109/LRA.2024.3379837)。
- 技术路线：PyBullet / Crazyflie、下视图像、DrQ-v2、三维速度控制和域随机化。
- 文档包括：CPU 可做的实验、固定版本环境安装、三维飞行示例、任务实现步骤、奖励公式疑点、训练参数、消融与评估流程。
- 当前状态：已完成 CPU 环境、移动平台降落任务、实时机载相机、DrQ-v2 训练与独立评估入口；已执行短训练。尚未完成论文规模的训练、域随机化和多种子正式对比。

## 可运行的 CPU 代码

- [变速平台近距离中心丢失修复：局部光流跟踪、末段衔接与 80000 步冻结对照](docs/near_tracking_v2.md)

- [运行方法、桌面入口与实现差异](docs/running_cpu_landing.md)
- [实际实验记录与结果](docs/cpu_experiment_report.md)
- [桌面观看真实模型训练：窗口、阶段和停止保存](docs/live_training.md)
- [最新 12000 步桌面训练结果与保存模型的独立测试](docs/live_training_12000_result.md)
- [新增辅助训练：已有移动平台成功降落、固定种子对比与两轮长训练](docs/guided_training.md)
- [任务与算法源码](landing_repro/)
- [平滑轨迹与软着陆：改动、初步对比、当前训练与桌面入口](docs/smooth_landing.md)
- [平台中心：部分可见与漏检预测、时间置信度、接近奖励](docs/platform_center_marker.md)
- [参考 PerchRL 的随机 B 样条、平滑变速和后续课程训练](docs/random_platform_motion.md)
- [暂停后的模型位置、桌面入口与恢复训练步骤](docs/resume_training.md)

桌面的 PID 调试演示使用平台真值，不能作为 DrQ-v2 学会降落的证据。
训练和评估中的策略只读取图像和无人机自身状态；模型结果以独立评估记录为准。

更新日期：2026-10-11。

最新状态：新增[近距离中心保持、动作约束与奖励](docs/near_landing_assist.md)后，针对第 66 回合中心丢失悬停问题，进一步增加[有时限的末段完成与下降提速](docs/terminal_landing_finish.md)。累计 **17013** 步的冻结模型在 6 个相同开发场景中，旧配置成功 5/6、新配置成功 6/6；平均回合用时降低，但 jerk 略增。该结果属于网络与辅助控制器的混合改进，不能替代独立评估或视作纯 DrQ-v2 成绩。桌面脚本从最近中断检查点继续训练。

后续状态：用户明确要求立即加大难度，已在累计 **42349** 步保存匀速阶段模型，直接切换到[阶段 1 随机变速直线课程](docs/random_platform_motion.md)，速度范围 0.08–0.35 m/s。后续按验证门槛自动升级到随机 B 样条转弯；变速阶段的性能需重新验证。

## 新增的辅助训练实验

示范预训练与在线纠错后，近距离移动平台固定种子测试已有 7/10 成功；更高难度仍不稳定。
这与论文纯 DrQ-v2 方法不同，详见 [训练改动、完整对照和局限](docs/guided_training.md)。
旧第一轮已完成 60000 步；旧第二轮在保存 6000 步检查点后，按用户要求切换到平滑轨迹改进训练。

## 平滑轨迹改进

新增自身指令记忆、动作变化损失、指令加速度/jerk 限制及软着陆奖励。初步测试真实飞行 jerk 下降，但降落成功率也下降，目前继续进行固定低速移动平台训练恢复跟踪能力。尚未证明稳定平滑降落，详情见 [完整实验说明](docs/smooth_landing.md)。
