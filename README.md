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

- [运行方法、桌面入口与实现差异](docs/running_cpu_landing.md)
- [实际实验记录与结果](docs/cpu_experiment_report.md)
- [任务与算法源码](landing_repro/)

桌面的 PID 调试演示使用平台真值，不能作为 DrQ-v2 学会降落的证据。
训练和评估中的策略只读取图像和无人机自身状态；模型结果以独立评估记录为准。

更新日期：2026-10-10。
