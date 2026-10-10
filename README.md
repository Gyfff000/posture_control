# posture_control

无人机控制与论文仿真复现笔记。

## 文献复现指南

[**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning** 的环境与实验步骤](docs/reproduction_ladosz_2024_vision_drqv2.md)

- 作者：Pawel Ladosz、Meraj Mammadov、Heejung Shin、Woojae Shin、Hyondong Oh。
- 来源：IEEE Robotics and Automation Letters，2024，9(5)：4575-4582。
- DOI：[10.1109/LRA.2024.3379837](https://doi.org/10.1109/LRA.2024.3379837)。
- 技术路线：PyBullet / Crazyflie、下视图像、DrQ-v2、三维速度控制和域随机化。
- 文档包括：CPU 可做的实验、固定版本环境安装、三维飞行示例、任务实现步骤、奖励公式疑点、训练参数、消融与评估流程。
- 当前状态：已完成 WSL2 Ubuntu 20.04 / Python 3.10 / CPU PyTorch / PyBullet 环境安装，PID 悬停验证和桌面三维圆形飞行演示均通过；尚未实现本文的移动平台任务和 DrQ-v2 降落训练。安装验收记录见指南第 14.1 节。

更新日期：2026-10-10。
