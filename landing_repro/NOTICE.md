# 方法来源与许可

目标论文：Pawel Ladosz, Meraj Mammadov, Heejung Shin, Woojae Shin, Hyondong Oh,
**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**,
IEEE Robotics and Automation Letters 9(5), 4575–4582 (2024).
DOI: https://doi.org/10.1109/LRA.2024.3379837

这里的降落任务、训练入口及评估脚本是本仓库的复现实现，不是论文作者公开的源码。
`agent.py` 参照 Facebook Research 的 DrQ-v2 实现改编，增加了无人机自身状态输入，
使用整数索引实现随机图像平移，并重写了内存回放、训练和模型保存接口。
参考提交：c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef。
来源：https://github.com/facebookresearch/drqv2

新增 `guided.py` / `train_guided.py` 使用训练真值生成示范与在线纠错标签，并加入监督约束、奖励修改和课程学习。这是本仓库的辅助训练改进，不能视为目标论文的纯 DrQ-v2 原始方法。独立测试仍只将图像和无人机自身状态输入策略。
上游 MIT 许可保留如下。

## DrQ-v2 upstream license

MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 无人机动力学依赖

运行时使用 learnsyslab/gym-pybullet-drones（MIT），固定提交
50d1a958def27121bcf63f3cc753049fee8e71e5。该依赖单独安装，没有把其模型和源码复制到此目录。
https://github.com/learnsyslab/gym-pybullet-drones

2026-10-11: smooth.py adds own-command memory (30D proprioception), command-smoothness penalties, command-jerk limits and soft-contact shaping. This is a new demonstration-assisted adaptation, not the original paper method or a full CAPS implementation. Privileged platform truth is used only for training labels/rewards and evaluation metrics; evaluated actions use pixels and own state/command memory.

2026-10-11: Optional prediction guidance adds RGB marker PnP, a linear constant-velocity Kalman filter, time-decayed heuristic confidence, and horizontal prediction-progress rewards during partial/failed detection. This auxiliary estimator is neither the policy's hidden output nor a new policy input. It does not use platform truth; physical contact and soft-landing training terms still use simulator state. This is a further adaptation, with learned performance awaiting controlled evaluation.

2026-10-11: Optional near_landing_assist retains bounded near-field estimates, adds estimated-state feedback and terminal action constraints, and rewards centering with low relative speed. Executed actions are a hybrid of network requests and an auxiliary controller. This is not a pure learned policy or the original paper algorithm. Implementation checks used frozen weights and no optimization; training resumed only after the user's subsequent explicit start instruction.
