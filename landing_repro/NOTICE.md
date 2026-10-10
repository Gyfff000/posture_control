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
