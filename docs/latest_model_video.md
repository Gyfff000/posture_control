# 最新模型展示视频

参考文献：**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**，Pawel Ladosz 等，IEEE RA-L 2024，DOI：10.1109/LRA.2024.3379837。

本视频录制实际 PyBullet 仿真，不是概念动画。实现为 DrQ-v2 与示范辅助、动作平滑、视觉中心估计及近距离控制器的结合，不能称为论文纯算法的完整复现。

## 录制内容

- 最新已保存的 95000 步检查点，使用其保存的环境与控制配置。
- 阶段 1 变速直线平台，配置速度范围 0.08–0.35 m/s。
- 冻结网络，无探索噪声，无优化更新。
- 预先固定种子 101001、101002、101003；全部展示，不按结果筛选。
- 全景、无人机模型近景、机载相机、真实飞行轨迹、辅助估计中心、置信度和最终事件。
- 1280×720，H.264 MP4，15 fps，按仿真时间 1 倍速播放。开头与结果画面另设停留时间。
- 外部视角、实际平台位置和误差只用于展示/评估，不输入视觉策略或中心估计器。

## 重新录制

在已经安装依赖的 WSL 环境中：

```bash
conda activate landing-drqv2
cd /mnt/d/paper_reproductions/posture_control
python -m landing_repro.record_demo \
  --checkpoint runs/demo_latest_20261011_232119/model_snapshot.pt \
  --output runs/my_video \
  --seeds 101001 101002 101003
```

使用本机检查点路径；其他机器需准备自己的兼容模型。环境已安装 Pillow、OpenCV 和 imageio-ffmpeg；中文字体目前使用 Windows 的微软雅黑，纯 Linux 环境需修改脚本字体路径。

输出 `latest_model_test.mp4`、`test_report.json`、每回合 CSV 和最终帧。报告记录检查点 SHA256、环境参数、完整结果与零优化更新检查。模型快照和大视频不上传 GitHub。

三回合仅用于展示，不能据此估计总体成功率。各回合的软着陆需满足物理接触、水平接触相对速度不超过 0.2 m/s、竖直接触速度不超过 0.18 m/s，以及倾角条件。
## 本次结果（2026-10-11）

三回合均成功并满足软着陆判据，水平误差分别约 0.061、0.252、0.154 m。视频时长 40.67 秒，完整解码检查通过。该三回合结果仅用于展示，不能称为模型总体成功率100%。
