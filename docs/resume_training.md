# 暂停后从哪里继续训练

2026-10-11：当前已停止，没有训练进程运行。最后保存累计 44876 个交互步、11161 次参数更新，运动阶段为 1（直线随机变速）。

## 模型位置

本次最新模型：

```text
D:\paper_reproductions\posture_control\runs\motion_curriculum_20261011_164731\latest.pt
```

里面包含网络权重、优化器和累计计数。该目录中的 `summary.json` 记录本轮停止情况，`episodes.jsonl` 和 CSV 记录实验过程。本轮尚未完成新阶段的完整验证，`best.pt` 是结束时的备用保存，不能视为经验证选出的最佳模型。续训使用 `latest.pt`。

## 方法一：双击桌面入口

桌面文件：**继续变速平台训练.cmd**。

1. 双击该文件。
2. 保持打开的终端窗口运行，等待训练界面出现。
3. 界面应显示 `Motion 1: variable-speed straight`，累计步数从保存值继续增长。
4. 使用界面的 `Stop and save model` 可以保存并停止；`Pause` 是保留进程的临时暂停，`Continue` 恢复。

不要在训练时关闭启动它的终端。脚本有重复进程检查，不应连续启动多个训练。

仓库中的同一入口为 `resume_variable_motion_training.cmd`。桌面的“显示当前无人机训练窗口.cmd”只唤回已运行的窗口，不能启动已停止的训练。

## 方法二：在 Windows PowerShell 输入

```powershell
wsl -d Ubuntu-20.04 --cd /mnt/d/paper_reproductions/posture_control --exec bash start_variable_motion_training.sh
```

脚本自动进入已有 Python 环境，不需要手动重新安装或激活 Conda。

## 如何选择模型

脚本读取 `runs/current_smooth_training.txt` 指向的最近一轮目录，优先选择其中的 `interrupted.pt`，没有时选择 `latest.pt`。当前目录没有 `interrupted.pt`，所以会选择上述 44876 步的 `latest.pt`，继承运动阶段 1。

每次启动产生一个新的运行目录，保留旧模型和日志。验证达标后自动升级到随机 B 样条阶段；每次启动最多再增加 80000 个交互步，也可能按性能条件提前停止。

经验缓存、随机数状态和中断中的飞行回合会重置，因此不是逐帧完全一致恢复。网络权重、优化器和累计计数继续使用。

如需明确从本次模型开始，可以在 WSL 中执行：

```bash
cd /mnt/d/paper_reproductions/posture_control
bash start_variable_motion_training.sh runs/motion_curriculum_20261011_164731/latest.pt 1
```

本页和桌面入口已创建，但没有执行续训。
