# 移动平台降落与 DrQ-v2：本机运行说明

目标文献：**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**，
Pawel Ladosz 等，IEEE RA-L，2024，DOI：[10.1109/LRA.2024.3379837](https://doi.org/10.1109/LRA.2024.3379837)。

本仓库现在提供基于该文献搭建的任务和 DrQ-v2 CPU 实现。它是带有明确补充配置的复现版本，
不是作者原始代码；完成短训练也不等于复现了论文报告的成功率。
实际训练与独立评估结果见 [运行报告](cpu_experiment_report.md)。

## 1. 本机怎么打开

Windows 桌面有三个入口：

| 入口 | 内容 |
|---|---|
| 移动平台降落演示.cmd | 三维物理仿真 + 实时下视相机；使用读取平台真值的 PID 调试控制器 |
| 查看DrQ短训练策略.cmd | 加载本机已训练的 DrQ-v2 模型；使用图像和无人机自身状态决定动作 |
| 运行DrQ短训练.cmd | CPU 上重新训练 4,000 步，日志保存到 runs/cpu_user_run |

演示回合从平台上方 **2 m** 开始，平台速度固定 **0.4 m/s**，方便看清画面。
它与训练、评估中的 **10±3 m** 初始高度不同，不用于计算论文场景成功率。
每次演示最多运行 30 分钟；关闭相机窗口即可退出整组演示。
不要重复打开同一个入口；重新训练时使用新的输出目录，避免覆盖已有实验。

本机代码目录：`D:\paper_reproductions\posture_control`。
模型保存在该目录下的 `runs/cpu_pilot_lr1e4/latest.pt`，无需联网下载。
权重和大型运行文件没有上传 GitHub；GitHub 提供源码、配置、说明和小型结果记录。

## 2. 在 Ubuntu 终端自己重跑

已经安装的环境名是 `landing-drqv2`。进入代码目录后执行：

```bash
cd /mnt/d/paper_reproductions/posture_control

# 检查奖励、速度控制、平台位移、相机、碰撞及超时
bash run_checks.sh

# 打开移动平台场景；这是调试控制器
bash run_landing_demo.sh --controller oracle

# CPU 训练。8,000 是环境策略步数，不是回合数
bash run_train_cpu.sh --steps 8000 --lr 0.0001 --seed 1 --output runs/my_pilot

# 独立种子评估训练好的策略
bash run_evaluate.sh --controller drq --checkpoint runs/my_pilot/latest.pt \
  --episodes 10 --seed 10001 --output runs/my_evaluation

# 检查不同真实图像是否会改变模型动作
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
python -m landing_repro.audit_policy runs/my_pilot/latest.pt

# 展示自己训练的模型
bash run_landing_demo.sh --controller drq --checkpoint runs/my_pilot/latest.pt
```

训练不使用调试控制器的轨迹、不预填专家经验，也不读取平台真值作为网络输入。
`--controller oracle` 只用于检查任务和提供诊断参考，不是论文中的视觉基线。

## 3. 换电脑安装

本机已验证：WSL2 Ubuntu 20.04、Miniforge Python 3.10.22、PyTorch 2.2.2+cpu、PyBullet 3.2.6。
Ubuntu 20.04 的系统 Python 3.8 没有被替换。
新机器先安装 Miniforge、Git 和可用的 WSLg/桌面图形环境，然后执行：

```bash
conda create -n landing-drqv2 python=3.10 pip -y
conda activate landing-drqv2
python -m pip install --upgrade pip wheel
python -m pip install setuptools==70.3.0
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python -m pip check
```

启动脚本默认 Miniforge 位于 `~/miniforge3`；其他安装路径需要修改脚本中的 `source` 行。
本机的软件渲染设置已写入 `run_landing_demo.sh`，不需要 NVIDIA/CUDA。
无窗口训练不依赖 GUI。可选的桌面截图使用 x11-apps 中的 xwd；截图失败不影响仿真。
Tk 相机窗口使用 Miniforge Python 的 tkinter，系统还需提供可用的 X11 图形库。

## 4. 代码分别做什么

| 文件 | 职责 |
|---|---|
| landing_repro/config.py | 环境、相机、奖励、控制和训练配置 |
| landing_repro/environment.py | Crazyflie 动力学、1×1 m 移动平台、相机、速度控制、奖励及结束条件 |
| landing_repro/agent.py | 图像编码器、Actor、双 Critic、目标网络、随机平移增强和 n-step 回放 |
| landing_repro/train.py | 采样、更新、TensorBoard 日志、检查点、保存与重载检查 |
| landing_repro/evaluate.py | 独立随机场景上的策略评估 |
| landing_repro/checks.py | 有针对性的物理、边界和 bootstrap 验证；含真值调试控制器 |
| landing_repro/audit_policy.py | 固定自身状态，更换真实图像，诊断策略是否忽略视觉 |
| landing_repro/demo.py | 实际 PyBullet 窗口和实时机载相机窗口 |

## 5. 观测、动作和奖励

**观测**：连续 3 帧 84×84 灰度图，加上连续 3 帧“自身世界坐标速度 3 维 + 姿态四元数 4 维”。
共 21 维附加状态；速度按 `[1.5, 1.5, 0.5]` 归一化，四元数统一正负号。
重置时重复首帧，清除上个回合的历史。绝对位置、平台位置和平台速度不进入网络。

**动作**：3 维 `[-1,1]`，对应世界坐标速度 `[±1.5, ±1.5, ±0.5] m/s`。
由底层 PID 转为电机 RPM，实际无人机由 PyBullet 积分运动，没有通过改写无人机位置完成飞行。

**平台**：每个物理步更新运动学位置并设置线速度；有真实碰撞几何。
完整高度场景每回合均匀采样平台速度 0–1 m/s，直线运动，默认航向 0°。

**成功**：无人机接触平台上表面，且各链接 AABB 的水平包络均在平台边界内。
倾斜平台坐标下使用保守包络，边缘或侧面接触判为失败；撞地为失败，30 秒为超时。
这是首次接触判定，没有增加停稳等待、倾角阈值或相对速度阈值。

**奖励**：下降分量、水平接近分量与终止分量组合，含平台可见性门控。
分割真值只用于计算可见性，不进入网络。小惩罚 -0.01、成功 120、水平权重 0.4。
所有解释与额外保护见下一节。

## 6. 与论文不能等同的地方

以下选择已写入配置，不能称为作者公开的原始参数：

1. **频率解释**：选择 300 Hz 物理、60 Hz 底层控制、30 Hz 策略。由此每策略步 10 个物理步，30 秒最多 900 步。这是对表中数字的一种一致解释，未经作者确认。
2. **相机**：90° FOV，安装偏移 `[0.045, 0, -0.008] m`，近裁剪 0.001 m，远裁剪 100 m。选用该偏移避免模型遮挡与接触前相机穿入平台。
3. **21 维状态**：采用 3×7 的历史速度与四元数，这是实现补充，不代表已确认论文图 3 的 21 维组成。
4. **水平奖励括号**：默认采用 `(rho ** ((ds-dxy)/ds)-1)/(rho-1)`。印刷字面形式另有配置分支，但会产生过大的奖励。
5. **实际下降速度保护**：若实际下降速度超过 0.5 m/s，下降奖励取小惩罚。只检查指令速度的首次尝试出现奖励爆炸。该保护是对印刷式的明确补充。
6. **终止优先**：成功和失败事件的终止奖励优先于可见性门控，防止接触时视野变化覆盖成功奖励；这也是自行规定的事件顺序。
7. **速度平滑**：限制水平指令加速度为每轴 3 m/s²、垂直为 1 m/s²。降低策略动作突变引起的控制器失稳；属于补充的底层控制设置。
8. **网络实现**：采用 DrQ-v2 上游的 LayerNorm、双 Q、探索噪声与目标 Critic 软更新，没有 SAC 的熵温度。图像增强采用 replicate padding 后 ±4 像素整数平移。论文图示/正文与此存在差异。
9. **CPU 调试预算**：batch=16、replay=5000、warmup=256、每 4 步更新、n-step=3；这些不是论文的 batch=1024、replay=300000、warmup=10000。
10. **学习率**：保留了按论文表格 0.005 运行的失败/饱和记录；后续 CPU 调试默认 0.0001。不能把这轮结果写成完全照表复现。
11. **范围**：目前为单一背景、无 DTD 域随机化；未实现 FF-IBVS、DDPGLN、Vision SAC、多种子正式对比和实机实验。

## 7. 日志怎么看

- `config.json`：该次运行的实际配置，优先于源码后续修改后的默认值。
- `metrics.jsonl`：每次更新的 Critic/Actor 损失、图像编码器梯度、目标 Q 值。
- `episodes.csv`：每回合结束原因和回报。训练回报增加不能单独证明降落成功率增加。
- `summary.json`：实际步数、更新数、参数变化及保存/重载一致性。
- `latest.pt`：包含网络、优化器和配置。未保存经验回放及所有随机状态，不能据此宣称断点完全等价续训；当前入口是从头训练。
- `image_sensitivity.json`：同样的自身状态下，不同真实图像对应的动作。动作恒定可能表示饱和或忽略视觉，需要结合独立评估判断。
- `failure.json`：训练异常记录。
- `tensorboard/`：可在环境中运行 `tensorboard --logdir runs --host 127.0.0.1 --port 6006` 后查看。

评估中的 `final_relative_speed_m_s` 是结束状态的速度。碰撞会改变速度，它不等于接触前下降速度。
环境还输出 `pre_contact_relative_velocity`，来自接触前最后一个底层控制步，时间误差最多约 1/60 秒。
仍需进一步精细化记录才能分析真正瞬时撞击速度。

## 8. 如何推进正式实验

先要求模型在独立场景中产生可靠接触，并确认图像变化确实影响动作；
再扩大训练预算，固定奖励解释和底层控制，完成 5 个独立种子及每种子 500 回合评估。
随后加入无可见性门控消融和论文的其他基线。扩展前应核实原论文未公开配置，
不要把当前小样本结果与其表 II 直接比较。

参考：[DrQ-v2 上游](https://github.com/facebookresearch/drqv2)、[无人机库](https://github.com/learnsyslab/gym-pybullet-drones)。
许可及方法来源见 [NOTICE](../landing_repro/NOTICE.md)。
