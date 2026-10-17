# 无显卡阶段的验证结果与服务器交接

更新：2026-10-17。本次推进的是第一篇论文的环境检查、冻结测试、训练设备入口和交接准备。

## 1. 对应论文与当前状态

**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**  
Pawel Ladosz等，IEEE Robotics and Automation Letters，2024；DOI：10.1109/LRA.2024.3379837。

当前为独立论文实现，奖励公式等歧义尚未得到作者确认，不能宣布完成原论文的数值复现。旧的辅助视觉模型和成绩另行保留，不能混入本实验。

本机长训已保存停止：

| 内容 | 保存值 |
|---|---|
| 实验目录 | `runs/paper_ladosz2024_guarded_seed1_20261017_191847` |
| 环境交互步数 | 12,728 |
| 网络更新数 | 1,365 |
| 完整训练回合 | 34 |
| 训练中成功回合 | 0 |
| 可用经验条数 | 12,724 / 300,000 |
| 模型文件 | 实验目录内的 `latest.pt` |
| 状态 | stopped_and_saved；不继续占用本机长期训练 |

保存内容含网络、优化器、随机状态和磁盘经验池。物理回合不恢复到中断瞬间；待写入的n-step尾段会丢弃。模型文件的SHA256及软件版本见交接清单。

## 2. 这次实际验证了什么

### 环境正确性

以下检查均通过：

- 世界坐标速度的轴向、xyzw四元数及21维历史编码。
- 移动平台后，图像改变，而完整21维自身状态不混入平台真值。
- 机体姿态改变时，机载相机画面随之改变。
- 回合重置后，图像和自身状态历史均清空。
- 连续30个策略步中，匀速平台的位置与速度、物理时间一致。
- 0°和45°平台朝向下，中心接触判为success，擦边为platform_edge，平台外撞地为ground，共6个接触案例。
- 超时是truncated，碰撞是terminated；结束后必须重置。
- 距离奖励边界、LOS门控和接触奖励符合当前显式实现。

**接触案例通过人为设置碰撞位置验证代码，不是训练模型完成了6次降落。** 它们只能说明当前判定逻辑符合本实现，不能证明与作者的接触容差相同。

磁盘回放、梯度更新、检查点重载、1024批量更新和CPU断点恢复也通过回归检查。卷积激活按32条分块重计算，BatchNorm和损失仍使用完整1024条批量。

### 当前模型的冻结测试

使用保存的12,728步模型，固定测试种子901001—901003；测试时使用策略均值，无探索噪声、无参数更新。

| 测试种子 | 结果 | 落地时间（秒） | 落地时水平误差（米） |
|---|---|---:|---:|
| 901001 | 撞地 | 22.53 | 60.85 |
| 901002 | 撞地 | 17.80 | 48.64 |
| 901003 | 撞地 | 24.03 | 50.50 |

本次为 **0/3成功**，是小样本诊断，不是论文要求的500回合最终成功率。测试前后检查点哈希一致；设备入口修改前后，相同种子的终止事件、时间、误差和回报完全相同。

### 找到的关键问题：动作饱和

三个测试回合中，策略一直给出归一化动作约 `[-1, +1, -1]`，即水平速度目标约 `[-1.5, +1.5]` 米/秒、下降速度目标约 `-0.5` 米/秒，因而持续飞偏。

进一步从保存的经验池抽128条观测检查：

- 参数均为有限值，没有发现NaN或Inf。
- 推理模式和训练批量统计模式中，动作分量的饱和比例均为100%。
- 推理模式最大绝对logit约193，sigmoid导数均值约2.57e-38，说明输出已进入难以通过梯度改变的区域。
- 因此，问题不只是界面不动，也不能仅凭增加训练时间保证改善。

尚未确定根因。需要核对论文学习率、动作输出、BatchNorm统计、奖励尺度和训练细节。不能根据一次诊断就断言其中某一项有错。后续如比较较小学习率等配置，必须另建“敏感性实验”并保留论文参数基线，不把调参结果冒充逐项复现。

## 3. 现在可以自己操作的入口

桌面文件夹 **无人机训练入口**：

- **冻结测试当前论文模型.cmd**：显示三种视角，运行3个冻结测试回合，不训练。结束后窗口关闭，报告和机载截图保存在 `results/evaluations/` 下；画面标题为FROZEN TEST。
- **开始论文原方法独立实验.cmd**：会从已保存进度继续训练。当前建议先完成动作饱和的短程排查，再开始长训。

在WSL激活已有环境、进入项目后，也可以运行：

```bash
# 环境验收
python -m paper_repro.environment_audit

# 冻结测试：output必须是新目录；增加--gui可显示桌面窗口
python -m paper_repro.evaluate \
  --checkpoint runs/paper_ladosz2024_guarded_seed1_20261017_191847/latest.pt \
  --output results/evaluations/my_first_test --episodes 3 --device cpu

# 只诊断输出饱和，不训练、不改原模型和经验池状态
python -m paper_repro.policy_diagnostic \
  --run runs/paper_ladosz2024_guarded_seed1_20261017_191847 \
  --output results/my_policy_diagnostic.json
```

## 4. 服务器准备：已加入口，GPU执行待实测

训练、冻结测试和实验清单现支持 `--device cpu|cuda|auto`：

- 明确指定cuda但不可用时会报错，防止以为用了GPU、实际仍在CPU上运行。
- 网络、观测批量、动作输出、优化器恢复均加入设备处理。
- GPU训练会保存CUDA随机状态。同设备恢复和跨设备恢复应分别记录；跨CPU/GPU不能保证相同随机轨迹。
- 本机已验证CPU路径、自动选择CPU、无GPU时报错和断点恢复。**没有GPU，CUDA分支尚未完成运行验证，也没有可靠的服务器耗时数据。**
- PyBullet物理和软件渲染仍在CPU上；GPU用于网络计算，不代表全部计算都搬到显卡。

### 第一篇的安装模板

在服务器已准备好兼容NVIDIA驱动、Git和Conda后，使用独立环境：

```bash
conda create -n landing-drqv2 python=3.10 -y
conda activate landing-drqv2
git clone https://github.com/Gyfff000/posture_control.git
cd posture_control

# 示例为PyTorch 2.2.2的CUDA 12.1构建，须先确认服务器GPU/驱动兼容。
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-paper.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m paper_repro.environment_audit

# 先验证完整1024批量能在GPU更新，结果必须为passed且device=cuda。
python -m paper_repro.gradient_probe --device cuda --batch 1024 --output gpu_probe.json
```

PyTorch 2.2.2的CPU、CUDA11.8和CUDA12.1安装选项已核对[官方历史版本页面](https://pytorch.org/get-started/previous-versions/#v222)。这些命令是安装模板，尚未在服务器执行；较新GPU可能需要更新的PyTorch，届时须重新检查版本兼容和数值回归。

GPU合成检查通过后，应先在新目录做短程、无图形训练试跑，核对内存、loss和动作饱和，才安排完整实验；合成更新速度不能直接当作端到端训练速度。

```bash
# 新建试跑，包含10000步随机预热和少量真正更新。
# 无--gui，服务器无需桌面；这不是完整论文训练。
python -m paper_repro.train --device cuda --output runs/gpu_pilot_seed1 \
  --seed 1 --max-steps 10040 --eval-every-episodes 0 --final-eval-episodes 0

python -m paper_repro.policy_diagnostic --run runs/gpu_pilot_seed1 \
  --output gpu_pilot_policy_diagnostic.json
```

上面10040步试跑只有约21次更新，主要验证执行；还不足以确认长期稳定。正式实验前应单独安排更长的稳定性对照，并保留全部配置。

正式训练入口不传--max-steps时使用4000回合/3.6M步上限；5种子与消融清单可先生成审核：

```bash
python -m paper_repro.suite --root runs/paper_gpu_suite --device cuda
# 只有确认短程诊断正常后，才加--run实际执行。
```

### 如要迁移当前检查点

必须传输整个实验目录，包括 `config.json`、`latest.pt`、`replay/*.npy`、`replay/state.json`、日志和summary。只传latest.pt可以测试模型，不能恢复原经验池训练。经验池逻辑文件约12.8GB，复制前需确认空间；可压缩传输。

模型与经验池不提交GitHub。源码、检查报告和说明提交GitHub。服务器恢复命令如下，但当前检查点存在动作饱和，不建议未经诊断就长训：

```bash
python -m paper_repro.train --device cuda \
  --output runs/paper_ladosz2024_guarded_seed1_20261017_191847 \
  --seed 1 --variant baseline --reward-interpretation normalized \
  --vertical-reward actual_guard --resume
```

## 5. 第二篇PerchRL的边界

**PerchRL: Vision-Based Agile Perching on Inclined Platforms under Rapid and Irregular Motion**，arXiv:2606.03441v3。

当前仍只有CPU组件和协议准备；本次没有开始PerchRL训练。现有DrQ-v2环境不是PerchRL的OmniDrones/Isaac Sim环境。其GPU、原任务源码、完整PPO/TCN配置和三维接触任务等缺口见 `full_reproduction_plan.md`。第一篇的服务器安装模板不能用于宣称已经配好PerchRL。

## 6. 证据文件

在 `results/paper_audit_20261017/`：

- `environment_acceptance.json`：环境验收与6个注入接触案例。
- `component_checks.json`：经验池、梯度和组件回归。
- `device_and_resume_check.json`：设备选择与短程断点恢复。
- `device_batch1024_probe.json`：设备接口修改后的1024批量CPU更新。
- `evaluation_regression.json`：修改设备接口前后的冻结测试一致性，以及隐藏窗口渲染检查。
- `frozen_12728_actions/evaluation.json`：3回合冻结测试、动作均值和饱和比例。
- `action_saturation_diagnostic.json`：128条保存观测上的动作饱和诊断。
- `server_handoff_manifest.json`：源文件指纹、软件版本、检查点哈希和保存状态核对。
