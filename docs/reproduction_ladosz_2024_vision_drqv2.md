# 复现指南：Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning

> 文档日期：2026-10-10。依据用户提供的论文全文、论文图表和所列官方项目源码整理。
> 已在用户的 WSL2 Ubuntu 20.04 上完成 CPU 环境、移动平台任务、机载相机、DrQ-v2 短训练与评估流程。实际运行与偏差见 [CPU 运行说明](running_cpu_landing.md) 和 [实验报告](cpu_experiment_report.md)；尚未复现论文规模的训练及其成功率。
> 下文标记“论文给出”的内容来自原文；标记“建议”“推断”“待核实”的内容是实施方案，不能作为作者原始配置引用。

## 1. 明确复现的是哪篇文献

- **英文题名**：Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning
- **中文参考译名**：基于视觉深度强化学习的移动平台自主降落。
- **作者**：Pawel Ladosz、Meraj Mammadov、Heejung Shin、Woojae Shin、Hyondong Oh。
- **期刊**：IEEE Robotics and Automation Letters，2024，9(5)：4575-4582。
- **DOI**：[10.1109/LRA.2024.3379837](https://doi.org/10.1109/LRA.2024.3379837)。
- **发表信息**：[IEEE Xplore](https://ieeexplore.ieee.org/document/10476692/)。
- **机构**：第一作者署名英国曼彻斯特大学，其他作者署名韩国蔚山科学技术院（UNIST）。
- **本文范围**：复现论文 V-A 节的仿真实验。V-B 节的 Crazyflie 实机实验需要硬件，暂不实施。

这篇论文研究无人机如何利用下视相机图像和自身速度、姿态，在移动的水平平台上自主降落。它采用 **DrQ-v2 图像强化学习 + 自定义奖励 + 三维速度控制 + 域随机化**。

它与此前讨论的 PerchRL、DMU_UAV_land 教学仓库属于不同工作。本文使用 **PyBullet / gym-pybullet-drones**，没有要求使用 Isaac Sim、Isaac Lab、PPO、AprilTag、EKF 或 TCN。把输入换成 AprilTag 解算的位置，也会改变本文的端到端视觉方法。

## 2. 要复现的系统是什么

整个系统应按下面的链路搭建：

```text
PyBullet 中的 Crazyflie 和移动平台
    -> 固定在机体上的下视相机
    -> 灰度图像、缩放到 84×84、堆叠 3 帧
    + 无人机自身速度和姿态
    -> 图像编码器 + DrQ-v2 Actor
    -> x、y、z 三维目标速度
    -> 底层速度/姿态控制器
    -> 电机转速与无人机物理运动

训练时另外使用仿真真值：
    -> 平台是否可见、相对距离、接触结果
    -> 奖励和终止信号
    -> 经验回放、Critic 更新、Actor 更新
```

**重要边界**：仿真真值可以用于计算奖励和评价指标；原方法的 Actor 不应直接读取平台位置、平台速度或相对位置。论文所说的视觉方法还使用无人机自身速度和姿态，并非只有一张图像。

本文的动作是目标速度，不是直接输出电机 RPM，也不是推力和角速度 CTBR。底层控制器仍然需要实现速度跟踪。

## 3. 没有独立显卡时能做到哪一步

| 阶段 | 无独立显卡是否可做 | 交付结果 |
|---|---|---|
| 安装 PyBullet，运行 Crazyflie 飞行示例 | 可以尝试 | 真实仿真软件窗口、飞行日志 |
| 添加水平移动平台、下视相机、接触判定 | 可以 | 场景画面、机载相机图像、回合日志 |
| 验证奖励函数与动作映射 | 可以 | 边界检查、奖励分量和输入输出记录 |
| 用规则/PID 控制器测试环境 | 可以 | 环境与控制接口验证；需标明控制器类型 |
| DrQ-v2 少量前向推理、梯度更新和短流程检查 | 可以，小批量更合适 | 证明数据流和更新流程可用 |
| 从像素训练至论文规模，并完成多种方法、5 个种子的对比 | CPU 理论上可计算，时间与内存负担较大 | 建议有 GPU 后完成；不承诺训练时长 |
| 论文实机实验 | 当前不能 | 需要 Crazyflie、相机等硬件 |

PyBullet 提供 CPU 渲染器和 OpenGL 可视化，因此“没有 NVIDIA 显卡”不等于“不能打开 PyBullet 仿真”。不过 GUI 仍依赖可用的图形驱动与显示环境。[PyBullet 项目说明](https://pypi.org/project/pybullet/3.2.6/)

对目前约 16 GB 内存的电脑，优先完成环境、相机、奖励和短流程检查。完整训练建议另备 GPU，并采用节省内存的经验回放实现。**GPU 规格和训练时间未在这篇论文中给出，不能推断作者用了什么显卡，也不能把其他项目的“55 分钟”搬过来。**

## 4. 软件环境与版本选择

### 4.1 论文明确给出的软件

| 项目 | 论文信息 |
|---|---|
| 物理仿真 | PyBullet |
| 无人机模型库 | gym-pybullet-drones，参考文献 [23] |
| 无人机 | Crazyflie；原文讨论 Crazyflie 2.1 |
| 学习算法 | DrQ-v2，参考文献 [21] |
| 图像数据 | 下视相机灰度图，84×84，3 帧堆叠 |
| 域随机化背景 | 从 DTD 数据集中选取约 400 张纹理 |

原文没有完整列出操作系统、Python、PyTorch、CUDA、相机内参、物理频率、底层控制器配置及软件提交号。

### 4.2 本指南建议的起步环境

建议优先采用 **Ubuntu 22.04 + Python 3.10 + CPU PyTorch**。Windows 用户可以通过 WSL2 的 Ubuntu 22.04 使用这一环境。原生 Windows 也可以研究，但 PyBullet 的安装可能需要 C++ 编译工具，初学阶段用 Linux 路线更容易保持一致。

下面是一套用于起步的固定版本组合，**是本指南的实施组合，不是作者公开的原始环境**。其中 Python、PyTorch、PyBullet 与无人机库已在 WSL2 Ubuntu 20.04 中通过下述运行验证；DrQ-v2 的任务适配和完整训练尚未验证。

| 依赖 | 建议版本/来源 |
|---|---|
| Python | 3.10 |
| setuptools | 70.3.0，保留旧版仿真库使用的 pkg_resources |
| gym-pybullet-drones | 提交 `50d1a958def27121bcf63f3cc753049fee8e71e5`，2024-03-24 |
| PyBullet | 3.2.6 |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| PyTorch | 2.2.2，先使用 CPU wheel |
| Gymnasium | 0.28.1 |
| Stable-Baselines3 | 2.0.0，仅为该版本仿真库依赖，不替代 DrQ-v2 |
| Matplotlib / transforms3d | 3.7.5 / 0.4.1 |
| DrQ-v2 参考实现 | `facebookresearch/drqv2`，提交 `c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef` |

选择 2024 年的仿真库提交，是为了避免最新版依赖不断变化。**不能声称这一提交就是作者使用的版本**。截至文档日期，上游 main 已升级到另一套 Python 3.12 等依赖，安装本指南时不要把 main 的安装要求和固定提交混用。

已核对上述固定提交的 [pyproject.toml](https://github.com/learnsyslab/gym-pybullet-drones/blob/50d1a958def27121bcf63f3cc753049fee8e71e5/pyproject.toml) 和示例入口。PyPI 提供 PyBullet 3.2.6 的 Linux x86_64 / CPython 3.10 wheel；该发行版没有列出 Windows wheel，Windows pip 可能转为源码构建。[发行文件](https://pypi.org/project/pybullet/3.2.6/#files)

DrQ-v2 官方仓库的原始环境包含 Python 3.8、CUDA、MuJoCo/dm_control 等旧依赖。那些配置主要服务于其原始 benchmark；复现本文的 PyBullet 任务时，需要复用算法并改写环境适配层，不能认为运行上游默认训练命令就会训练无人机降落。

## 5. 第一步：先运行一个真实的三维飞行示例

本节命令可以直接用于安装和运行上游飞行示例。**示例飞行成功仅代表仿真底座可用，尚不代表完成论文降落实验。**

### 5.1 Windows 中准备 WSL2（已有 Ubuntu 时跳过安装）

在 PowerShell 查看现有发行版：

```powershell
wsl --list --verbose
```

如果没有 Ubuntu 22.04，可在管理员 PowerShell 安装：

```powershell
wsl --install -d Ubuntu-22.04
```

按系统提示完成重启和首次创建 Linux 用户。随后打开 **Ubuntu 22.04 终端**，后续命令都在 Ubuntu 内执行。Windows 与 Ubuntu 的 Python 环境互相独立，不要把 Windows Python 3.12 的包安装位置混入 Ubuntu。

WSL 图形窗口需要 WSLg 支持。若 GUI 不可用，可先完成无窗口的物理仿真及 CPU 相机渲染；导出的相机图片与三维交互窗口是两种不同交付物。

### 5.2 Ubuntu 中创建独立环境

```bash
sudo apt update
sudo apt install -y python3.10-venv python3-pip git build-essential libgl1 libglib2.0-0

mkdir -p ~/uav_landing_repro
cd ~/uav_landing_repro
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install setuptools==70.3.0
python --version
```

期望显示 Python 3.10.x。以后每次重新打开 Ubuntu 终端，先执行：

```bash
cd ~/uav_landing_repro
source .venv/bin/activate
```

### 5.3 安装 CPU 依赖和固定版本仿真库

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5 transforms3d==0.4.1 pybullet==3.2.6 gymnasium==0.28.1 stable-baselines3==2.0.0 pytest==7.4.4

git clone https://github.com/learnsyslab/gym-pybullet-drones.git
cd gym-pybullet-drones
git checkout 50d1a958def27121bcf63f3cc753049fee8e71e5
python -m pip install -e .
python -m pip check
```

CPU PyTorch 安装方式参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。若依赖解析失败，保存完整错误并先解决冲突，不要忽略错误继续跑训练。以后换 GPU 环境时，应按具体显卡和驱动选择官方支持的 PyTorch/CUDA wheel。

检查导入：

```bash
python -c "import torch, pybullet, gymnasium, gym_pybullet_drones; print('imports OK'); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

CPU 环境输出 `CUDA: False` 正常。

### 5.4 打开单机飞行窗口，再检查无窗口模式

在 `gym-pybullet-drones` 根目录运行：

```bash
python gym_pybullet_drones/examples/pid.py --num_drones 1 --gui true --plot false --duration_sec 12 --obstacles false
```

应看到 PyBullet 三维窗口中的 Crazyflie 按 PID 参考轨迹运动。这里展示的是仿真软件界面。

再运行一次无窗口模式，确认后续可批量采样：

```bash
python gym_pybullet_drones/examples/pid.py --num_drones 1 --gui false --plot false --duration_sec 12 --obstacles false
```

可选检查速度控制示例：

```bash
python gym_pybullet_drones/examples/pid_velocity.py --gui true --plot false --duration_sec 5
```

这个固定版本的速度示例内部创建 4 架无人机；其接口也不是本文的三维速度接口。它用于学习速度如何经 PID 转换成电机转速，不可直接拿来作为论文训练环境。

最后保存环境清单：

```bash
python -m pip freeze > ../requirements-resolved.txt
git rev-parse HEAD > ../gym-pybullet-drones-commit.txt
```

**本步验收**：导入正常、GUI 或无窗口物理运行正常、记录实际包版本。若没有 GUI，先排查显示和 OpenGL；不要靠修改奖励或训练参数解决图形初始化问题。

## 6. 第二步：搭建本文的移动平台降落环境

本节描述目标任务。当前已提供 `landing_repro/environment.py`、训练和评估入口；实际可运行命令见 [CPU 运行说明](running_cpu_landing.md)。本机模型保存在 runs 目录，未上传大型权重。以下论文要求与实现补充仍须区分。

### 6.1 场景和平台运动

按论文 II 节和 V-A 节：

1. 使用 Crazyflie 四旋翼物理模型。
2. 添加 **1 m × 1 m 的水平平台**，中间为带圆圈的 H 标志。
3. 无人机初始位置在平台上方 10 m，添加最多 ±3 m 的偏移；需把各坐标方向的采样范围写进配置。原文有“each direction”表述，不能未说明就固定或随机高度。
4. 平台每回合匀速向前移动，速度在 0-1 m/s 范围采样。
5. 域随机化版本加入平台初始航向 ±15°。
6. 原文以直线匀速为主要场景，8 字形、随机 B 样条、倾斜平台属于后续扩展，不是本文原始实验。

平台应具有可接触的碰撞几何。若用运动学刚体更新平台位置，要同步其速度，并检查接触响应，防止“平台视觉上移动、接触物理却静止”。不要通过每步改写无人机位置来模拟速度控制。

### 6.2 时间与频率

论文 Table I 给出：

- 每回合 30 s，括号标为 900 steps。
- aggregated steps = 10。
- 域随机化时，相机观测与策略决策间隔为 0.1-0.4 s。

这三项没有完整解释各自对应物理步、控制步还是策略步。**不能直接把 900 次策略动作和每次 0.1-0.4 s 同时硬编码成 30 s 回合。**

建议先采用固定物理频率及控制频率，另设独立策略时钟。起步可用 240 Hz 物理、48 Hz 底层控制，它们来自上游示例而不是本文已确认配置。每次策略决策后重复执行底层控制，累计实际仿真秒数，到 30 s 时超时终止。随机间隔量化到物理步时记录实际间隔。原始频率和 aggregated steps 的含义列入待作者核实项。

### 6.3 相机与观测

- 相机安装在机体上，向下看，随无人机姿态变化；不能使用固定世界坐标俯视图代替。
- 每次观测由 RGB 转灰度、缩放为 84×84，再按时间顺序堆叠 3 帧，形成 `(3, 84, 84)` 图像。
- 图像存储建议使用 uint8；进入网络时再转换和归一化。
- 拼接无人机自身速度和姿态，不加入平台真值。
- 每次 reset 清空历史；第一帧如何填满历史需要固定规则，例如重复首帧，并记录为自行补充的实现。
- 相机 FOV、内外参、近远裁剪面和安装偏移未在原文完整给出；需要选择并写入配置。
- 调试时可使用 `getCameraImage(..., renderer=p.ER_TINY_RENDERER)` 生成 CPU 图像。在无 GUI 模式下也可取图，避免把 TinyRenderer 错当成 CUDA 渲染。

**验收**：分别保存高空、接近平台、平台出视野时的机载相机图像；确认姿态变化会改变视野，帧堆叠没有跨回合污染。

### 6.4 动作与底层控制

论文式 (6) 的速度映射为：

```text
v_cmd = 2 × (a - 0.5) × [1.5, 1.5, 0.5]
a 的每一维在 [0, 1]
```

对应三维速度上下界为 x/y ±1.5 m/s，z ±0.5 m/s。采用 z 轴向上的约定时，负 z 速度表示下降。还需要核实“水平 1.5 m/s”是分轴限制还是水平合速度限制。

官方 DrQ-v2 的动作通常在 [-1, 1]，因此可以等价地使用：

```text
v_cmd = a_drq × [1.5, 1.5, 0.5]
```

只能选择其中一种映射，避免重复缩放。若需要限制水平合速度，须记录这种额外处理。

上游 `VelocityAviary` 使用“方向 + 速度比例”的 4 维动作，并带有自己的速度上限；不能把本文 3 维动作直接传进去。需要写速度适配层，利用底层 PID 将目标速度转为 RPM，检查正负方向、饱和和速度跟踪效果。

### 6.5 成功、失败和超时判定

论文成功标准是：

1. **整个无人机在平台边界内**；
2. 无人机接触平台。

仅判断质心进入 1 m 正方形不够，需要考虑无人机尺寸和姿态下的水平包络。可用仿真接触点、无人机模型的几何边界来实施。

失败包括撞到地面；超时由最大回合长度结束。平台边缘碰撞、侧撞、机体翻覆是否计为失败，原文未给完整规则，需要在自己的实现中显式规定并报告。

不要额外引入 PerchRL 的角度、相对速度阈值后，继续把成功率称为本文同一标准的成功率。可以单独记录接触速度、倾角，作为扩展安全指标。Gymnasium 实现中区分 `terminated`（成功/失败）和 `truncated`（超时）；经验回放处理 bootstrap 时也要区分。

## 7. 第三步：实现并核对奖励函数

论文 IV 节式 (7)-(10) 的设计思想是：

- 平台可见时，组合水平接近奖励、下降奖励和着陆奖励。
- 平台不可见时，只给小的负奖励。
- 鼓励以安全速度下降，避免仅奖励低高度导致长期悬停。
- 成功着陆给大正奖励，失败给小负奖励，避免策略因为过度惩罚而拒绝下降。

论文参数见 Table I：

| 参数 | 值 |
|---|---|
| 小惩罚 Rp | -0.01 |
| 水平、下降、终止权重 α、β、γ | 0.4、1、1 |
| 下降奖励参数 ζ | 30 |
| 水平奖励参数 ρ | 20 |
| 水平奖励启动距离 ds | 10 m |
| 安全下降速度 vl | -0.5 m/s |
| 成功着陆奖励 Rs | 120 |

总奖励：

```text
若平台可见：R = β Rz + α Rxy + γ Rl
否则：R = Rp
```

这里奖励权重 γ=1 与强化学习折扣 γ=0.99 是两个不同参数，代码中应分别命名。

**必须处理的原文疑点：**

1. 式 (8) 用“指令速度 vc”判断是否越过安全范围，指数中却用实际速度 v。不要不加说明地把它们合并成同一个变量。指令限幅也不能保证实际接触速度永远不超限。
2. 式 (9) 印刷指数为 `ρ^(ds - dxy/ds)`。按 ds=10 直接计算会产生很大的奖励，并在边界处不连续。常见、符合归一化意图的形式是 `ρ^((ds-dxy)/ds)`，但这只能作为**待确认的解释**，不能悄悄改写后称为作者公式。
3. “平台可见”没有给出可见面积阈值。训练时可以用分割标签/投影真值定义可见性，但不能把该真值作为额外 Actor 输入。遮挡和边缘仅露出少量像素的规则也应固定。
4. 不可见时的总奖励门控是否会覆盖终止奖励，式 (7) 与实现的事件顺序需要明确。

建议把式 (9) 的两种解释做成可切换配置，先画出或计算奖励边界；若采用归一化形式，在报告中明确这是自行解释，并保留公式疑点。获得作者代码或确认前，不把数值对齐视为已经解决。

最低检查项：

- 中心、奖励边界和边界外的水平奖励。
- 悬停、正常下降、上升、超速下降时的奖励。
- 平台可见/不可见时的总奖励。
- 成功、失败、超时三类事件的奖励与结束行为。
- 所有分量均为有限数，且没有出现错误公式导致的巨大数量级。

**本步验收**：逐步记录 `visible、Rz、Rxy、Rl、R_total`，可以解释每一种边界情形。

## 8. 第四步：接入 DrQ-v2，先检查数据流

### 8.1 参考代码与需要改动的地方

[官方 DrQ-v2](https://github.com/facebookresearch/drqv2) 是算法参考实现，不是本文作者的完整降落代码。本文 PDF 没有提供明确的任务代码链接；截至文档日期的检索未定位到能确认归属作者的完整降落任务源码，不能据此断言代码从未公开。

下载固定参考提交：

```bash
cd ~/uav_landing_repro
git clone https://github.com/facebookresearch/drqv2.git
cd drqv2
git checkout c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef
```

在第 5 节创建的同一个 Python 环境中，可为算法适配准备以下额外依赖：

```bash
python -m pip install hydra-core==1.3.2 omegaconf==2.3.0 dm-env==1.6 termcolor==2.4.0 pandas==1.5.3 imageio==2.34.0 imageio-ffmpeg==0.4.9 tensorboard==2.16.2 opencv-python==4.9.0.80
python -m pip check
```

这仍是本指南建议的适配环境，不是作者原始环境。若保留上游 Hydra 配置中的 submitit launcher，还需处理该插件；本地单进程训练可以改为基础 launcher。若保留 `train.py` 顶部的 `import dmc`，它会继续引入 dm_control，因此要在自定义训练入口中替换对应导入和环境构建，不能仅安装这些包就认为适配完成。

需要实现或改动：

1. 用自定义 PyBullet 降落环境替换官方的 dm_control 环境入口。
2. 让数据结构同时携带图像与无人机自身速度/姿态。
3. 修改编码器后的特征拼接、Actor 和 Critic 输入。
4. 对接三维速度动作与底层控制器。
5. 接入本文奖励、接触判定和域随机化。
6. 保存训练日志、配置、种子、最佳模型和评估轨迹。

上游默认训练任务是 MuJoCo/DeepMind Control，直接运行 `python train.py task=quadruped_walk` 不会得到本文实验。也没有证据表明 Stable-Baselines3 的默认 PPO、SAC 配置能替代本文的 DrQ-v2。

### 8.2 网络结构与实现差异

论文 Fig. 3 给出的编码器是 4 层卷积：32 通道、3×3，第一层 stride=2，后续 stride=1，ReLU。后续特征维度为 50，隐藏层为 512、512，Actor 输出 3 维速度动作。

需要注意：

- Fig. 3 的线性层输入写为 **39221**。84×84 输入经上述 CNN 得到 39200 维图像特征，多出的 21 维可能与额外状态有关，但正文未明确其组成和堆叠方式，不能直接认定为某一种 IMU 配置。
- Fig. 3 标注 BatchNorm1d；官方 DrQ-v2 参考实现采用 LayerNorm。二者不同，复现时需说明使用哪一种。
- 图中的 Critic 示意与官方双 Q 实现的对应关系需核实；不要因为图只画一条输出就任意删除双 Q。
- 原文描述图像增强为 ±4 像素随机平移与零填充；官方代码先使用 replicate padding，再进行 grid_sample。严格对齐时需记录这一差异。
- 若为起步自行选择“速度 3 维 + 姿态若干维”的拼接，应标明是补充设计；不能说已经完整还原 Fig. 3 的 21 维输入。

### 8.3 论文训练参数

| 参数 | Table I 数值 |
|---|---|
| frame stack | 3 |
| image resolution | 84×84 |
| learning rate | 0.005 |
| target soft update τ | 0.0012 |
| batch size | 1024 |
| replay size | 300000 |
| discount | 0.99 |
| warm-up steps | 10000 |
| episode number | 4000 |
| random seeds | 1、2、3、4、5 |
| aggregated steps | 10 |
| automatic entropy tuning | yes，存在方法对应疑点 |

Table I 将 automatic entropy tuning 列在 DrQ-v2 参数中，但官方 DrQ-v2 是带探索噪声的确定性 Actor-Critic，没有 SAC 式温度自动调节。该条可能与 Vision SAC 或作者改动有关，需核实；不要为了照表增加一个并不属于标准 DrQ-v2 的温度模块。

学习率、τ 等数值应按表先记录，不要自动替换为官方仓库默认值。n-step、探索噪声衰减、更新频率等未在本文完整列出；可先从固定上游实现出发，但必须在配置中标记为补充选择。

### 8.4 CPU 短流程检查与正式训练

CPU 阶段可以先用一个种子、小批量（例如 16 或 32）、较小 replay（例如 5000），完成“采样 -> 存储 -> 取 batch -> 前向 -> 反向 -> 保存 -> 加载”的短流程检查。这些数值是**调试建议**，不属于论文配置，也不用于宣称复现论文成功率。

确认以下项目后再扩大训练：

- Actor 输入仅包含允许的观测，没有平台真值泄漏。
- 输出速度范围和坐标系正确。
- 图像堆叠、非图像状态、动作和 reward 的时间索引对应。
- 损失有限、梯度可更新、目标网络有软更新。
- 模型重载后，同一评估观测得到相同动作。
- 超时和真实终止的 bootstrap 处理正确。

正式训练按 5 个独立随机种子分别执行，不共享训练好的权重；评估使用独立环境和独立随机数流。

## 9. 内存与训练成本怎样控制

按 300000 条记录、3 帧灰度、84×84、uint8 计算，光一份完整堆叠观测就约 **6.35 GB**。同时独立保存 obs 和 next_obs 约 **12.70 GB**；若全部用 float32，约 **50.80 GB**，还未包括模型、临时 batch、优化器和系统占用。

因此约 16 GB 内存的电脑不适合朴素地把两份 float32 图像 replay 全放进内存。

建议：

- 以 uint8 保存图像。
- 相邻记录复用原始单帧，取 batch 时再堆叠，正确处理回合边界。
- 必要时使用磁盘 episode、memmap、缓存上限，并监控真实常驻内存。
- 初期关闭 GUI 和视频录制做采样，只在演示和评估回放时开启。
- CPU 数据加载先使用 0 个 worker 排查问题，再按实测调整。
- 不同时启动 5 个训练种子；可以顺序跑。
- 每次 checkpoint 连同配置、代码提交号和实际训练步数保存。

论文 Fig. 5 显示无域随机化训练约百万步量级，Fig. 7 显示域随机化约五百万步量级。Table I 的 4000 episodes 与这些曲线步数的统计口径也需核实。原文未报告可直接套用的 CPU/GPU 小时数，应先实测每秒采样和更新次数，再估算预算。

## 10. 第五步：按实验矩阵做比较与消融

### 10.1 优先完成的最小研究实验

| 实验 | 保留的部分 | 改动 | 回答的问题 |
|---|---|---|---|
| A：Vision DrQ-v2，无随机化 | 完整视觉方法与奖励 | 固定地面和平台纹理 | 方法是否能学会基础任务 |
| B：No LOS | 与 A 一致 | 去掉“平台可见才给其他奖励”的门控 | 可见性奖励是否帮助学习 |
| C：固定下降速度 | 与 A 尽量一致 | 垂直速度由规则固定，策略只控制水平 | 学习垂直速度是否有用 |
| D：域随机化 | 完整视觉方法与奖励 | 加入第 11 节的随机化 | 能否应对背景、光照和时延变化 |

No LOS 不是“没有相机”，它只移除奖励中的可见性门控。固定下降速度的具体常数原文未明确给出，应记录自己的选择；改为二维动作时也要记录网络输出变化。

CPU 上可以先实现并检查这些开关，但要比较训练结果，每个版本都必须独立训练。不能只把已经训练好的 A 政策在评估时删掉奖励，就声称完成了 No LOS 训练消融。

### 10.2 完整对比需要增加的算法

- Vision SAC：保持视觉输入、环境与奖励，改用 SAC。
- FF-IBVS：参考文献 [10] 的前馈图像视觉伺服。
- DDPGLN：参考文献 [13] 的强化学习降落方法。
- FF-IBVS、DDPGLN 分别评估无外加估计噪声、外加最高 ±10% 状态估计误差的设置。

后两种方法需要继续阅读对应论文并实现其控制/网络配置；普通 PID 可以用于环境调试，但不能直接更名为 FF-IBVS 或 DDPGLN。噪声的具体分布和零值附近处理未完整公开，实施时需明确；这些噪声不是直接加在 Vision DrQ-v2 的相对位置输入上，因为它没有该输入。

### 10.3 评估流程与指标

论文给出的评估安排：

1. 训练结果跨 5 个随机种子汇总。
2. 每训练 100 episodes，评估 10 episodes。
3. 每个方法、每个种子选取最高评分网络。
4. 对比使用 500 个评估 episodes；正文未完全说明这是每种子 500 次还是合计 500 次。实施建议每种子 500 次，并注明这是采用的口径。

为了防止模型选择污染最终结果，建议用训练中验证集选择 checkpoint，再用独立固定测试列表最终评估。评估关闭探索噪声，各方法尽量使用相同的初始条件、平台速度与测试纹理。

至少记录：

- 成功率：成功回合数 / 所有评估回合数。
- 失败率、超时率以及各类失败原因。
- 着陆中心误差，单位和计算方式自行明确定义。
- 接触时水平相对速度、垂直速度、姿态角，作为补充安全指标。
- 着陆时间、平台可见时间占比、推理耗时。
- 每个种子的结果，以及跨种子均值和标准差。

仅成功回合的误差与所有终止回合的误差要分别标注，不能把失败回合删除后误导性地展示结果。

论文 Table II 的原始结果抄录如下，仅供对照：

| 方法/变体 | 成功率 | Landing error（表中数值） |
|---|---:|---:|
| Vision DrQ-v2，无随机化 | 97.4% | 0.52 |
| Vision DrQ-v2，域随机化 | 91.8% | 0.63 |
| Vision DrQ-v2，No LOS | 82.4% | 1.78 |
| Vision DrQ-v2，固定下降速度 | 75.2% | 2.02 |
| FF-IBVS，有外加噪声 | 65.7% | 0.61 |
| FF-IBVS，无外加噪声 | 87.8% | 0.58 |
| DDPGLN，有外加噪声 | 31.6% | 1.64 |
| DDPGLN，无外加噪声 | 75.4% | 0.64 |
| Vision SAC | 60.1% | 0.75 |

**表中的 Landing error 未在表头标明单位，统计定义也不够完整，因此不能自行把 0.52 写成“0.52 m 的成功着陆中心误差”。**报告自己的结果时必须给出单位、公式和样本范围。以上均为原论文结果，不是本仓库或本次任务取得的结果。

## 11. 第六步：逐项加入域随机化

按论文 III-D 节：

| 随机化 | 论文设置 | 实施提醒 |
|---|---|---|
| 地面纹理 | 约 400 张 DTD 纹理，每回合随机选择 | 保存纹理清单与划分，测试纹理独立 |
| 亮度 | 图像乘以 0.5-1.5 范围的随机系数 | 具体分布未完整给出；处理后裁剪到有效范围 |
| 初始位置 | 10 m 高度附近，最多 ±3 m 偏移 | 明确各轴采样与高度基准 |
| 平台速度 | 每回合从 0-1 m/s 范围采样 | 基础任务也是移动平台；“无随机化”对哪些项关闭需说明 |
| 初始航向 | -15° 到 15° | 角度转弧度，确定世界坐标和相机坐标关系 |
| 地面干扰物 | 随机形状和大小的 3D 物体 | 数量、尺寸、位置分布未公开，须自行记录 |
| 平台纹理 | 从 2 m 高度拍摄的真实平台图片，裁剪后贴图 | 原始图片未随 PDF 给出；自己绘制 H 是替代素材 |
| 观测/决策间隔 | 0.1-0.4 s | 明确采样粒度、动作保持方式和超时计时 |

DTD：[官方数据集](https://www.robots.ox.ac.uk/~vgg/data/dtd/)。约 400 张是论文选用数量，不是整个数据集大小。

建议先做基础版本，再逐项增加纹理、亮度、航向、干扰物和时序扰动。阶段调试可以逐项开关，最终应与论文的完整组合对照。若替换了真实平台照片，只能报告“使用自建平台素材的复现版本”，不能认为已严格复现 sim-to-real 视觉差异。

域随机化后的结果应分别在已见背景、未见背景上评估。只换测试背景、不重新训练，测试的是已有策略的泛化能力，不等同于域随机化训练实验。

## 12. 需要补写哪些代码

下面是建议的文件职责，**不是当前已经提供的可运行文件列表**：

| 文件/模块 | 需要实现的内容 |
|---|---|
| `envs/moving_landing_env.py` | 移动平台、物理步进、接触与超时 |
| `envs/downward_camera.py` | 机体相机、灰度缩放、3 帧堆叠 |
| `envs/velocity_adapter.py` | 三维目标速度到 PID/RPM |
| `envs/rewards.py` | 式 (7)-(10)、奖励分量和公式模式 |
| `envs/randomization.py` | 纹理、亮度、初始条件、时序扰动 |
| `agents/drqv2_landing.py` | 图像与自身状态编码、Actor、双 Q、更新 |
| `replay.py` | uint8 图像回放、历史重建、终止处理 |
| `train.py` / `evaluate.py` | 训练、验证、checkpoint 选择、独立测试 |
| `configs/` | 原文参数、补充假设、CPU 调试配置 |
| `results/` | CSV、曲线、截图、视频和配置副本 |

实现顺序：先环境和底层控制，再相机与奖励，然后算法接口，最后训练与实验对照。每一步保留可检查的日志与截图，避免一直训练到最后才发现动作方向或相机坐标错了。

## 13. 复现前必须补齐的未公开信息

| 问题 | 当前证据 | 应采取的处理 |
|---|---|---|
| 作者完整任务代码、权重、依赖版本 | PDF 未提供明确代码链接 | 继续查作者主页/补充材料，或自行实现并声明 |
| 相机内参、安装参数、裁剪范围 | 未完整给出 | 固定配置，验证目标可见性，报告参数 |
| Fig. 3 多出的 21 维如何组成 | 仅能确认图中 39221 | 不把某种猜测作为原始配置 |
| BatchNorm 与官方 LayerNorm 差别 | 论文图、参考源码不同 | 显式选择并记录，必要时比较 |
| 式 (9) 指数括号 | 原文疑似存在排版问题 | 保留原式与解释版本，寻求作者确认 |
| 式 (8) 实际/指令速度混用 | 原文使用两个不同变量 | 按变量区分实现并记录 |
| 自动熵调节归属 | 与标准 DrQ-v2 不一致 | 不直接照搬 SAC 模块 |
| 物理/控制/策略时序 | 30 s、900 steps、aggregate=10 与随机间隔并列 | 单独维护时钟，报告自己的时序 |
| Landing error 的单位、样本和公式 | Table II 未完整说明 | 自己定义并与原表分开比较 |
| 500 次评估的种子统计口径 | 正文有歧义 | 报告实际采用口径 |
| 原始纹理子集与平台实拍图 | 未给完整文件 | 保存替代素材来源及划分 |
| 固定下降速度、基线重调参和噪声细节 | 未完整公开 | 阅读对应参考文献，标明补充设置 |
| 作者训练硬件和耗时 | 未报告 | 实测预算，不编造时长 |

这些缺项意味着：即使训练成功，也应先称为“依据论文描述实现的复现版本”，直到关键差异得到确认后，再讨论严格数值对齐。

## 14. 建议里程碑与汇报方式

| 里程碑 | 完成条件 | 可如实汇报的进展 |
|---|---|---|
| M1：仿真底座 | 第 5 节示例可运行、包版本已保存 | 已搭建 PyBullet/Crazyflie 仿真环境 |
| M2：本文场景 | 平台、机载图像、速度控制、接触事件可检查 | 已实现移动平台降落任务环境 |
| M3：奖励和数据流 | 奖励检查通过、DrQ-v2 更新和存取模型可用 | 已接通视觉强化学习训练流程 |
| M4：基础方法 | 独立种子训练、独立测试得到结果 | 已完成基础场景实验，并报告配置差异 |
| M5：消融与泛化 | 完成 No LOS、固定下降速度、域随机化对照 | 已研究奖励模块和随机化的作用 |
| M6：完整算法比较 | 实现并评估 FF-IBVS、DDPGLN、Vision SAC | 已完成论文主要仿真对比的复现版本 |

截至 2026-10-10，已完成 M1 仿真底座、M2 移动平台环境及 M3 的 CPU 数据流验收，执行了带明确补充配置的 DrQ-v2 短训练与独立评估。尚未完成 M4 的论文规模训练、M5/M6 正式实验。下节保留初始安装记录，新的实现差异和实际结果见 CPU 运行说明与实验报告。

### 14.1 本机安装与运行记录（2026-10-10）

目标文献仍为 Ladosz 等人的 **Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**，DOI：10.1109/LRA.2024.3379837。

| 项目 | 实际配置或结果 |
|---|---|
| 系统 | Windows / WSL2 / Ubuntu 20.04，x86_64，WSLg |
| 独立环境 | Miniforge，环境名 landing-drqv2；未修改系统 Python 3.8 |
| Python | 3.10.22 |
| PyTorch | 2.2.2+cpu，CUDA 不可用，CPU 卷积前向和反向检查通过 |
| 仿真 | PyBullet 3.2.6；gym-pybullet-drones 2.0.0，提交 50d1a958def27121bcf63f3cc753049fee8e71e5 |
| 其他主要版本 | NumPy 1.24.4、SciPy 1.10.1、Gymnasium 0.28.1、Stable-Baselines3 2.0.0 |
| 依赖完整性 | pip check 通过 |
| PID 悬停验证 | 物理频率 240 Hz，控制频率 48 Hz，运行 8 秒；最终到目标点的三维位置误差约 0.00476 m |
| 相机验证 | CPU TinyRenderer 输出 84×84 非均匀图像；这是渲染接口检查，没有验证论文机载相机的外参配置 |
| 桌面演示 | Crazyflie / DSLPIDControl，半径 0.35 m、目标高度 0.7 m、周期 12 s 的圆形轨迹；实际三维窗口已启动 |
| 图形兼容 | Intel D3D12 渲染出现黑画面；设置 LIBGL_ALWAYS_SOFTWARE=1 后，Mesa llvmpipe 软件渲染正常 |

**这些数字属于安装验收和 PID 示例，不是本文的降落精度、降落成功率或 DrQ-v2 训练结果。** 圆形轨迹演示没有移动降落平台，也没有学习策略。

本机启动方式：双击 Windows 桌面的 `启动无人机仿真.cmd`。脚本调用已安装的 WSL 环境，打开 PyBullet 三维窗口，演示最多运行 30 分钟；关闭窗口即可停止。也可以在 Ubuntu 终端执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
bash /mnt/d/perchrl_repro/vision_drqv2_setup/launch_demo.sh
```

本机文件位置（以下为本次实际安装路径）：

- 安装、验证和演示脚本：`D:\perchrl_repro\vision_drqv2_setup`。
- 依赖快照：该目录中的 `requirements-installed.txt`；其中 editable 安装路径指向本机 Linux 用户目录，不可原样当作跨机器安装清单。
- 验收报告：该目录中的 `environment_check.json` 和 `hover_positions.csv`。
- 实际仿真窗口截图：该目录中的 `simulation_window.png`。
- Linux 源码：`~/uav_landing_repro/gym-pybullet-drones`。
- 运行日志：`~/uav_landing_repro/demo_results/demo.log`；演示轨迹记录为同目录下的 `desktop_flight.csv`。

以上为最初安装验收记录。后续已添加水平移动平台、下视相机、速度动作接口和接触判定，并接入 DrQ-v2。参见 [后续可运行实现](running_cpu_landing.md)；不再使用上游默认 MuJoCo benchmark 代替本任务。

组会优先展示真实的仿真窗口、机载相机画面和任务判定日志。截图必须注明是上游示例、自己实现的控制器或学习策略，不能把 PID 演示称为 DrQ-v2 成果。

## 15. 来源与引用

1. Ladosz, P., Mammadov, M., Shin, H., Shin, W., & Oh, H. Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning. IEEE Robotics and Automation Letters, 9(5), 4575-4582, 2024. [DOI](https://doi.org/10.1109/LRA.2024.3379837)。本指南参数、公式和结果来自用户提供的原文，主要核对 II、III、IV、V-A 节、Fig. 3、Fig. 5-7、Table I-II。
2. gym-pybullet-drones：[官方仓库](https://github.com/learnsyslab/gym-pybullet-drones)，原 utiasDSL 地址已重定向；[本指南固定提交](https://github.com/learnsyslab/gym-pybullet-drones/tree/50d1a958def27121bcf63f3cc753049fee8e71e5)，[PID 示例](https://github.com/learnsyslab/gym-pybullet-drones/blob/50d1a958def27121bcf63f3cc753049fee8e71e5/gym_pybullet_drones/examples/pid.py)。
3. DrQ-v2：[官方算法代码](https://github.com/facebookresearch/drqv2/tree/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef)，[网络与更新实现](https://github.com/facebookresearch/drqv2/blob/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/drqv2.py)，[默认配置](https://github.com/facebookresearch/drqv2/blob/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/cfgs/config.yaml)。
4. PyBullet：[官方维护者发布页](https://pypi.org/project/pybullet/3.2.6/)，含 CPU 渲染支持与 Linux Python 3.10 发行文件。
5. PyTorch：[官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。
6. DTD：[牛津大学 VGG 官方数据集](https://www.robots.ox.ac.uk/~vgg/data/dtd/)。
7. 论文作者提供的实验视频：[YouTube](https://youtu.be/BXyjc6C6TgM)。视频链接来自论文；本次未以视频替代参数或代码核验。

本仓库仅发布自行撰写的复现指南；论文 PDF、论文页面截图和第三方训练素材没有随文档上传。

```bibtex
@article{ladosz2024autonomous,
  title   = {Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning},
  author  = {Ladosz, Pawel and Mammadov, Meraj and Shin, Heejung and Shin, Woojae and Oh, Hyondong},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2024},
  volume  = {9},
  number  = {5},
  pages   = {4575--4582},
  doi     = {10.1109/LRA.2024.3379837}
}
```
