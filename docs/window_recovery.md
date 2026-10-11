# 训练在运行，但桌面窗口没有出现

## 2026-10-11：原生显示连接恢复记录

本机 WSL 2.7.12.0、WSLg 1.0.73.2。故障时 Windows 任务栏有原生窗口，但桌面没有画面；窗口标题带 `[WARN:COPY MODE]`。Linux 窗口坐标正常仍然不能证明 Windows 实际显示正常。

关键日志在 `/mnt/wslg/weston.log`：

```text
rdp_allocate_shared_memory: Failed to open "/mnt/shared_memory/{...}" with error: Input/output error
RDP backend: use_gfxredir = 0
```

这与 [Microsoft WSLg #1456](https://github.com/microsoft/wslg/issues/1456) 和 [OpenVMM #4274](https://github.com/microsoft/openvmm/issues/4274) 描述的共享内存显示故障一致。相关目录属于 WSLg 系统发行版；不能根据 Ubuntu 中没有 `/mnt/shared_memory` 就直接判断缺少挂载，也不能把修改模型当作显示修复。

本次操作：

1. 先向已核对 PID 和命令行的训练进程发送 SIGINT，由训练程序保存 `interrupted.pt`。实际校验过文件可加载，累计 **65000 步**，运动阶段仍为 1。
2. 在训练已经停止后，通过 WSLg 系统发行版重启 Weston 图形组件。会关闭现有 Linux 图形窗口，应先保存其他图形应用的内容。
3. 等待显示客户端重新连接。初始日志仍可能是 `use_gfxredir = 0`，本次约 15 秒后出现更新的 **`use_gfxredir = 1`**；判断时应看最后一条状态，不能只看日志开头。
4. 启动原生 Tk 测试窗口，Windows 进程中的窗口标题已不带 `COPY MODE` 警告。
5. 关闭临时镜像观看程序，从65000步模型续训，恢复原始 PyBullet 三维窗口和 Tk 训练监控窗口。原生窗口保留原有按钮和鼠标操作，不是实时镜像。

完整关闭 WSL 的操作曾被自动审批拦截，理由是可能终止其他未保存的 Linux 工作；**本次未执行 `wsl --shutdown`**，使用了更窄的图形组件重启方案。

恢复后的运行目录为 `runs/motion_curriculum_20261011_195010`，原模型保留在 `runs/motion_curriculum_20261011_194156/interrupted.pt`。使用已有续训入口重新启动会重置经验缓存和中断的飞行回合，并按入口默认值重新分配后续80000步预算；当前总步数上限145000。网络、优化器和累计步数继承保存值。

本次确认了显示链路启用、Windows 原生窗口标题和训练步数继续增长；实际桌面画面与点击操作需要由用户确认。这里记录的是当前会话的恢复，不能视为已修改或修复 WSL 自身的底层缺陷。

## Windows 桌面仍不显示 Linux 窗口时

### 两个独立的桌面窗口

双击桌面的 **打开无人机训练双窗口.cmd**：

- **无人机训练监控**：三个视角、当前步数、训练日志。
- **无人机三维仿真**：独立 PyBullet 场景，包含无人机、平台、飞行轨迹与中心估计线。

这两个 Windows 原生观看窗口分别接收当前 Linux 窗口的实时画面。没有启动第二个训练或模型演示，也没有重新加载权重。窗口并排放置，可以移动、缩放和关闭。关闭两个观看窗口不会停止训练。

它们是故障期间的备用实时镜像，不能点击图中的训练按钮，也不能直接拖动三维相机。上面的恢复记录说明了后续如何保存模型并恢复原生显示；正常使用原生窗口时不需要开启这些镜像窗口。

三维画面服务只绑定本机 127.0.0.1:8766；训练监控使用 127.0.0.1:8765。Windows 观看程序的 `runs/desktop_dual_viewer_status.json` 记录两路画面实际绘制的帧数和时间，供检查更新是否正常。

### 单个浏览器窗口

恢复 Linux 窗口坐标并不保证 Windows 能实际显示它。增加了 **查看实时无人机训练.cmd** 桌面入口：在独立 Edge 窗口打开本机实时画面，同时显示步数、训练状态和画面延迟。网址为 http://localhost:8765 。

画面直接采集正在运行的 Linux 训练窗口，约每秒 4 帧，包含无人机近景、平台近景、机载相机和实时日志。不是录像或另起一个策略演示。图中的按钮是画面的一部分，不能点击；关闭浏览器窗口不停止训练。每 5000 步的验证期间训练步数暂停，画面仍显示测试飞行。

首次使用入口会启动独立的本机画面服务（仅绑定 127.0.0.1），不会启动或重启训练。服务日志在当前运行目录的 `desktop_monitor.stdout.log` 和 `desktop_monitor.stderr.log` 中。入口适用于当前电脑的 WSL 与 Python 路径。

## Linux 窗口恢复记录

2026-10-11 排查：训练进程仍在更新，独立 PyBullet 窗口的 Linux 窗口坐标为约 (-32768, -32768)，超出了桌面范围。普通的置顶请求没有恢复它。重新映射这个 Linux 窗口后，坐标恢复为桌面范围内的 (169, 118)，训练继续运行。

双击桌面的 **显示当前无人机训练窗口.cmd**。这个入口读取当前运行目录，核对训练进程，再请求训练监控恢复自身，并恢复独立三维窗口。不会创建另一个训练进程。

也可以在 PowerShell 执行：

```powershell
wsl -d Ubuntu-20.04 --cd /mnt/d/paper_reproductions/posture_control --exec /home/ubuntu/miniforge3/envs/landing-drqv2/bin/python -m landing_repro.show_training_window
```

如果训练进程已经停止，使用 **继续变速平台训练.cmd**。正在训练时不要关闭启动终端，也不要用关闭 WSL 的方式恢复窗口。

这次日志中的 `llvmpipe` 表示 CPU 软件渲染；`Box bound precision lowered` 是数组类型转换提示。它们不是这次窗口消失的退出错误。`live_status.json` 中步数持续增加，才是训练仍在运行的直接依据。窗口坐标恢复已验证；Windows 前台实际显示需要在桌面确认。
