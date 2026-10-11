# 训练在运行，但桌面窗口没有出现

2026-10-11 排查：训练进程仍在更新，独立 PyBullet 窗口的 Linux 窗口坐标为约 (-32768, -32768)，超出了桌面范围。普通的置顶请求没有恢复它。重新映射这个 Linux 窗口后，坐标恢复为桌面范围内的 (169, 118)，训练继续运行。

双击桌面的 **显示当前无人机训练窗口.cmd**。这个入口读取当前运行目录，核对训练进程，再请求训练监控恢复自身，并恢复独立三维窗口。不会创建另一个训练进程。

也可以在 PowerShell 执行：

```powershell
wsl -d Ubuntu-20.04 --cd /mnt/d/paper_reproductions/posture_control --exec /home/ubuntu/miniforge3/envs/landing-drqv2/bin/python -m landing_repro.show_training_window
```

如果训练进程已经停止，使用 **继续变速平台训练.cmd**。正在训练时不要关闭启动终端，也不要用关闭 WSL 的方式恢复窗口。

这次日志中的 `llvmpipe` 表示 CPU 软件渲染；`Box bound precision lowered` 是数组类型转换提示。它们不是这次窗口消失的退出错误。`live_status.json` 中步数持续增加，才是训练仍在运行的直接依据。窗口坐标恢复已验证；Windows 前台实际显示需要在桌面确认。
