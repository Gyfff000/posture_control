# WSLg窗口仅有任务栏图标的修复记录

日期：2026-10-12。

## 已确认的原因

Windows端训练窗口标题出现`[WARN:COPY MODE]`。WSLg日志显示：

```text
rdp_allocate_shared_memory: Failed to open "/mnt/shared_memory/{...}" with error: Input/output error
RDP backend: use_gfxredir = 0
```

`/mnt/shared_memory`实际已经挂载为virtiofs。该故障与Microsoft WSLg项目记录的不可见窗口问题一致：<https://github.com/microsoft/wslg/issues/1456>。因此没有叠加一个tmpfs挂载，也没有修改Mesa、GPU驱动或训练算法。

## 本次处理与证据

1. 训练在205000步完成检查点保存，并使用PyTorch重新加载验证。
2. 通过短暂的独立Ubuntu会话保持发行版存活，避免修复过程中共享内存会话再次销毁。
3. 以WSLg系统发行版的root身份，验证Weston进程实际路径后只向该图形服务发送TERM；由WSLGd重新启动它。
4. 等待日志明确显示`use_gfxredir = 1`，才继续启动训练。
5. Windows原生进程查询确认新窗口标题不再有`COPY MODE`；模型从205000步继续，平台曲线难度和2米高度保持继承。

仅重新唤起X11窗口或关闭训练后直接启动新WSLg会话，不能保证修复上述共享内存故障。原来Linux端截图正常，也不能证明Windows端实际可见。

## 桌面修复入口

桌面`修复训练窗口.cmd`调用仓库`scripts/repair_training_display.ps1`。

- 正常显示模式：重新显示现有训练面板和Bullet窗口。
- COPY MODE：先检查是否有其他Linux图形窗口；有则停止处理并报告。否则保存并验证当前模型、重启Weston、确认正常传输，再从检查点续训。
- 检查点保存失败或图形传输仍异常时，不继续启动另一份训练。
- 不执行`wsl --shutdown`，不重启Windows，不强行结束其他WSL工作。
- 当前深度恢复入口适用于本仓库的高度课程训练。

只读检查：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\paper_reproductions\posture_control\scripts\repair_training_display.ps1 -CheckOnly
```

恢复会加载模型和优化器，但经验池、未完成回合及随机数状态重新初始化；不属于完整训练现场的精确恢复。
