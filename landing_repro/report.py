"""Build a report from the actual local experiment artifacts."""
import json
from pathlib import Path
import shutil


def main():
    root = Path("runs")
    dst = Path("docs/results")
    dst.mkdir(parents=True, exist_ok=True)
    inputs = {
        "initial_failure": "cpu_short_seed1/failure.json",
        "initial_config": "cpu_short_seed1/config.json",
        "first_finite_summary": "cpu_short_seed1_safe/summary.json",
        "first_finite_sensitivity": "cpu_short_seed1_safe/image_sensitivity.json",
        "first_finite_eval": "eval_drq_paper_lr/summary.json",
        "pilot_summary": "cpu_pilot_lr1e4/summary.json",
        "pilot_config": "cpu_pilot_lr1e4/config.json",
        "pilot_eval": "eval_drq_pilot/summary.json",
        "pilot_sensitivity": "cpu_pilot_lr1e4/image_sensitivity.json",
        "oracle_eval": "eval_oracle/summary.json",
        "environment_checks": "validation/checks.json",
    }
    records = {}
    for name, relative in inputs.items():
        source = root / relative
        records[name] = json.loads(source.read_text())
        shutil.copyfile(source, dst / f"{name}.json")
    for name, relative in [("first_finite_episodes", "cpu_short_seed1_safe/episodes.csv"),
                           ("pilot_episodes", "cpu_pilot_lr1e4/episodes.csv"),
                           ("pilot_evaluation_episodes", "eval_drq_pilot/episodes.csv")]:
        shutil.copyfile(root / relative, dst / f"{name}.csv")
    a, b = records["first_finite_summary"], records["pilot_summary"]
    ea, eb, oracle = records["first_finite_eval"], records["pilot_eval"], records["oracle_eval"]
    sa, sb = records["first_finite_sensitivity"], records["pilot_sensitivity"]
    metrics = [json.loads(line) for line in (root / "cpu_pilot_lr1e4/metrics.jsonl").read_text().splitlines()]
    nonzero = sum(m["encoder_grad_norm"] > 0 for m in metrics)
    report = f"""# CPU 移动平台降落与 DrQ-v2 实验记录

日期：2026-10-10。
目标文献：**Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning**，
Pawel Ladosz 等，IEEE RA-L 2024，DOI：[10.1109/LRA.2024.3379837](https://doi.org/10.1109/LRA.2024.3379837)。

## 当前结论

移动平台场景、机载相机、速度控制、奖励与碰撞接口，以及 DrQ-v2 的真实采样、梯度更新、
保存、重载和独立评估均已运行。**当前 CPU 调试训练未证明学会可靠降落，也未完成论文的完整复现。**
实现补充和命令见 [运行说明](running_cpu_landing.md)。

## 实际执行结果

| 方法/用途 | 训练步数 | 参数更新 | 独立评估成功数 | 说明 |
|---|---:|---:|---:|---|
| 平台真值 + PID，环境诊断 | 无学习 | 无 | {oracle['successes']}/{oracle['episodes']} | 不属于 DrQ-v2，也不是论文视觉方法 |
| DrQ-v2，学习率 0.005 | {a['steps']} | {a['updates']} | {ea['successes']}/{ea['episodes']} | 策略饱和，不能据此声称学到视觉控制 |
| DrQ-v2 CPU 调试，学习率 0.0001 | {b['steps']} | {b['updates']} | {eb['successes']}/{eb['episodes']} | 小预算实验，与论文训练配置有差异 |

所有独立评估使用种子 10001–10010；初始高度为平台上方 10±3 m，水平偏移每轴 ±3 m，
平台速度 0–1 m/s，回合限时 30 秒。
末轮事件分布：`{json.dumps(eb['events'], ensure_ascii=False)}`。
真值调试控制器的事件分布：`{json.dumps(oracle['events'], ensure_ascii=False)}`。
这 10 个回合不等于论文的 5 种子、每种子 500 回合统计。

## 训练与模型检查

- 首轮有限训练：实际 {a['updates']} 次更新，参数最大变化 {a['parameter_max_change']:.6g}；模型重载前后动作最大差 {a['checkpoint_action_max_error']:.6g}。
- 后续调试：实际 {b['updates']} 次更新，参数最大变化 {b['parameter_max_change']:.6g}；模型重载前后动作最大差 {b['checkpoint_action_max_error']:.6g}。
- 后续调试的编码器非零梯度更新数：{nonzero}/{len(metrics)}。
- 固定自身状态、更换真实图像，首轮饱和策略的最大动作变化为 {sa['actor_max_action_image_difference']:.6g}；后续策略为 {sb['actor_max_action_image_difference']:.6g}。
- 图像敏感性只能检查数据流和策略响应，不能单独证明着陆能力。
- 后续策略的三个测试动作仍接近 `[1, 1, -1]`；约百万分之几的差异不足以说明它已学会有效视觉反馈。
- 后续短训练仅完成 {b['episodes_completed']} 个完整回合，预算与论文 4,000 回合相差很大；当前无法区分更多训练与实现差异各自的影响。
- 后续调试最终 Critic 损失：{b['final_metrics']['critic_loss']:.6g}；Actor 损失：{b['final_metrics']['actor_loss']:.6g}。

## 保留的失败记录

最初按指令速度门控下降奖励，在失稳下降时出现极大正奖励，导致 Critic 损失非有限。
保留记录见 [initial_failure.json](results/initial_failure.json)。随后加入实际下降速度检查和速度指令加速度限制。
再次使用学习率 0.005 时训练可运行，但策略在不同图像下均输出饱和动作。
后续学习率改为 0.0001；这些修正是明确补充，不能写成作者原始实现。

## 可展示的界面

桌面已启动三维移动平台场景和实时机载相机，窗口中写明控制器类型。
当前调试演示从 2 m 开始，平台速度为 0.4 m/s，方便观察；不用于上述评估统计。
三维视口导出文件为 `runs/desktop_oracle/rendered_viewport.png`，机载相机为 `onboard_camera.png`。
视口导出不等于整张桌面截图。WSLg 下可选的 X11 窗口抓图可能失败，仿真和相机不受影响。
训练模型可通过桌面“查看DrQ短训练策略.cmd”加载并展示。

## 后续工作

需要先获得稳定的独立评估降落，再研究更大训练预算和探索设置，并核实论文未明确的奖励及网络配置。
尚未实施 DTD 域随机化、无可见性奖励的正式消融、FF-IBVS/DDPGLN/Vision SAC 对比或实机验证。
不把训练回报上涨、损失有限、PID 降落演示当作论文成功率的复现证据。

所有本报告数字均来自 [原始小型结果文件](results/)；完整日志和模型保留在本机 runs 目录。
"""
    Path("docs/cpu_experiment_report.md").write_text(report, encoding="utf-8")
    print("Report written from actual artifacts")


if __name__ == "__main__":
    main()
