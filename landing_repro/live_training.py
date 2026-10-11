"""Desktop monitor for the SAME environment and agent used by train.py."""
import json
import os
import time
import tkinter as tk
from tkinter import scrolledtext

from PIL import ImageTk

from .scene_views import inspection_views, save_inspection
from .center_marker import update_center_marker


class LiveTraining:
    def __init__(self, output):
        self.output = output
        self.stop = False
        self.paused = False
        self.last_draw = 0.
        self.last_status = 0.
        self.mode = "training"
        self.replay_requested = False
        self.focus = "drone"
        self.snapshot_saved = False
        self.root = tk.Tk()
        self.root.title("LIVE DrQ-v2 TRAINING | actual model + exploration")
        self.root.geometry("1040x760+40+30")
        self.root.protocol("WM_DELETE_WINDOW", self.request_stop)
        self.heading = tk.Label(self.root, text="LIVE DrQ-v2 CPU TRAINING", font=("Arial", 15, "bold"))
        self.heading.pack()
        self.saved_result = tk.Label(self.root, text="Actual physics. External close views are for inspection; policy uses onboard images and own state.", wraplength=1000)
        self.saved_result.pack()
        views = tk.Frame(self.root)
        views.pack(pady=5)
        self.view_labels = []
        for title in ("DRONE: real model, tracking close view", "PLATFORM: real model, tracking close view", "ONBOARD: camera + display-only marker"):
            group = tk.LabelFrame(views, text=title)
            group.pack(side="left", padx=3)
            label = tk.Label(group)
            label.pack()
            self.view_labels.append(label)
        tk.Label(self.root,text='GREEN = camera estimate | ORANGE = prediction (normal max 2s; armed near-field max 8s) | C = confidence | UNKNOWN = no estimate. Auxiliary estimator.',wraplength=1000).pack()
        self.stats = tk.Label(self.root, text="Starting...", justify="left", anchor="w", font=("Courier", 11))
        self.stats.pack(fill="x", padx=10)
        controls = tk.Frame(self.root)
        controls.pack(pady=6)
        self.pause_button = tk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=8)
        self.stop_button = tk.Button(controls, text="Stop and save model", command=self.request_stop)
        self.stop_button.pack(side="left")
        self.replay_button = tk.Button(controls, text="Test saved model (no training)", command=self.request_replay, state="disabled")
        self.replay_button.pack(side="left", padx=8)
        for focus in ("drone", "platform", "overview"):
            tk.Button(controls, text="3D: " + focus, command=lambda f=focus: setattr(self, "focus", f)).pack(side="left", padx=2)
        tk.Label(self.root, text="Episode results (success / ground / edge / timeout)").pack()
        self.log = scrolledtext.ScrolledText(self.root, height=6, width=100, state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=5)
        tk.Label(self.root, text="Model and logs: " + str(output), wraplength=1000).pack()
        self.pump()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text="Continue" if self.paused else "Pause")

    def request_stop(self):
        self.stop = True

    def request_replay(self):
        self.replay_requested = True

    def show_completed(self, summary):
        self.mode = "finished"
        self.heading.configure(text="TRAINING FINISHED - model and results saved")
        self.saved_result.configure(text=(f"Saved training: {summary['steps']} steps | {summary['updates']} updates | "
            f"{summary['episodes_completed']} completed episodes | {summary['training_successes']} successful landings\n"
            "This window stays open. Click 'Test saved model' to run a NEW evaluation, without parameter updates."))
        self.stats.configure(text="Training has ended. The three views show the last physical state.\n"
                             "A model checkpoint exists; successful training is not established by saving it.")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(text="Close windows")
        self.replay_button.configure(state="normal")
        self.pump()

    def begin_review(self):
        self.mode = "review"
        self.paused = False
        self.heading.configure(text="SAVED MODEL TEST - NO TRAINING / NO PARAMETER UPDATES")
        self.pause_button.configure(state="normal", text="Pause")
        self.replay_button.configure(state="disabled")
        self.root.title("Saved DrQ-v2 model test | drone + platform close views")

    def review_finished(self, results=None):
        self.mode = "finished"
        self.heading.configure(text="SAVED MODEL TEST FINISHED - results stay open")
        self.pause_button.configure(state="disabled")
        self.replay_button.configure(state="normal")
        if results is not None:
            successes = sum(int(row['success']) for row in results)
            self.stats.configure(text=(f"Completed test episodes: {len(results)} | Successful landings: {successes}\n"
                "No parameter updates were performed during this test.\n" +
                "\n".join(f"Test {row['episode']}: {row['event'].upper()} | XY error {row['final_xy_error_m']:.2f} m" for row in results)))
        status_path = self.output / "live_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            status.update(state="test_completed" if not self.stop else "test_stopped", mode="review")
            if results is not None:
                status.update(episodes_completed=len(results), successes=sum(int(row['success']) for row in results))
            status_path.write_text(json.dumps(status, indent=2))

    def pump(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.stop = True

    def wait_if_paused(self):
        self.pump()
        while self.paused and not self.stop:
            self.pump()
            time.sleep(.05)

    def episode_result(self, episode, info, episode_return):
        prefix = "TEST" if self.mode == "review" else "TRAIN"
        line = f"{prefix} episode {episode}: {info['event'].upper()} | return={episode_return:.2f}\n"
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")

    def update(self, env, step, total, warmup, updates, episode, successes, replay_size,
               episode_return, action, info, metrics, elapsed, force=False):
        now = time.monotonic()
        phase = "RANDOM WARM-UP (model not controlling)" if step <= warmup else "MODEL + EXPLORATION NOISE"
        phase = getattr(self, "training_label", phase)
        if self.mode == "review":
            phase = "SAVED MODEL TEST (no parameter updates)"
        status = {"pid": os.getpid(), "step": step, "total_steps": total, "phase": phase,
                  "updates": updates, "episodes_completed": episode, "successes": successes,
                  "replay_size": replay_size, "event": info["event"], "time_s": info["time_s"],
                  "relative_position": info["relative_position"], "action": action.tolist(),
                  "wall_seconds": elapsed, "metrics": metrics, "state": "running", "mode": self.mode}
        center=update_center_marker(env)
        status['center_estimate']=center
        status['prediction_reward']=info.get('prediction_reward',0.)
        status['prediction_guidance']=env.cfg.prediction_guidance
        status['near_guard']=info.get('near_guard',{})
        status['near_center_reward']=info.get('near_center_reward',0.)
        status['executed_normalized_action']=info.get('executed_normalized_action',action.tolist())
        if force or now - self.last_status >= 1.:
            (self.output / "live_status.json").write_text(json.dumps(status, indent=2))
            self.last_status = now
        if force or now - self.last_draw >= .12:
            env.debug_camera_focus = self.focus
            env.update_debug_camera()
            views = inspection_views(env)
            for label, view in zip(self.view_labels, views):
                image = ImageTk.PhotoImage(view)
                label.configure(image=image)
                label.image = image
            if not self.snapshot_saved:
                save_inspection(views, self.output / "inspection_views.png", phase)
                self.snapshot_saved = True
            loss = metrics.get("critic_loss")
            loss_text = "pending" if loss is None else f"{loss:.5f}"
            self.stats.configure(text=(f"{phase}\n"
                f"Steps: {step:6d} / {total}   Updates: {updates}\n"
                f"Episode: {episode+1}   Completed: {episode}\n"
                f"{'Test' if self.mode == 'review' else 'Training'} landings: {successes}   Replay: {replay_size}\n"
                f"Sim time: {info['time_s']:5.1f}s   Wall: {elapsed:6.1f}s\n"
                f"Height above deck: {info['relative_position'][2]:6.2f} m\n"
                f"Horizontal distance to platform: {(info['relative_position'][0]**2+info['relative_position'][1]**2)**.5:6.2f} m | Onboard sees pad: {info['visible']}\n"
                f"Normalized action: {action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}\n"
                f"Episode return: {episode_return:8.2f}\n"
                f"Critic loss: {loss_text}\n"
                f"Current result: {info['event'].upper()} | Center: {center['mode'].upper()} | Confidence: {center.get('confidence',0.):.2f}\n"
                f"Since full detection: {center.get('age_seconds') or 0.:.2f}s | Prediction reward: {info.get('prediction_reward',0.):+.4f} | Guidance: {'ON' if env.cfg.prediction_guidance else 'OFF'}\n"
                f"Near assist: {'ACTIVE' if info.get('near_guard',{}).get('active') else 'OFF'} | Center reward: {info.get('near_center_reward',0.):+.4f}\n"
                f"Executed action: {' '.join(f'{v:+.2f}' for v in info.get('executed_normalized_action',action))}"))
            self.last_draw = now
        self.pump()

    def hold_result(self, seconds=.8):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.stop:
            self.pump()
            time.sleep(.03)

    def validation_frame(self, env, group, episode, episodes, info, action,
                         episode_return, finished, training_step, training_total, updates):
        """Show the separate evaluation world without counting it as training."""
        now = time.monotonic()
        if finished or now - self.last_draw >= .25:
            for label, view in zip(self.view_labels, inspection_views(env)):
                image = ImageTk.PhotoImage(view)
                label.configure(image=image)
                label.image = image
            self.heading.configure(text='INDEPENDENT VALIDATION - LIVE TEST FLIGHT / NO LEARNING')
            self.stats.configure(text=(
                f'Test group: {group} | Flight {episode}/{episodes}\n'
                f'Training held at: {training_step}/{training_total} steps | Updates: {updates}\n'
                f'Test simulation time: {info["time_s"]:.1f} s | Result: {info["event"].upper()}\n'
                f'Height above deck: {info["relative_position"][2]:.2f} m | Onboard sees pad: {info["visible"]}\n'
                f'Test return: {episode_return:.2f} | Network action: {action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}\n'
                f'Center: {env._center_tracker.result["mode"]} | C={env._center_tracker.result.get("confidence",0.):.2f} | Prediction reward={info.get("prediction_reward",0.):+.4f}\n'
                f'Near assist: {info.get("near_guard",{}).get("active",False)} | Executed: {info.get("executed_normalized_action",action.tolist())}\n'
                'Three views show the TEST world. Separate PyBullet window holds the training world.\n'
                'Training resumes automatically when validation finishes.'))
            self.last_draw = now
        if finished or now - self.last_status >= 1.:
            status = dict(pid=os.getpid(), state='validating', mode='validation',
                          step=training_step, total_steps=training_total, updates=updates,
                          test_group=group, test_episode=episode, test_episodes=episodes,
                          test_sim_time=info['time_s'], event=info['event'])
            status['center_estimate']=update_center_marker(env)
            status['near_guard']=info.get('near_guard',{})
            status['prediction_reward']=info.get('prediction_reward',0.)
            status['prediction_guidance']=env.cfg.prediction_guidance
            (self.output/'live_status.json').write_text(json.dumps(status, indent=2))
            self.last_status = now
        self.pump()

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
