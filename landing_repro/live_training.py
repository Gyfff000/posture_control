"""Desktop monitor for the SAME environment and agent used by train.py."""
import json
import os
import time
import tkinter as tk
from tkinter import scrolledtext

from PIL import Image, ImageTk


class LiveTraining:
    def __init__(self, output):
        self.output = output
        self.stop = False
        self.paused = False
        self.last_draw = 0.
        self.last_status = 0.
        self.root = tk.Tk()
        self.root.title("LIVE DrQ-v2 TRAINING | actual model + exploration")
        self.root.geometry("460x850+920+30")
        self.root.protocol("WM_DELETE_WINDOW", self.request_stop)
        tk.Label(self.root, text="LIVE DrQ-v2 CPU TRAINING", font=("Arial", 15, "bold")).pack()
        tk.Label(self.root, text="Actual training environment | random warm-up, then noisy policy\n"
                 "A successful landing is NOT guaranteed.", wraplength=450).pack()
        self.camera = tk.Label(self.root)
        self.camera.pack()
        self.stats = tk.Label(self.root, text="Starting...", justify="left", anchor="w", font=("Courier", 11))
        self.stats.pack(fill="x", padx=10)
        controls = tk.Frame(self.root)
        controls.pack(pady=6)
        self.pause_button = tk.Button(controls, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=8)
        tk.Button(controls, text="Stop and save model", command=self.request_stop).pack(side="left")
        tk.Label(self.root, text="Episode results (success / ground / edge / timeout)").pack()
        self.log = scrolledtext.ScrolledText(self.root, height=9, width=57, state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=5)
        tk.Label(self.root, text="Model and logs: " + str(output), wraplength=450).pack()
        self.pump()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text="Continue" if self.paused else "Pause")

    def request_stop(self):
        self.stop = True

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
        line = f"Episode {episode}: {info['event'].upper()} | return={episode_return:.2f}\n"
        self.log.configure(state="normal")
        self.log.insert("end", line)
        self.log.see("end")
        self.log.configure(state="disabled")

    def update(self, env, step, total, warmup, updates, episode, successes, replay_size,
               episode_return, action, info, metrics, elapsed, force=False):
        now = time.monotonic()
        phase = "RANDOM WARM-UP (model not controlling)" if step <= warmup else "MODEL + EXPLORATION NOISE"
        status = {"pid": os.getpid(), "step": step, "total_steps": total, "phase": phase,
                  "updates": updates, "episodes_completed": episode, "successes": successes,
                  "replay_size": replay_size, "event": info["event"], "time_s": info["time_s"],
                  "relative_position": info["relative_position"], "action": action.tolist(),
                  "wall_seconds": elapsed, "metrics": metrics, "state": "running"}
        if force or now - self.last_status >= 1.:
            (self.output / "live_status.json").write_text(json.dumps(status, indent=2))
            self.last_status = now
        if force or now - self.last_draw >= .12:
            env.update_debug_camera()
            image = ImageTk.PhotoImage(Image.fromarray(env.rgb).resize((300, 300)))
            self.camera.configure(image=image)
            self.camera.image = image
            loss = metrics.get("critic_loss")
            loss_text = "pending" if loss is None else f"{loss:.5f}"
            self.stats.configure(text=(f"{phase}\n"
                f"Steps: {step:6d} / {total}   Updates: {updates}\n"
                f"Episode: {episode+1}   Completed: {episode}\n"
                f"Training landings: {successes}   Replay: {replay_size}\n"
                f"Sim time: {info['time_s']:5.1f}s   Wall: {elapsed:6.1f}s\n"
                f"Height above deck: {info['relative_position'][2]:6.2f} m\n"
                f"Normalized action: {action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}\n"
                f"Episode return: {episode_return:8.2f}\n"
                f"Critic loss: {loss_text}\n"
                f"Current result: {info['event'].upper()}"))
            self.last_draw = now
        self.pump()

    def hold_result(self, seconds=.8):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.stop:
            self.pump()
            time.sleep(.03)

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
