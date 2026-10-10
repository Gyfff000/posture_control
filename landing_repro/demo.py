"""Two real Linux GUI windows: PyBullet physics and its onboard camera."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import imageio_ffmpeg
import numpy as np
import pybullet as p
from PIL import Image, ImageTk
import tkinter as tk

from .agent import DrQV2
from .checks import oracle_action
from .config import EnvConfig
from .environment import MovingPlatformLanding


def capture_window(out):
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
    windows = re.findall(r'(0x[0-9a-fA-F]+) "Bullet Physics[^\n]*', tree)
    if len(windows) == 1:
        raw = out / "window.xwd"
        subprocess.run(["xwd", "-silent", "-id", windows[0], "-out", str(raw)], check=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(raw), "-frames:v", "1", str(out / "simulation_window.png")], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["oracle", "drq"], default="oracle")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/cpu_pilot_lr1e4/latest.pt"))
    parser.add_argument("--minutes", type=float, default=30)
    args = parser.parse_args()
    agent = None
    if args.controller == "drq":
        agent, _ = DrQV2.load(args.checkpoint)
    # Close-range scene is for visibility, not the paper's 10 +/- 3 m evaluation.
    cfg = EnvConfig(initial_height=2.0, offset_xy=.25, offset_z=0,
                    platform_speed_min=.4, platform_speed_max=.4)
    env = MovingPlatformLanding(cfg, gui=True)
    out = Path("runs") / ("desktop_" + args.controller)
    out.mkdir(parents=True, exist_ok=True)
    title = "TRUTH-BASED PID DEBUG - not DrQ" if args.controller == "oracle" else "DrQ-v2 CPU SHORT TRAINING - unproven policy"
    root = tk.Tk()
    root.title("Onboard camera | " + title)
    root.geometry("320x370+1080+80")
    label = tk.Label(root, text=title, wraplength=310)
    label.pack()
    camera = tk.Label(root)
    camera.pack()
    stop = [False]
    root.protocol("WM_DELETE_WINDOW", lambda: stop.__setitem__(0, True))
    finish = time.monotonic() + args.minutes * 60
    captured = False
    episode = 0
    try:
        with (out / "events.jsonl").open("w") as log:
            while time.monotonic() < finish and not stop[0] and p.isConnected(env.client):
                obs, _ = env.reset(seed=100 + episode)
                start = time.monotonic()
                step, debug_id = 0, -1
                while time.monotonic() < finish and not stop[0] and p.isConnected(env.client):
                    action = oracle_action(env) if args.controller == "oracle" else agent.act(obs, evaluation=True)
                    obs, _, term, trunc, info = env.step(action)
                    if step % 3 == 0:
                        env.update_debug_camera()
                        image = ImageTk.PhotoImage(Image.fromarray(env.rgb).resize((320, 320)))
                        camera.configure(image=image)
                        camera.image = image
                        label.configure(text=f"{title}\nt={info['time_s']:.1f}s   height={info['relative_position'][2]:.2f}m")
                        root.update()
                        debug_id = p.addUserDebugText(title, env.platform_position + [0, 0, .2], [0, 0, 0],
                                                      textSize=1.1, replaceItemUniqueId=debug_id,
                                                      physicsClientId=env.client)
                    if not captured and info["time_s"] > 2.:
                        cam = p.getDebugVisualizerCamera(physicsClientId=env.client)
                        _, _, rgba, _, _ = p.getCameraImage(640, 480, cam[2], cam[3],
                            renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=env.client)
                        viewport = np.asarray(rgba, np.uint8).reshape(480, 640, 4)[..., :3]
                        Image.fromarray(viewport).save(out / "rendered_viewport.png")
                        try:
                            capture_window(out)
                        except (OSError, subprocess.CalledProcessError) as exc:
                            print(f"Optional window capture unavailable: {exc}", flush=True)
                        Image.fromarray(env.rgb).resize((420, 420)).save(out / "onboard_camera.png")
                        captured = True
                    step += 1
                    delay = start + step / cfg.policy_hz - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    if term or trunc:
                        log.write(json.dumps({"controller": args.controller, "episode": episode, **info}) + "\n")
                        log.flush()
                        print(f"{title}: episode={episode} event={info['event']}", flush=True)
                        time.sleep(.8)
                        break
                episode += 1
    except (KeyboardInterrupt, p.error, tk.TclError) as exc:
        print(f"Demo closed: {exc}")
    finally:
        env.close()
        root.destroy()


if __name__ == "__main__":
    main()
