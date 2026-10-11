"""Two native Windows viewing windows for the currently running Linux trainer."""
import io
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import tkinter as tk
from urllib.request import urlopen

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent


def get_status(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/status', timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def ensure_service(port, view, run_name):
    current = get_status(port)
    if current is None:
        run = ROOT / run_name
        with (run / f'desktop_{view}.stdout.log').open('ab') as stdout, (run / f'desktop_{view}.stderr.log').open('ab') as stderr:
            subprocess.Popen([
                'wsl.exe', '-d', 'Ubuntu-20.04', '--cd', '/mnt/d/paper_reproductions/posture_control',
                '--exec', '/home/ubuntu/miniforge3/envs/landing-drqv2/bin/python', '-m',
                'landing_repro.desktop_monitor', '--view', view, '--port', str(port),
            ], stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = get_status(port)
            if current:
                break
            time.sleep(.3)
    if not current or not current.get('trainer_alive') or current.get('run') != run_name:
        raise RuntimeError(f'没有连接到本次训练的 {view} 画面。当前训练没有被重启。')


class View:
    def __init__(self, root, port, title, geometry, report, key):
        self.port, self.report, self.key = port, report, key
        self.closed = False
        self.frames = queue.Queue(maxsize=1)
        self.window = tk.Toplevel(root)
        self.window.title(title)
        self.window.geometry(geometry)
        self.window.configure(background='#eeeeee')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.message = tk.Label(self.window, text='正在连接当前训练…', anchor='w', font=('Microsoft YaHei', 11))
        self.message.pack(fill='x', padx=8, pady=6)
        self.image = tk.Label(self.window, background='#eeeeee')
        self.image.pack(fill='both', expand=True)
        tk.Label(self.window, text='实时观看窗口；图中按钮和鼠标视角操作不可用。关闭此窗口不会停止训练。',
                 wraplength=700, fg='#555555').pack(fill='x', padx=8, pady=4)
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(2500, lambda: None if self.closed else self.window.attributes('-topmost', False))
        threading.Thread(target=self.fetch, daemon=True).start()
        self.rendered = 0
        self.tick()

    def fetch(self):
        while not self.closed:
            start = time.monotonic()
            try:
                with urlopen(f'http://127.0.0.1:{self.port}/frame.jpg', timeout=2) as response:
                    frame = Image.open(io.BytesIO(response.read())).convert('RGB')
                status = get_status(self.port) or {}
                item = (frame, status, '')
            except Exception as error:
                item = (None, {}, str(error))
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put(item)
            time.sleep(max(0., .3 - (time.monotonic() - start)))

    def tick(self):
        if self.closed:
            return
        try:
            frame, status, error = self.frames.get_nowait()
            if frame is not None:
                width = max(1, self.image.winfo_width())
                height = max(1, self.image.winfo_height())
                frame.thumbnail((width, height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(frame)
                self.image.configure(image=photo)
                self.image.image = photo
                self.rendered += 1
                age = time.time() - status.get('frame_time', time.time())
                state = {'running': '训练中', 'validating': '独立验证中'}.get(status.get('state'), status.get('state', '等待状态'))
                self.message.configure(text=f"{state} ｜ 步数 {status.get('step', '-')} ｜ 更新 {status.get('updates', '-')} ｜ 画面延迟 {age:.1f} 秒")
                self.report[self.key] = dict(rendered_frames=self.rendered, updated=time.time(),
                                             step=status.get('step'), state=status.get('state'),
                                             source_frame_time=status.get('frame_time'), error=status.get('error', ''))
            else:
                self.message.configure(text='画面连接中断：' + error)
                self.report[self.key] = dict(error=error, updated=time.time())
        except queue.Empty:
            pass
        self.window.after(70, self.tick)

    def close(self):
        self.closed = True
        self.window.destroy()


def main():
    root = tk.Tk()
    root.withdraw()
    run_name = (ROOT / 'runs/current_smooth_training.txt').read_text().strip()
    try:
        ensure_service(8765, 'training', run_name)
        ensure_service(8766, 'physics', run_name)
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        left = max(640, int(width * .55))
        right = max(500, width - left - 60)
        window_height = min(960, height - 120)
        report = {'run': run_name}
        views = [View(root, 8765, '无人机训练监控 — 三个视角与实时日志', f'{left}x{window_height}+10+35', report, 'training'),
                 View(root, 8766, '无人机三维仿真 — 当前训练的真实画面', f'{right}x{window_height}+{left + 30}+35', report, 'physics')]

        def heartbeat():
            if all(view.closed for view in views):
                root.destroy()
                return
            (ROOT / 'runs/desktop_dual_viewer_status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            root.after(1000, heartbeat)

        heartbeat()
        root.mainloop()
    except Exception as error:
        from tkinter import messagebox
        messagebox.showerror('无人机双窗口', str(error))
        root.destroy()


if __name__ == '__main__':
    main()
