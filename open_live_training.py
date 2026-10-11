"""Windows entry point: open the current training stream in its own Edge window."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
URL = 'http://localhost:8765'


def status():
    try:
        with urlopen(URL + '/status', timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def main():
    run_name = (ROOT / 'runs/current_smooth_training.txt').read_text().strip()
    current = status()
    if current is None:
        output = ROOT / run_name
        with (output / 'desktop_monitor.stdout.log').open('ab') as stdout, (output / 'desktop_monitor.stderr.log').open('ab') as stderr:
            subprocess.Popen([
                'wsl.exe', '-d', 'Ubuntu-20.04', '--cd', '/mnt/d/paper_reproductions/posture_control',
                '--exec', '/home/ubuntu/miniforge3/envs/landing-drqv2/bin/python',
                '-m', 'landing_repro.desktop_monitor',
            ], stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = status()
            if current is not None:
                break
            time.sleep(.5)
    if not current or not current.get('trainer_alive'):
        raise RuntimeError('没有连接到正在运行的训练。请先使用“继续变速平台训练.cmd”。查看当前运行目录中的 desktop_monitor.stderr.log 可了解画面服务错误。')
    if current.get('run') != run_name:
        raise RuntimeError('画面服务连接的是旧的训练目录，请关闭旧画面服务后重开。训练本身没有被修改。')
    candidates = [Path(os.environ.get(name, 'C:/Program Files')) / 'Microsoft/Edge/Application/msedge.exe'
                  for name in ('ProgramFiles(x86)', 'ProgramFiles')]
    edge = next((path for path in candidates if path.is_file()), None)
    if edge:
        subprocess.Popen([str(edge), '--app=' + URL, '--window-size=1200,960', '--window-position=40,40'])
    else:
        webbrowser.open(URL)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('无人机实时训练画面', str(error))
        root.destroy()
