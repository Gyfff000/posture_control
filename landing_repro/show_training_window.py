"""Ask the running monitor to restore itself; does not start another trainer."""
from pathlib import Path
import time


def main():
    root=Path(__file__).resolve().parent.parent
    run=Path((root/'runs/current_smooth_training.txt').read_text().strip())
    if not run.is_absolute():
        run=root/run
    pid=int((run/'training.pid').read_text())
    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(bytes([0]))
    if b'landing_repro.smooth' not in command or str(run.relative_to(root)).encode() not in command:
        raise RuntimeError('The recorded trainer is no longer running')
    request=run/'show_window.request'
    request.write_text(str(time.time()))
    for _ in range(50):
        if not request.exists():
            print('Training window restored and raised by the running monitor.')
            try:
                from .x11_window import restore_bullet_window
                print('PyBullet restore requested:', restore_bullet_window())
            except (OSError, RuntimeError) as error:
                print('Training continues; could not restore the separate 3D window:', error)
            return
        time.sleep(.1)
    raise RuntimeError('Monitor did not acknowledge the display request; model was not restarted')


if __name__=='__main__': main()
