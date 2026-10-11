"""Serve the actual Linux training window to a local desktop browser."""
import argparse
import ctypes as c
import ctypes.util
import io
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


class LinuxWindowCapture:
    def __init__(self, view='training'):
        self.x = x = c.CDLL(ctypes.util.find_library('X11'))
        d, w = c.c_void_p, c.c_ulong
        x.XOpenDisplay.argtypes = [c.c_char_p]
        x.XOpenDisplay.restype = d
        x.XDefaultRootWindow.argtypes = [d]
        x.XDefaultRootWindow.restype = w
        x.XQueryTree.argtypes = [d, w, c.POINTER(w), c.POINTER(w), c.POINTER(c.POINTER(w)), c.POINTER(c.c_uint)]
        x.XFetchName.argtypes = [d, w, c.POINTER(c.c_void_p)]
        x.XFree.argtypes = [c.c_void_p]
        x.XGetGeometry.argtypes = [d, w, c.POINTER(w), c.POINTER(c.c_int), c.POINTER(c.c_int),
                                  c.POINTER(c.c_uint), c.POINTER(c.c_uint), c.POINTER(c.c_uint), c.POINTER(c.c_uint)]
        x.XGetImage.argtypes = [d, w, c.c_int, c.c_int, c.c_uint, c.c_uint, c.c_ulong, c.c_int]

        class XImage(c.Structure):
            _fields_ = [('width', c.c_int), ('height', c.c_int), ('xoffset', c.c_int), ('format', c.c_int),
                        ('data', c.c_void_p), ('byte_order', c.c_int), ('bitmap_unit', c.c_int),
                        ('bitmap_bit_order', c.c_int), ('bitmap_pad', c.c_int), ('depth', c.c_int),
                        ('bytes_per_line', c.c_int), ('bits_per_pixel', c.c_int),
                        ('red_mask', c.c_ulong), ('green_mask', c.c_ulong), ('blue_mask', c.c_ulong)]

        x.XGetImage.restype = c.POINTER(XImage)
        x.XDestroyImage.argtypes = [c.POINTER(XImage)]
        x.XCloseDisplay.argtypes = [d]
        self.d = x.XOpenDisplay(None)
        if not self.d:
            raise RuntimeError('Linux display is unavailable')
        matches = []

        def visit(window, depth=0):
            name = c.c_void_p()
            if x.XFetchName(self.d, window, c.byref(name)) and name.value:
                try:
                    title = c.string_at(name).decode(errors='replace')
                    prefixes = ('Bullet Physics ExampleBrowser',) if view == 'physics' else ('UAV LANDING TRAINING', 'LIVE DrQ-v2 TRAINING', 'ASSISTED VISUAL')
                    if title.startswith(prefixes):
                        matches.append(window)
                finally:
                    x.XFree(name)
            if depth >= 3:
                return
            root, parent, children, count = w(), w(), c.POINTER(w)(), c.c_uint()
            if x.XQueryTree(self.d, window, c.byref(root), c.byref(parent), c.byref(children), c.byref(count)):
                ids = [children[i] for i in range(count.value)]
                if children:
                    x.XFree(children)
                for child in ids:
                    visit(child, depth + 1)

        visit(x.XDefaultRootWindow(self.d))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one running training monitor; found {len(matches)}')
        self.window = matches[0]

    def capture(self):
        root, px, py = c.c_ulong(), c.c_int(), c.c_int()
        width, height, border, depth = (c.c_uint() for _ in range(4))
        if not self.x.XGetGeometry(self.d, self.window, c.byref(root), c.byref(px), c.byref(py),
                                  c.byref(width), c.byref(height), c.byref(border), c.byref(depth)):
            raise RuntimeError('Training window closed')
        pointer = self.x.XGetImage(self.d, self.window, 0, 0, width.value, height.value, c.c_ulong(-1).value, 2)
        if not pointer:
            raise RuntimeError('Could not read the Linux training window')
        try:
            im = pointer.contents
            if (im.bits_per_pixel, im.byte_order, im.red_mask, im.green_mask, im.blue_mask) != (32, 0, 0xff0000, 0xff00, 0xff):
                raise RuntimeError('Unsupported Linux window pixel format')
            data = c.string_at(im.data, im.bytes_per_line * im.height)
            return Image.frombytes('RGB', (im.width, im.height), data, 'raw', 'BGRX', im.bytes_per_line, 1)
        finally:
            self.x.XDestroyImage(pointer)


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>无人机实时训练画面</title>
<style>body{margin:16px;background:#f3f4f6;color:#18202c;font:16px Arial,"Microsoft YaHei",sans-serif}h1{font-size:23px;margin:8px 0}p{margin:8px 0}img{max-width:100%;height:auto;background:white;border:1px solid #aaa}#status{font-weight:bold} .note{color:#555}</style>
<h1>无人机移动平台降落 · 正在运行的训练界面</h1>
<p id="status">正在连接训练进程…</p>
<p class="note">下面是现有训练窗口的实时画面（约每秒4帧），图中的按钮不能点击。关闭本网页不会停止训练。验证期间训练步数会暂停，画面显示独立测试飞行。</p>
<img src="/stream" alt="正在连接实时训练画面">
<p id="detail"></p>
<script>async function tick(){try{let r=await fetch('/status',{cache:'no-store'});let s=await r.json();let age=(Date.now()/1000-s.frame_time).toFixed(1);document.getElementById('status').textContent=`${s.trainer_alive?'训练进程在线':'训练进程已结束'} ｜ 累计步数 ${s.step??'-'} ｜ 参数更新 ${s.updates??'-'} ｜ 状态 ${s.state??'-'} ｜ 画面延迟 ${age} 秒`;document.getElementById('detail').textContent=s.error||`实际训练目录：${s.run}；这些画面来自正在训练的环境，没有启动另一个模型演示。`;}catch(e){document.getElementById('status').textContent='实时画面服务已断开，训练进程的状态需要另行确认。';}}setInterval(tick,1000);tick();</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--view', choices=['training', 'physics'], default='training')
    parser.add_argument('--capture-once')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    relative = (root / 'runs/current_smooth_training.txt').read_text().strip()
    run = root / relative
    pid = int((run / 'training.pid').read_text())

    def alive():
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().split(bytes([0]))
            return b'landing_repro.smooth' in command and relative.encode() in command
        except OSError:
            return False

    if not alive():
        raise RuntimeError('The current trainer is not running')
    capture = LinuxWindowCapture(args.view)
    if args.capture_once:
        capture.capture().save(args.capture_once)
        print('Saved actual training window:', args.capture_once)
        return
    shared = {'jpeg': b'', 'frame_time': 0., 'error': ''}

    def producer():
        while alive():
            start = time.monotonic()
            try:
                frame = capture.capture()
                buffer = io.BytesIO()
                frame.save(buffer, format='JPEG', quality=85)
                shared.update(jpeg=buffer.getvalue(), frame_time=time.time(), error='')
            except Exception as error:
                shared['error'] = str(error)
                break
            time.sleep(max(0., .25 - (time.monotonic() - start)))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                previous = 0.
                try:
                    while alive():
                        jpeg, stamp = shared['jpeg'], shared['frame_time']
                        if jpeg and stamp != previous:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                            self.wfile.flush()
                            previous = stamp
                        time.sleep(.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == '/frame.jpg':
                body, kind = shared['jpeg'], 'image/jpeg'
            elif self.path == '/frame.png':
                if not shared['jpeg']:
                    self.send_error(503, 'Waiting for first frame')
                    return
                buffer = io.BytesIO()
                Image.open(io.BytesIO(shared['jpeg'])).save(buffer, format='PNG')
                body, kind = buffer.getvalue(), 'image/png'
            elif self.path == '/status':
                try:
                    status = json.loads((run / 'live_status.json').read_text())
                except (OSError, ValueError):
                    status = {}
                status.update(run=relative, trainer_alive=alive(), frame_time=shared['frame_time'], error=shared['error'], view=args.view)
                body, kind = json.dumps(status).encode(), 'application/json'
            elif self.path == '/':
                page = PAGE
                if args.view == 'physics':
                    page = page.replace('无人机实时训练画面', '无人机三维仿真实时画面').replace('无人机移动平台降落 · 正在运行的训练界面', '无人机移动平台降落 · 当前三维仿真窗口')
                body, kind = page.encode(), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=producer, daemon=True).start()
    print(f'Actual training window live at http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
