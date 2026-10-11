"""Restore the Linux PyBullet window through its X11 window manager."""
import ctypes as c
import ctypes.util
import time


def restore_bullet_window():
    x = c.CDLL(ctypes.util.find_library('X11'))
    display = c.c_void_p
    window = c.c_ulong
    x.XOpenDisplay.argtypes = [c.c_char_p]
    x.XOpenDisplay.restype = display
    x.XDefaultRootWindow.argtypes = [display]
    x.XDefaultRootWindow.restype = window
    x.XQueryTree.argtypes = [display, window, c.POINTER(window), c.POINTER(window),
                            c.POINTER(c.POINTER(window)), c.POINTER(c.c_uint)]
    x.XFetchName.argtypes = [display, window, c.POINTER(c.c_void_p)]
    x.XFree.argtypes = [c.c_void_p]
    x.XMapRaised.argtypes = [display, window]
    x.XWithdrawWindow.argtypes = [display, window, c.c_int]
    x.XSync.argtypes = [display, c.c_int]
    x.XMoveWindow.argtypes = [display, window, c.c_int, c.c_int]
    x.XInternAtom.argtypes = [display, c.c_char_p, c.c_int]
    x.XInternAtom.restype = window
    x.XFlush.argtypes = [display]
    x.XCloseDisplay.argtypes = [display]

    class Data(c.Union):
        _fields_ = [('b', c.c_char * 20), ('s', c.c_short * 10), ('l', c.c_long * 5)]

    class ClientMessage(c.Structure):
        _fields_ = [('type', c.c_int), ('serial', c.c_ulong), ('send_event', c.c_int),
                    ('display', display), ('window', window), ('message_type', window),
                    ('format', c.c_int), ('data', Data)]

    class Event(c.Union):
        _fields_ = [('client', ClientMessage), ('pad', c.c_long * 24)]

    x.XSendEvent.argtypes = [display, window, c.c_int, c.c_long, c.POINTER(Event)]
    d = x.XOpenDisplay(None)
    if not d:
        raise RuntimeError('Cannot connect to the Linux graphical display')
    root = x.XDefaultRootWindow(d)
    matches = []

    def visit(w, depth=0):
        name = c.c_void_p()
        if x.XFetchName(d, w, c.byref(name)) and name.value:
            try:
                if c.string_at(name).startswith(b'Bullet Physics ExampleBrowser'):
                    matches.append(w)
            finally:
                x.XFree(name)
        if depth >= 3:
            return
        r, parent, children, count = window(), window(), c.POINTER(window)(), c.c_uint()
        if x.XQueryTree(d, w, c.byref(r), c.byref(parent), c.byref(children), c.byref(count)):
            ids = [children[i] for i in range(count.value)]
            if children:
                x.XFree(children)
            for child in ids:
                visit(child, depth + 1)

    try:
        visit(root)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one PyBullet window, found {len(matches)}')
        w = matches[0]
        # Re-map an offscreen WSLg surface so the compositor also recreates its
        # Windows presentation. Raising an already mapped surface is insufficient.
        x.XWithdrawWindow(d, w, 0)
        x.XSync(d, 0)
        time.sleep(.2)
        x.XMapRaised(d, w)
        x.XMoveWindow(d, w, 80, 80)
        event = Event()
        event.client.type = 33  # ClientMessage
        event.client.display = d
        event.client.window = w
        event.client.message_type = x.XInternAtom(d, b'_NET_ACTIVE_WINDOW', 0)
        event.client.format = 32
        event.client.data.l[0] = 2  # Application-independent window management request
        x.XSendEvent(d, root, 0, (1 << 20) | (1 << 19), c.byref(event))
        x.XFlush(d)
        return hex(w)
    finally:
        x.XCloseDisplay(d)
