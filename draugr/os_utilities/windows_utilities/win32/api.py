import win32api
import win32con
import win32gui


def press_key(hwnd, s, key, start_sec, hold_sec):
    """
    send a keyboard input to the given window

    :param hwnd:
    :type hwnd:
    :param s:
    :type s:
    :param key:
    :type key:
    :param start_sec:
    :type start_sec:
    :param hold_sec:
    :type hold_sec:
    :return:
    :rtype:
    """
    priority = 2
    duration = start_sec + hold_sec

    s.enter(
        start_sec,
        priority,
        win32api.SendMessage,
        argument=(hwnd, win32con.WM_KEYDOWN, key, 0),
    )
    s.enter(
        duration,
        priority,
        win32api.SendMessage,
        argument=(hwnd, win32con.WM_KEYUP, key, 0),
    )

    # win32gui.SetForegroundWindow(hwnd)
    # win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, key, 0)
    # sleep(sec)
    # win32api.SendMessage(hwnd, win32con.WM_KEYUP, key, 0)


def press_key_foreground(hwnd, s, key, start_sec, hold_sec):
    """
    # send a keyboard input to the given window
    """
    priority = 2
    foreground_time = 0.15
    duration = start_sec + hold_sec

    s.enter(
        start_sec - foreground_time,
        priority,
        win32gui.SetForegroundWindow,
        argument=(hwnd,),
    )
    s.enter(
        start_sec,
        priority,
        win32api.SendMessage,
        argument=(hwnd, win32con.WM_KEYDOWN, key, 0),
    )
    s.enter(
        duration - foreground_time,
        priority,
        win32gui.SetForegroundWindow,
        argument=(hwnd,),
    )
    s.enter(
        duration,
        priority,
        win32api.SendMessage,
        argument=(hwnd, win32con.WM_KEYUP, key, 0),
    )

    # win32gui.SetForegroundWindow(hwnd)
    # win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, key, 0)
    # sleep(sec)
    # win32api.SendMessage(hwnd, win32con.WM_KEYUP, key, 0)
