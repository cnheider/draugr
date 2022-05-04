__all__ = [
    "get_screen_resolution",
    "screen_res_tk",
    "screen_res_pygame",
    "screen_res_gtk",
    "screen_res_xlib",
    "screen_res_mac",
    "screen_res_win",
]

import sys


def screen_res_xlib():
    import xlib

    root = xlib.display.Display().screen().root
    return root.get_geometry()[2], root.get_geometry()[3]


def screen_res_pygame():
    import pygame

    return pygame.display.Info().current_w, pygame.display.Info().current_h


def screen_res_gtk():
    import gtk

    screen = gtk.Window().get_screen()
    return screen.get_width(), screen.get_height()


def screen_res_tk():
    import Tkinter

    root = Tkinter.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height


def screen_res_win():
    import ctypes.wintypes

    return ctypes.windll.user32.GetSystemMetrics(
        0
    ), ctypes.windll.user32.GetSystemMetrics(1)


def screen_res_mac():
    try:
        import Quartz
    except:
        assert (
            False
        ), "You must first install pyobjc-core and pyobjc: MACOSX_DEPLOYMENT_TARGET=10.11 pip install pyobjc"
    main_disp_id = Quartz.CGMainDisplayID()
    return Quartz.CGDisplayPixelsWide(main_disp_id), Quartz.CGDisplayPixelsHigh(
        main_disp_id
    )


def get_screen_resolution():
    if sys.platform == "win32":
        return screen_res_win()
    elif sys.platform == "darwin":
        return screen_res_mac()
    elif sys.platform == "linux":
        return screen_res_gtk()
    elif sys.platform == "linux2":
        return screen_res_gtk()
    elif sys.platform == "cygwin":
        return screen_res_gtk()
    else:
        return screen_res_tk()
