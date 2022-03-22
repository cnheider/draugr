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


def screen_res():
    import ctypes.wintypes

    return ctypes.windll.user32.GetSystemMetrics(
        0
    ), ctypes.windll.user32.GetSystemMetrics(1)


def mac_res():
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
