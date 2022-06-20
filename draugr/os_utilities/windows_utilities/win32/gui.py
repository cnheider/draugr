import win32gui

__all__ = ["list_window_names", "get_inner_windows", "find_all_windows"]

verbose = False


def list_window_names():
    """

    :return:
    :rtype:
    """
    result = []

    def win_enum_handler(hwnd, ctx):
        """

        :param hwnd:
        :type hwnd:
        :param ctx:
        :type ctx:
        """
        if win32gui.IsWindowVisible(hwnd):
            if verbose:
                print(hex(hwnd), f'"{str(win32gui.GetWindowText(hwnd))}"')
            result.append(win32gui.GetWindowText(hwnd))

    win32gui.EnumWindows(win_enum_handler, None)
    return result


def get_inner_windows(whndl):
    """

    :param whndl:
    :type whndl:
    :return:
    :rtype:
    """

    def callback(hwnd, hwnds):
        """

        :param hwnd:
        :type hwnd:
        :param hwnds:
        :type hwnds:
        :return:
        :rtype:
        """
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            hwnds[win32gui.GetClassName(hwnd)] = hwnd
        return True

    hwnds = {}
    win32gui.EnumChildWindows(whndl, callback, hwnds)
    return hwnds


def find_all_windows(name):
    """

    :param name:
    :type name:
    :return:
    :rtype:
    """
    result = []

    def win_enum_handler(hwnd, ctx):
        """

        :param hwnd:
        :type hwnd:
        :param ctx:
        :type ctx:
        """
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == name:
            result.append(hwnd)

    win32gui.EnumWindows(win_enum_handler, None)
    return result


if __name__ == "__main__":
    print(get_inner_windows(find_all_windows(list_window_names()[-1])[-1]))
