#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/9/22
           """

__all__ = []


def uahsduiasdj():
    """

    :return:
    :rtype:
    """
    import wx  # pip install -U wxPython
    from PIL import Image

    SIZE = (640, 480)

    def get_image():
        """

        :return:
        :rtype:
        """
        # Put your code here to return a PIL image from the camera.
        return Image.new("L", SIZE)

    def pil_to_wx(image):
        """

        :param image:
        :type image:
        :return:
        :rtype:
        """
        width, height = image.size
        buffer = image.convert("RGB").tostring()
        bitmap = wx.BitmapFromBuffer(width, height, buffer)
        return bitmap

    class Panel(wx.Panel):
        """ """

        def __init__(self, parent):
            super(Panel, self).__init__(parent, -1)
            self.SetSize(SIZE)
            self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
            self.Bind(wx.EVT_PAINT, self.on_paint)
            self.update()

        def update(self):
            """ """
            self.Refresh()
            self.Update()
            wx.CallLater(15, self.update)

        def create_bitmap(self):
            """

            :return:
            :rtype:
            """
            image = get_image()
            bitmap = pil_to_wx(image)
            return bitmap

        def on_paint(self, event):
            """

            :param event:
            :type event:
            """
            bitmap = self.create_bitmap()
            dc = wx.AutoBufferedPaintDC(self)
            dc.DrawBitmap(bitmap, 0, 0)

    class Frame(wx.Frame):
        """ """

        def __init__(self):
            style = wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER & ~wx.MAXIMIZE_BOX
            super(Frame, self).__init__(None, -1, "Camera Viewer", style=style)
            panel = Panel(self)
            self.Fit()

    app = wx.PySimpleApp()
    frame = Frame()
    frame.Center()
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    uahsduiasdj()
