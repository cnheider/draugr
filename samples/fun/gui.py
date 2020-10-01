#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 30-09-2020
           """

# !/usr/bin/python
import tkinter


class FullscreenBorderlessApp(tkinter.Frame):
    def __init__(self, parent, *args, **kwargs):
        tkinter.Frame.__init__(self, parent)

        self.parent = parent

        self.parent.title("Fullscreen Application")

        self.pack(fill="both", expand=True, side="top")

        self.parent.wm_state("zoomed")

        self.parent.bind("<F11>", self.fullscreen_toggle)
        self.parent.bind("<Escape>", self.fullscreen_cancel)

        self.fullscreen_toggle()

        self.label = tkinter.Label(
            self, text="Fullscreen", font=("default", 120), fg="black"
        )
        self.label.pack(side="top", fill="both", expand=True)

    def fullscreen_toggle(self, event="none"):
        self.parent.focus_set()
        self.parent.overrideredirect(True)
        self.parent.overrideredirect(
            False
        )  # added for a toggle effect, not fully sure why it's like this on Mac OS
        self.parent.attributes("-fullscreen", True)
        self.parent.wm_attributes("-topmost", 1)

    def fullscreen_cancel(self, event="none"):
        self.parent.overrideredirect(False)
        self.parent.attributes("-fullscreen", False)
        self.parent.wm_attributes("-topmost", 0)

        self.center_window()

    def center_window(self):
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()

        w = screen_width * 0.7
        h = screen_height * 0.7

        x = int((screen_width - w) / 2)
        y = int((screen_height - h) / 2)

        self.parent.geometry(f"{int(w):d}x{int(h):d}+{x:d}+{y:d}")


if __name__ == "__main__":
    root = tkinter.Tk()
    FullscreenBorderlessApp(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
