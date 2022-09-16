#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/9/22
           """

__all__ = []

import cv2
from PIL import ImageTk, Image

from draugr.opencv_utilities import AsyncVideoStream


def asiudjh():
    import tkinter

    class App(tkinter.Tk):
        def __init__(self, generator, delay_ms=10):
            tkinter.Tk.__init__(self)
            self.label = tkinter.Label(text="your image here", compound="top")
            self.label.pack(side="top", padx=8, pady=8)
            self.iteration = 0
            self.delay_ms = delay_ms
            self.image = generator()
            self.update(pack=(generator, delay_ms))

        def update(self, pack, event=None):
            generator, delay_ms = pack
            self.iteration += 1
            self.image = generator()
            self.delay_ms = delay_ms
            self.label.configure(image=self.image, text=f"Iteration {self.iteration}")
            self.after(self.delay_ms, self.update, pack)  # reschedule func

    with AsyncVideoStream() as vs:
        app = App(
            generator=lambda: ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(vs.read()[-1], cv2.COLOR_BGR2RGBA))
            ),
            delay_ms=1,
        )
        app.mainloop()


if __name__ == "__main__":
    asiudjh()
