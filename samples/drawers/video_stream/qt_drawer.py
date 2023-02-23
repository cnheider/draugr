#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/9/22
           """

__all__ = []

from PIL.ImageQt import ImageQt
from PyQt6 import QtGui
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QLabel, QScrollArea, QApplication

from draugr.opencv_utilities import AsyncVideoStream


def aushduah():
    """

    :return:
    :rtype:
    """
    import sys

    class CameraViewer(QMainWindow):
        """ """

        def __init__(self, hz=33):
            super().__init__()

            self.imageLabel = QLabel()
            self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
            self.imageLabel.setScaledContents(True)

            self.scrollArea = QScrollArea()
            self.scrollArea.setWidget(self.imageLabel)
            self.setCentralWidget(self.scrollArea)

            self.setWindowTitle("Image Viewer")
            self.resize(640, 480)

            timer = QTimer(self)
            timer.timeout.connect(self.open)
            timer.start(hz)

        def open(self):
            """

            :return:
            :rtype:
            """
            # get data and display
            _, pilimg = vs.read()
            image = ImageQt(pilimg)
            if image.isNull():
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.imageLabel.adjustSize()

    with AsyncVideoStream() as vs:
        app = QApplication(sys.argv)
        camera_viewer = CameraViewer()
        camera_viewer.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    aushduah()
