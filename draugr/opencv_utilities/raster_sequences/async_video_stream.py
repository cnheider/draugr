#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25-01-2021
           """

from threading import Thread
from typing import Any, Union, Sequence, MutableMapping, Tuple

import cv2

__all__ = ["AsyncVideoStream"]

import numpy


class AsyncVideoStream:
    """threaded async wrapper around opencv cv2.VideoCapture with alike interface"""

    def __init__(
        self, src: Union[int, str] = 0, thread_name: str = None, group: Any = None
    ):
        """
        threaded async wrapper around opencv cv2.VideoCapture with alike interface

        :param src:
        :param thread_name:
        """
        self._stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self._stream.read()

        self._stopped = False
        self._thread = Thread(
            target=self.update, name=thread_name, args=(), daemon=True, group=group
        )

    def start(self):
        """
            # start the thread to read frames from the video stream
        :return:
        """

        self._thread.start()
        return self

    def update(self) -> None:
        """Updates the stream"""
        while not self._stopped:  # keep looping infinitely until the thread is stopped
            (
                self.grabbed,
                self.frame,
            ) = self._stream.read()  # otherwise, read the next frame from the stream

    def read(self) -> Tuple[bool, numpy.ndarray]:
        """read latest frame"""
        return self.grabbed, self.frame  # return the frame most recently read

    def stop(self) -> None:
        """Releases the stream"""
        self._stream.release()
        self._stopped = True  # indicate that the thread should be stopped

    # noinspection PyPep8Naming
    def isOpened(self) -> bool:
        """Returns whether the stream is open"""
        return self._stream.isOpened()

    def __call__(self, *args: Sequence[Any], **kwargs: MutableMapping[str, Any]):
        return self.frame

    def __next__(self):
        return self.frame

    def __iter__(self) -> "AsyncVideoStream":
        return self.start()

    def __del__(self):
        self.stop()
        self._thread.join()

    def __enter__(self) -> "AsyncVideoStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


if __name__ == "__main__":
    for a in AsyncVideoStream():
        pass
