#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

TODO: UNIFY PROGRESS BARs interfaces, base on system and python context

           Created on 8/24/22
           """

__all__ = ["ETABar"]

from typing import Iterator

from progress.colors import color  # shit

from draugr.python_utilities import in_ipynb
from progress.bar import Bar  # IT is shit

from warg import passes_kws_to, drop_unused_kws


class ETABar(Bar, Iterator):
    """Progress bar that displays the estimated time of completion.
    TODO: REMOVE THIS PIECE OF SHIT!

    """

    def __next__(self):
        return self.next()

    default_suffix = "%(percent).1f%% - %(eta)ds"
    suffix = default_suffix
    message = "%(eta)"
    bar_prefix = " "
    bar_suffix = " "
    empty_fill = "∙"
    fill = "█"

    def update(self, n: int) -> None:
        """

        :param n:
        :type n:
        :return:
        :rtype:
        """
        rv = self.next(n)
        super().update()
        return rv

    @drop_unused_kws
    @passes_kws_to(Bar.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def writeln(self, line: str):
        """Writes the line to the console.

        Description:
            This method is Jupyter notebook aware, and will do the
            right thing when in that environment as opposed to being
            run from the command line.

        Args:
            line (str): The message to write
        """
        if in_ipynb():
            from IPython.display import clear_output

            clear_output(wait=True)
            self.fill = "#"
            print(line)
        else:
            Bar.writeln(self, line)

    def info(self, text: str):
        """Appends the given information to the progress bar message.

        Args:
            text (str): A status message for the progress bar.
        """
        self.suffix = f"{ETABar.default_suffix} {text}"

    def update(self):
        """ """
        filled_length = int(self.width * self.progress)
        empty_length = self.width - filled_length

        message = self.message % self
        bar = color(self.fill * filled_length, fg=self.color)
        empty = self.empty_fill * empty_length
        suffix = self.suffix % self
        line = "".join([message, self.bar_prefix, bar, empty, self.bar_suffix, suffix])
        self.writeln(line)
