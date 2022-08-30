#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

TODO: UNIFY PROGRESS BARs interfaces, base on system and python context

           Created on 8/24/22
           """

__all__ = ["ETABar"]

from draugr.python_utilities import in_ipynb
from progress.bar import Bar


class ETABar(Bar):
    """Progress bar that displays the estimated time of completion."""

    suffix = "%(percent).1f%% - %(eta)ds"
    bar_prefix = " "
    bar_suffix = " "
    empty_fill = "∙"
    fill = "█"

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
        self.suffix = f"%(percent).1f%% - %(eta)ds {text}"
