#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 31-10-2020
           """


from turtle import *

from warg import PlaybackShell, str_to_tuple


class TurtleShell(PlaybackShell):
    intro = "Welcome to the turtle shell.   Type help or ? to list commands.\n"
    prompt = "(turtle) "

    # ----- basic turtle commands -----
    def do_forward(self, arg):
        """Move the turtle forward by the specified distance:  FORWARD 10"""
        forward(*str_to_tuple(arg))

    def do_right(self, arg):
        """Turn turtle right by given number of degrees:  RIGHT 20"""
        right(*str_to_tuple(arg))

    def do_left(self, arg):
        """Turn turtle left by given number of degrees:  LEFT 90"""
        left(*str_to_tuple(arg))

    def do_goto(self, arg):
        """Move turtle to an absolute position with changing orientation.  GOTO 100 200"""
        goto(*str_to_tuple(arg))

    def do_home(self, arg):
        """Return turtle to the home position:  HOME"""
        home()

    def do_circle(self, arg):
        """Draw circle with given radius an options extent and steps:  CIRCLE 50"""
        circle(*str_to_tuple(arg))

    def do_position(self, arg):
        """Print the current turtle position:  POSITION"""
        print(f"Current position is {position()}\n")

    def do_heading(self, arg):
        """Print the current turtle heading in degrees:  HEADING"""
        print(f"Current heading is {heading()}\n")

    def do_color(self, arg):
        """Set the color:  COLOR BLUE"""
        color(arg.lower())

    def do_undo(self, arg):
        """Undo (repeatedly) the last turtle action(s):  UNDO"""

    def do_reset(self, arg):
        """Clear the screen and return turtle to center:  RESET"""
        reset()

    def do_bye(self, arg):
        """Stop recording, close the turtle window, and exit:  BYE"""
        print("Thank you for using Turtle")
        self.close()
        bye()
        return True


if __name__ == "__main__":
    TurtleShell().cmdloop()
