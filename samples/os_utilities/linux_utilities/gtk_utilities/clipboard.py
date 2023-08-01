#!/usr/bin/python3

from gi.repository import Gtk, Gdk


class Hello(Gtk.Window):
    def __init__(self):
        super(Hello, self).__init__()
        clipboard_ = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard_.set_text("hello world", -1)
        Gtk.main_quit()


def main():
    Hello()
    Gtk.main()


if __name__ == "__main__":
    main()
