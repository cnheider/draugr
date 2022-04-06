#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           """

__all__ = []

import ctypes
import ctypes.util

try:  # macOS Big Sur+ use "a built-in dynamic linker cache of all system-provided libraries"

    appkit = ctypes.cdll.LoadLibrary("AppKit.framework/AppKit")
    objc = ctypes.cdll.LoadLibrary("libobjc.dylib")
except OSError:  # revert to full path for older OS versions and hardened programs

    appkit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

void_p = ctypes.c_void_p
ull = ctypes.c_uint64

objc.objc_getClass.restype = void_p
objc.sel_registerName.restype = void_p

# See https://docs.python.org/3/library/ctypes.html#function-prototypes for arguments description
MSGPROTOTYPE = ctypes.CFUNCTYPE(void_p, void_p, void_p, void_p)
msg = MSGPROTOTYPE(
    ("objc_msgSend", objc), ((1, "", None), (1, "", None), (1, "", None))
)


def _utf8(s):
    if not isinstance(s, bytes):
        s = s.encode("utf8")
    return s


def n(name):
    return objc.sel_registerName(_utf8(name))


def C(classname):
    return objc.objc_getClass(_utf8(classname))


def is_dark_mode_active():
    NSAutoreleasePool = objc.objc_getClass("NSAutoreleasePool")
    pool = msg(NSAutoreleasePool, n("alloc"))
    pool = msg(pool, n("init"))

    NSUserDefaults = C("NSUserDefaults")
    stdUserDef = msg(NSUserDefaults, n("standardUserDefaults"))

    NSString = C("NSString")

    key = msg(NSString, n("stringWithUTF8String:"), _utf8("AppleInterfaceStyle"))
    appearanceNS = msg(stdUserDef, n("stringForKey:"), void_p(key))
    appearanceC = msg(appearanceNS, n("UTF8String"))

    if appearanceC is not None:
        out = ctypes.string_at(appearanceC)
    else:
        out = None

    msg(pool, n("release"))

    if out is not None:
        if out.decode("utf-8") == "Dark":
            return True

    return False


if __name__ == "__main__":
    print(is_dark_mode_active())
