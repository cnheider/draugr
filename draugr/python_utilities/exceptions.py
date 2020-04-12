#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02/03/2020
           """

__all__ = ["NoData", "IncompatiblePackageVersions"]


class NoData(Exception):
    """

"""

    def __init__(self, msg="No Data Available"):
        Exception.__init__(self, msg)


class IncompatiblePackageVersions(Exception):
    """

  """

    def __init__(self, *packages, **versions):
        str_o = ", "
        str_l = []

        for p in packages:
            if isinstance(p, str):
                s = f"({p},"
                if p in versions:
                    s += f"{versions[p]}"
                else:
                    s += f"NotSpecified"
                s += ")"
                str_l.append(s)
            else:
                str_l.append(f"{p.__name__, p.__version__}")

        Exception.__init__(self, f"Packages {str_o.join(str_l)} are not compatible")


if __name__ == "__main__":

    def main():
        raise IncompatiblePackageVersions(
            "numpy", "scipy", "Something", numpy="0.0.1", scipy="0.0.2"
        )

    def main2():
        import numpy
        import scipy

        raise IncompatiblePackageVersions(numpy, scipy)

    main2()
