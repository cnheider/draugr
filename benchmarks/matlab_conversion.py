#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-02-2021
           """

import timeit


def conversion_benchmark() -> None:
    """
    :rtype: None
    """
    # Numpy to matlab
    print(
        "\nCompare Numpy to matlab conversion strategy : (a bit long with several matlab.engine opening)"
    )
    setup_np2mat = (
        "import numpy\n"
        "import matlab\n"
        "from draugr.matlab_utilities import matlab_to_ndarray, ndarray_to_matlab\n"
        "import array\n"
        "np_a=numpy.random.uniform(size=(10000))*(.5+0.1236*1j) \n"
    )
    print(
        " > From matlab.double(np_a.tolist()) : "
        + str(
            timeit.timeit(
                "mat_a = matlab.double(np_a.real.tolist())",
                setup=setup_np2mat,
                number=100,
            )
        )
        + " s"
    )
    print(
        " > From ndarray_to_matlab [use pre alloc] : "
        + str(
            timeit.timeit(
                "mat_a = ndarray_to_matlab(np_a)", setup=setup_np2mat, number=100
            )
        )
        + " s"
    )

    # Matlab to numpy
    print("\nCompare matlab to numpy conversition strategy :")
    setup_tolist = (
        "import numpy\n"
        "import matlab\n"
        "from draugr.matlab_utilities import matlab_to_ndarray, ndarray_to_matlab\n"
        "eng = matlab.engine.start_matlab()\n"
        "mrd = eng.rand(matlab.int64([1,10000]),nargout=1)\n"
    )
    print(
        " > From numpy.array : "
        + str(
            timeit.timeit(
                "nprd = numpy.array(mrd,dtype = float) ", setup=setup_tolist, number=100
            )
        )
        + " s"
    )

    print(
        " > From numpy.asarray [use _data] : "
        + str(
            timeit.timeit(
                "nprd = numpy.asarray(mrd._data,dtype = float) ",
                setup=setup_tolist,
                number=100,
            )
        )
        + " s"
    )

    print(
        " > From matlab_to_ndarray [use _data buffer] : "
        + str(
            timeit.timeit(
                "nprd = matlab_to_ndarray(mrd) ", setup=setup_tolist, number=100
            )
        )
        + " s"
    )
    setup_tolist_cpx = (
        "import numpy\n"
        "import matlab\n"
        "from draugr.matlab_utilities import matlab_to_ndarray, ndarray_to_matlab\n"
        "eng = matlab.engine.start_matlab()\n"
        "mrd = eng.log(eng.linspace(-1.,1.,10000.))\n"
    )
    print(
        " > From matlab_to_ndarray [use _real _imag buffer complex] : "
        + str(
            timeit.timeit(
                "nprd = matlab_to_ndarray(mrd) ", setup=setup_tolist_cpx, number=100
            )
        )
        + " s"
    )


if __name__ == "__main__":
    conversion_benchmark()
