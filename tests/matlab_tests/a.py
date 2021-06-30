#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-02-2021
           """

import matlab
import numpy

from draugr.matlab_utilities import dict_to_sparse, matlab_to_ndarray, ndarray_to_matlab


def asiduj_test():
    """
    Test of the module

    It runs the doctest and create other tests with matlab engine calls."""
    import scipy.linalg as spl

    print("Run matlab engine...")
    if len(matlab.engine.find_matlab()) == 0:
        # si aucune session share, run
        eng = matlab.engine.start_matlab()
    else:
        # connect to a session
        eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
        print("connected...")

    print("Further tests....\n")

    # create matlab data
    # ------------------------------------------------------------------------
    mf = eng.rand(3)
    mc = matlab.double([[1 + 1j, 0.3, 1j], [1.2j - 1, 0, 1 + 1j]], is_complex=True)
    mi64 = matlab.int64([1, 2, 3])
    mi8 = matlab.int8([1, 2, 3])
    mb = matlab.logical([True, True, False])

    # Test conversion from matlab to numpy
    # ------------------------------------------------------------------------
    npf = matlab_to_ndarray(mf)  # no copy, if mf is changed, npf change!
    npc = matlab_to_ndarray(mc)  # still copy for complex (only)
    npi64 = matlab_to_ndarray(mi64)
    npi8 = matlab_to_ndarray(mi8)
    npb = matlab_to_ndarray(mb)

    # Test conversion from numpy to matlab
    # ------------------------------------------------------------------------
    npi = numpy.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=numpy.int64, order="F"
    )
    mi = ndarray_to_matlab(npi)
    mc2 = ndarray_to_matlab(npc)
    mf2 = ndarray_to_matlab(npf)  # copy, because npf has 'F' order (comes from mlarray)
    mi64_2 = ndarray_to_matlab(npi)
    mb2 = ndarray_to_matlab(npb)

    # test orientation in the matlab workspace
    # ------------------------------------------------------------------------
    eng.workspace["mi"] = mi64_2
    eng.workspace["mc2"] = mc2

    # check results
    # ------------------------------------------------------------------------
    npcc = numpy.array(
        [
            [1.0, 1.1 + 1j],
            [
                1.12 + 0.13j,
                22.1,
            ],
        ],
        dtype=numpy.complex,
    )  # assume C
    mcc = ndarray_to_matlab(npcc)
    npcc_inv = spl.inv(npcc)
    mcc_inv = eng.inv(mcc)
    print("Are the inverse of matrix equal ?")
    print(mcc_inv)
    print(npcc_inv)

    #    # no copy check
    #    # ------------------------------------------------------------------------
    #    # complex
    #
    #    npcc[0,0]=0.25
    #    print("Are the data reuse ?", ", OWNDATA =", mcc._real.flags.owndata,
    #          "same base =", mcc._real.base is npcc,
    #          ', If one is modified, the other is modified =', mcc._real[0]==npcc[0,0])
    #

    # test sparse matrix requiert Recast4py.m
    K1, K2 = eng.sptest(3.0, nargout=2)
    Ksp1 = dict_to_sparse(K1)
    Ksp2 = dict_to_sparse(K2)


def doctest_test():
    """run test procedure with doctest"""
    import doctest

    # invoke the testmod function to run tests contained in docstring
    stats = doctest.testmod()
    print(stats)
    return stats
