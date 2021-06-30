#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-09-2020
           """

from typing import Any

import matlab
import numpy
from scipy import sparse

__all__ = ["ndarray_to_matlab", "matlab_to_ndarray", "dict_to_sparse"]

from draugr.matlab_utilities.matlab_conversion_utilities import (
    get_strides_c,
    get_strides_f,
)


def matlab_to_ndarray(matlab_array: Any) -> numpy.ndarray:
    """
    Convert matlab mlarray to numpy array
    The conversion is realised without copy for real data thank to the frombuffer
    protocol. The np ndarray type depends on the type of matlab data.

    Paramerters
    -----------
    matlab_array : mlarray
    the matlab array to convert

    Returns
    -------
    numpy_array : numpy array
    the converted numpy array

    Examples
    --------
    Complex 2D array
    >>> mc=matlab.double([[(1+1j),(0.3+0j),1j],[(-1+1.2j),0j,(1+1j)]], is_complex=True)
    >>> matlab_to_ndarray(mc)
    array([[ 1. +1.j ,  0.3+0.j ,  0. +1.j ],
       [-1. +1.2j,  0. +0.j ,  1. +1.j ]])
    >>> m3 = matlab.double([[[1.0,10.0],[2.0,20.0],[3.0,30.0]],[[4.0,40.0],[5.0,50.0],[6.0,60.0]]])
    >>> np3 = matlab_to_ndarray(m3)
    >>> np3[...,0]
    array([[1., 2., 3.],
       [4., 5., 6.]])
    >>> np3[...,1]
    array([[10., 20., 30.],
       [40., 50., 60.]])
    >>> np3.flags['OWNDATA'] # no copy
    False
    References
    ----------
    https://stackoverflow.com/questions/34155829/how-to-efficiently-convert-matlab-engine-arrays-to-numpy-ndarray/34155926


    for real:
    ma._data.itemsize # get item size
    ma._data.typecode # get item type
    ma._python_type
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html#numpy.dtype
    https://docs.python.org/3/library/array.html

    numpy kind

    A character code (one of ‘biufcmMOSUV’) identifying the general kind of data.
    b 	boolean
    i 	signed integer
    u 	unsigned integer
    f 	floating-point
    c 	complex floating-point
    m 	timedelta
    M 	datetime
    O 	object
    S 	(byte-)string
    U 	Unicode
    V 	void


    :param matlab_array:
    :return:"""

    if "mlarray" not in str(type(matlab_array)):
        raise TypeError(f"Expected matlab.mlarray. Got {type(matlab_array)}")

    if matlab_array._is_complex:
        numpy_type = "f8"
        numpy_array = numpy.empty(matlab_array.size, dtype=complex)
        numpy_array.real = numpy.frombuffer(
            matlab_array._real, dtype=numpy_type
        ).reshape(matlab_array.size, order="F")
        numpy_array.imag = numpy.frombuffer(
            matlab_array._imag, dtype=numpy_type
        ).reshape(matlab_array.size, order="F")
        return numpy_array

    # tuple that define type
    if isinstance(matlab_array._data, numpy.ndarray):
        return matlab_array._data.reshape(matlab_array.size)

    matlab_type = (
        matlab_array._data.typecode,
        matlab_array._data.itemsize,
    )  # use test to define the type, note not comprehensive a few types are missing, like uint!!
    if matlab_type == ("d", 8):  # double
        numpy_type = "f8"
    elif matlab_type == ("B", 1):  # logical is given as a int
        numpy_type = "bool"
    elif matlab_type == ("b", 1):  # int8
        numpy_type = "i1"
    elif matlab_type == ("i", 4):
        numpy_type = "i4"
    elif matlab_type == ("i", 8):  # int64
        numpy_type = "i8"
    elif matlab_type == ("l", 8):  # int64
        numpy_type = "i8"
    else:
        numpy_type = "f8"  # default

    return numpy.frombuffer(matlab_array._data, dtype=numpy_type).reshape(
        matlab_array.size, order="F"
    )  # no copy with the buffer


def ndarray_to_matlab(numpy_array):
    """Conversion of a numpy array to matlab mlarray

    The conversion is realised without copy for real data. First an empty initialization is realized.
    Then the numpy array is affected to the _data field. Thus the data field is not really an
    array.array but a numpy array. Matlab doesn't see anything...
    For complex data, the strides seems to not work properly with matlab.double.

    Paramerters
    -----------
    numpy_array : numpy array
      the array to convert
    Returns
    -------
    matlab_array : mlarray
      the converted array that can be passe to matlab
    Examples
    --------
    >>> npi=numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=numpy.int64,order='C')
    >>> ndarray_to_matlab(npi)
    matlab.int64([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    >>> npif=numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=numpy.int64,order='F')
    >>> ndarray_to_matlab(npif)
    matlab.int64([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

    >>> npcf=numpy.array([[1,2+0.2j,3],[4,5,6],[7,8,9],[10+0.1j,11,12]],dtype=numpy.complex,order='F')
    >>> ndarray_to_matlab(npcf)
    matlab.double([[(1+0j),(2+0.2j),(3+0j)],[(4+0j),(5+0j),(6+0j)],[(7+0j),(8+0j),(9+0j)],[(10+0.1j),(11+0j),(12+0j)]], is_complex=True)

    References
    -----------
    https://scipy-lectures.org/advanced/advanced_numpy/ (strides)
    """

    if "ndarray" not in str(type(numpy_array)):  # check numpy
        raise TypeError(f"Expect  numpy.ndarray. Got {type(numpy_array)}")

    shape = numpy_array.shape  # get shape

    num_elements = numpy.prod(shape)  # number of elements

    if numpy_array.flags.f_contiguous:  # compute strides (real case)
        strides = get_strides_f(shape)
        order = "F"
    else:
        strides = get_strides_c(shape)
        order = "C"

    if numpy.iscomplexobj(numpy_array):  # complex case

        matlab_array = matlab.double(
            initializer=None, size=(1, num_elements), is_complex=True
        )  # create empty matlab.mlarray
        # associate the data
        """
# associate the data (no copy), works on 2D array only...
matlab_array._real=numpy_array.ravel(order=order) # allow to map real and imaginary part continuously!
"""
        cpx = numpy_array.ravel(order="F")  # copy except for fortran like array
        matlab_array._real = (
            cpx.real.ravel()
        )  # second ravel to correct the strides 18->8
        matlab_array._imag = cpx.imag.ravel()
        matlab_array.reshape(shape)
        # matlab_array._strides=strides

    else:  # real case     # create empty matlab.mlarray
        if numpy_array.dtype == numpy.float64:
            matlab_array = matlab.double(
                initializer=None, size=(1, num_elements), is_complex=False
            )
        elif numpy_array.dtype == numpy.int64:
            matlab_array = matlab.int64(initializer=None, size=(1, num_elements))
        elif numpy_array.dtype == numpy.bool:
            matlab_array = matlab.logical(initializer=None, size=(1, num_elements))
        else:
            raise TypeError(f"Type {numpy_array.dtype} is missing")

        matlab_array._data = numpy_array.ravel(order=order)  # associate the data
        # print(matlab_array._data.flags,matlab_array._data,'\n') # control owner

        matlab_array.reshape(shape)  # back to original shape
        if (
            len(shape) > 1
        ):  # array strides are in number of cell (numpy strides are in bytes)
            # if len(shape)==1 no need to change. Format pb because _stride expect (1,1) and stride = (1,)
            matlab_array._strides = (
                strides  # change stride (matlab use 'F' order ie [nc,1] )
            )

    return matlab_array


def dict_to_sparse(K):
    """Create a scipy sparse CSR matrix from dictionary

    Parameters
    -----------
    K : dictionary K['i'], K['j'] and K['s']
      The sparse matrix in the coo format. K['i'], K['j'] contains the row
      and column index (int64) and the values K['s']  (double or complex)

    Returns
    -------
    Ksp :  sparse.csr_matrix
      The converted sparse matrix. csr is faster for computation."""
    # get shape
    shape = tuple(K["shape"]._data)
    # -1 because matlab index start to 1
    if len(K["i"]._data) > 1:
        ksp = sparse.coo_matrix(
            (
                matlab_to_ndarray(K["s"]).ravel(),
                (
                    matlab_to_ndarray(K["i"]).ravel() - 1,
                    matlab_to_ndarray(K["j"]).ravel() - 1,
                ),
            ),
            shape=shape,
        ).tocsr()
    else:
        raise TypeError(
            "The sparse matrix contains just one element and matlab returns just scalar..."
        )

    return ksp


if __name__ == "__main__":
    a = numpy.random.random((2, 1, 2))
    b = ndarray_to_matlab(a)
    c = matlab_to_ndarray(b)
    print(a.shape, b.size, c.shape)

    print(a)
    print(b)
    print(c)
