import numpy

NUMPY_TO_QGIS_TYPE_MAPPING = {  # Numpy type         C type      Description
    numpy.bool_: "QgisDataTypeEnum.uint8",  # bool Boolean (True or False) stored as a byte
    numpy.int_: "QgisDataTypeEnum.int32",  # long Platform-defined
    numpy.complex_: "QgisDataTypeEnum.unknown",
    numpy.float_: "QgisDataTypeEnum.unknown",
    numpy.uint8: "uint8_scsac",
}


def stest(arr):
    if isinstance(arr, numpy.ndarray) and numpy.issubdtype(arr.dtype, numpy.floating):
        ...


if __name__ == "__main__":
    print(NUMPY_TO_QGIS_TYPE_MAPPING[numpy.ones((9, 9), dtype=numpy.uint8).dtype.type])
