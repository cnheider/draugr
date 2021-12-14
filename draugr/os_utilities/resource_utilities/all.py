#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-05-2021
           """

from draugr.os_utilities.resource_utilities.cpu import worker_cores_available
from draugr.os_utilities.resource_utilities.ram import num_instance_no_paging


def get_num_instances(expected_size: int = 1024) -> int:
    """

    :param expected_size:
    :return:
    """
    return min(worker_cores_available(), num_instance_no_paging(expected_size))


if __name__ == "__main__":
    print(get_num_instances())
