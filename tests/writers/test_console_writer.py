#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from draugr.writers import ConsoleWriter

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


@pytest.mark.parametrize(
    ["tag", "val", "step"],
    (("signal", 0, 0), ("signal", 20, 1), ("signal", -1, 6)),
    ids=["signal_first", "signal_second", "signal_sixth"],
)
def test_valid_scalars(tag, val, step):
    with ConsoleWriter() as w:
        w.scalar(tag, val, step)


@pytest.mark.parametrize(
    ["tag", "val", "step"],
    (("signal", "", 0), ("signal", None, 1), ("signal", object(), 6)),
    ids=["str_scalar", "None_scalar", "object_scalar"],
)
def test_invalid_val_type_scalars(tag, val, step):
    try:
        with ConsoleWriter() as w:
            w.scalar(tag, val, step)
        assert False
    except Exception as e:
        assert True


@pytest.mark.parametrize(
    ["tag", "val", "step"],
    ((1, 0, 0), (None, 20, 1), (object(), -1, 6)),
    ids=["numeral_tag", "None_tag", "object_tag"],
)
def test_invalid_tag_scalars(tag, val, step):
    try:
        with ConsoleWriter() as w:
            w.scalar(tag, val, step)
        assert False
    except Exception as e:
        print(e)
        assert True


@pytest.mark.parametrize(
    ["tag", "val", "step"],
    (("signal", 0, ""), ("signal", 20, None), ("tag1", -0, object())),
    ids=["str_step", "None_step", "object_step"],
)
def test_invalid_step_type_scalars(tag, val, step):
    try:
        with ConsoleWriter() as w:
            w.scalar(tag, val, step)
        assert False
    except Exception as e:
        print(e)
        assert True
