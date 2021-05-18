#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11-03-2021
           """

__all__ = [
    "round_tight_bbox",
    "opaque_round_tight_bbox",
    "semi_opaque_round_tight_bbox",
    "lt_ann_transform",
    "lb_ann_transform",
    "rt_ann_transform",
    "rb_ann_transform",
    "arc_arrow",
]

round_tight_bbox = dict(
    boxstyle="round,pad=.2,rounding_size=.2", fc="1.0", alpha=0.5, ec="0.9"
)
opaque_round_tight_bbox = round_tight_bbox.copy()
opaque_round_tight_bbox["alpha"] = 1.0
semi_opaque_round_tight_bbox = round_tight_bbox.copy()
semi_opaque_round_tight_bbox["alpha"] = 0.75
lt_ann_transform = dict(xytext=(1.4, -0.8), ha="left", va="top", rotation=-35)
lb_ann_transform = dict(xytext=(1.4, 0.8), ha="left", va="bottom", rotation=35)
rt_ann_transform = dict(xytext=(-1.4, -0.8), ha="right", va="top", rotation=35)
rb_ann_transform = dict(xytext=(-1.4, 0.8), ha="right", va="bottom", rotation=-35)
arc_arrow = (dict(arrowstyle="->", connectionstyle="arc3"),)
