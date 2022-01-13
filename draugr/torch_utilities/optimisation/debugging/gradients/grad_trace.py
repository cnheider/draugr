#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/07/2020
           """

__all__ = ["register_bad_grad_hooks", "print_grad_trace"]

from typing import Any

import torch


def register_bad_grad_hooks(var: Any) -> callable:
    """

    :param var:
    :type var:
    :return:
    :rtype:"""

    def iter_graph(root, callback):
        """

        :param root:
        :type root:
        :param callback:
        :type callback:"""
        queue = [root]
        seen = set()
        while queue:
            fn = queue.pop()
            if fn in seen:
                continue
            seen.add(fn)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append(next_fn)
            callback(fn)

    fn_dict = {}

    def hook_callback(fn):
        """

        :param fn:
        :type fn:"""

        def register_grad(grad_input, grad_output):
            """

            :param grad_input:
            :type grad_input:
            :param grad_output:
            :type grad_output:"""
            fn_dict[fn] = grad_input

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_callback)

    def is_bad_grad(grad_output):
        """

        :param grad_output:
        :type grad_output:
        :return:
        :rtype:"""
        if grad_output is None:
            return False
        return torch.isnan(grad_output).any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        """

        :return:
        :rtype:"""
        from graphviz import Digraph

        node_attr = dict(
            style="filled",
            shape="box",
            align="left",
            fontsize="12",
            ranksep="0.1",
            height="0.2",
        )
        dot_ = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            """

            :param size:
            :type size:
            :return:
            :rtype:"""
            return f'({", ".join(map(str, size))})'

        def build_graph(fn):
            """

            :param fn:
            :type fn:"""
            if hasattr(fn, "variable"):  # if GradAccumulator
                u = fn.variable
                node_name = "Variable\n " + size_to_str(u.size())
                dot_.node(str(id(u)), node_name, fillcolor="lightblue")
            else:
                assert fn in fn_dict, fn
                fillcolor = "white"
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = "red"
                dot_.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, "variable", next_fn))
                    dot_.edge(str(next_id), str(id(fn)))

        iter_graph(var.grad_fn, build_graph)

        return dot_

    return make_dot


def print_grad_trace(var_grad_fn):
    """ """
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError as e:
                print_grad_trace(n[0])


if __name__ == "__main__":

    def asdifiejsf() -> None:
        """
        :rtype: None
        """
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)

        z = x / (y * 0)
        z = z.sum() * 2
        get_dot = register_bad_grad_hooks(z)
        z.backward()
        dot = get_dot()
        # dot.save('tmp.dot') # to get .dot
        # dot.render('tmp') # to get SVG

    asdifiejsf()
