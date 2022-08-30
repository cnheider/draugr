#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/30/22
           """

__all__ = ["Matrix"]


class Matrix(list):
    def __matmul__(self, B):  # Matrix multiplication A @ B
        A = self

        N = len(A)
        M = len(A[0])
        P = len(B[0])

        result = []
        for i in range(N):
            row = [0] * P
            result.append(row)

        for i in range(N):
            for j in range(P):
                for k in range(M):
                    result[i][j] += A[i][k] * B[k][j]
        return result


if __name__ == "__main__":

    def hasdhuh():
        # Example
        A = Matrix([[2, 0], [1, 9]])
        B = Matrix([[3, 9], [4, 7]])
        print(A @ B)

    hasdhuh()
