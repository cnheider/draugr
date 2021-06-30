#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28-03-2021
           """

import torch

from draugr import PROJECT_APP_PATH
from draugr.torch_utilities import (
    TensorBoardPytorchWriter,
)

if __name__ == "__main__":

    def main():
        """
        TODO:!!!!"""

        with TensorBoardPytorchWriter(
            PROJECT_APP_PATH.user_log / "Tests" / "Writers"
        ) as writer:
            vertices_tensor = torch.as_tensor(
                [
                    [1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, -1],
                ],
                dtype=torch.float,
            ).unsqueeze(0)
            colors_tensor = torch.as_tensor(
                [
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 0, 255],
                ],
                dtype=torch.int,
            ).unsqueeze(0)
            faces_tensor = torch.as_tensor(
                [
                    [0, 2, 3],
                    [0, 3, 1],
                    [0, 1, 2],
                    [1, 3, 2],
                ],
                dtype=torch.int,
            ).unsqueeze(0)

            writer.mesh(
                "my_mesh", vertices_tensor, colors=colors_tensor, faces=faces_tensor
            )

    main()
