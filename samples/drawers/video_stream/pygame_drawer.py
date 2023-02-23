#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/9/22
           """

__all__ = []

import time

import numpy
import pygame
from pygame import camera


def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.surfarray.make_surface(image)
    # return pygame.image.frombuffer(image.tostring(), image.size, "RGB")


def surface_to_numpy(surface) -> numpy.ndarray:
    """Convert a pygame surface into string"""
    return pygame.surfarray.pixels3d(surface)


if __name__ == "__main__":
    # Screen settings
    SCREEN = [640, 360]

    # Set game screen
    screen = pygame.display.set_mode(SCREEN)

    pygame.init()  # Initialize pygame
    camera.init()  # Initialize camera
    cameras = pygame.camera.list_cameras()
    # Load camera source then start
    cam = camera.Camera(cameras[0], SCREEN)
    cam.start()

    while 1:  # Ze loop
        time.sleep(1 / 120)  # 60 frames per second

        image = cam.get_image()  # Get current webcam image

        cv_image = surface_to_numpy(image)  # Create cv image from pygame image

        screen.fill([0, 0, 0])  # Blank fill the screen

        screen.blit(cvimage_to_pygame(cv_image), (0, 0))  # Load new image on screen

        pygame.display.update()  # Update pygame display
