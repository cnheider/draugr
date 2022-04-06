import pygame

__all__ = ["get_screen_resolution"]


def get_screen_resolution():
    info = pygame.display.Info()
    width = info.current_w
    height = info.current_h
    return width, height
