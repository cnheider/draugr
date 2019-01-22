#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

import six

COLORS = dict(
    # gray='30', #Black
    red='31',
    green='32',
    yellow='33',
    blue='34',
    magenta='35',
    cyan='36',
    white='37',
    crimson='38'
    )

DECORATIONS = dict(
    end='0',
    bold='1',
    dim='2',
    italic='3',
    underline='4',
    underline_end='24',  # '4:0',
    double_underline='21',  # '4:2'
    # double_underline_end='24',  # '4:0'
    curly_underline='4:3',
    blink='5',
    reverse_colors='7',
    invisible='8',  # still copyable
    strikethrough='9',
    overline='53',
    hyperlink='8;;'
    )


def generate_style(obj=None, *,
                   color='white',
                   bold=False,
                   highlight=False,
                   underline=False,
                   italic=False):
  attributes = []

  if color in COLORS:
    num = int(COLORS[color])
  else:
    num = int(COLORS['white'])

  if highlight:
    num += 10

  attributes.append(six.u(f'{num}'))

  if bold:
    attributes.append(six.u(f'{DECORATIONS["bold"]}'))

  if underline:
    attributes.append(six.u(f'{DECORATIONS["underline"]}'))

  if italic:
    attributes.append(six.u(f'{DECORATIONS["italic"]}'))

  end = DECORATIONS['end']

  attributes_joined = six.u(';').join(attributes)

  print_style = PrintStyle(attributes_joined, end)

  if obj:
    return print_style(obj)
  else:
    return print_style


class PrintStyle(object):
  def __init__(self, attributes_joined, end):
    self._attributes_joined = attributes_joined
    self._end = end

  def __call__(self, obj, *args, **kwargs):
    intermediate_repr = f'\x1b[{self._attributes_joined}m{obj}\x1b[{self._end}m'
    string = six.u(intermediate_repr)
    return string

