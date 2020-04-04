from itertools import combinations, product

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pynput import keyboard

from draugr import sprint

fig = pyplot.figure()
ax = Axes3D(fig, proj_type="ortho")  # 'persp'
# ax.set_aspect("equal")
# ax.autoscale(True)
ax.autoscale_view(None, False, False, False)


def sample_unit_circle(p=numpy.inf, samples=100, color="r", marker="^", size=3):
    """
    plot some 2D vectors with p-norm < 1
  """
    for i in range(samples):
        xyz = numpy.array(
            [
                [numpy.random.rand() * 2 - 1],
                [numpy.random.rand() * 2 - 1],
                [numpy.random.rand() * 2 - 1],
            ]
        )
        if numpy.linalg.norm(xyz, p) < 1:
            ax.scatter(*xyz, c=color, marker=marker, s=size, depthshade=True)


def remove_decoration():
    transparent = (1.0, 1.0, 1.0, 0.0)

    ax.w_xaxis.set_pane_color(transparent)
    ax.w_yaxis.set_pane_color(transparent)
    ax.w_zaxis.set_pane_color(transparent)

    ax.w_xaxis.line.set_color(transparent)
    ax.w_yaxis.line.set_color(transparent)
    ax.w_zaxis.line.set_color(transparent)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


remove_decoration()


# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')


def draw_l1(color="r", r=[-1, 0, 1]):
    cartesian_prod = numpy.array(list(product(r, r, r)))
    line_segments = combinations(cartesian_prod, 2)
    for p1, p2 in line_segments:
        diff = p1 - p2
        distance = numpy.sqrt(numpy.sum(diff ** 2))
        if (
            distance == numpy.sqrt(2)
            and numpy.sum(numpy.abs(p1) + numpy.abs(p2)) == 2
            and numpy.sum(numpy.abs(p1)) == 1
            and numpy.sum(numpy.abs(p2)) == 1
        ):
            ax.plot3D(*zip(p1, p2), color=color)


def draw_l2(color="g", resolution=3 ** 2):
    # u, v = numpy.mgrid[0:2 * numpy.pi:20j, 0:numpy.pi:10j]
    # x = numpy.cos(u) * numpy.sin(v)
    # y = numpy.sin(u) * numpy.sin(v)
    # z = numpy.cos(v)

    u = numpy.linspace(0, 2 * numpy.pi, resolution)
    v = numpy.linspace(0, numpy.pi, resolution)
    x = numpy.outer(numpy.cos(u), numpy.sin(v))
    y = numpy.outer(numpy.sin(u), numpy.sin(v))
    z = numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
    ax.plot_wireframe(x, y, z, color=color)


def draw_inf(color="b", r=[-1, 1]):
    cartesian_prod = numpy.array(list(product(r, r, r)))
    line_segments = combinations(cartesian_prod, 2)
    for s, e in line_segments:
        if numpy.sum(numpy.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color=color)


def rotate(pitch=30):
    for angle in range(0, 360):
        ax.view_init(pitch, angle)
        pyplot.draw()
        pyplot.pause(0.01)


if __name__ == "__main__":

    def main():

        draw_l1()
        draw_l2()
        draw_inf()
        # sample_unit_circle(1,color='r')
        # sample_unit_circle(2,color='g')
        # sample_unit_circle(color='b')

        COMBINATIONS = [
            {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="s")},
            {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char="S")},
        ]

        CALLBACKS = []
        current = set()

        def add_early_stopping_key_combination(callback, key="ctrl+c"):
            CALLBACKS.append(callback)
            sprint(
                f"\n\nPress any of:\n{COMBINATIONS}\n for early stopping\n",
                color="red",
                bold=True,
                highlight=True,
            )
            print("")
            return keyboard.Listener(on_press=on_press, on_release=on_release)

        def on_press(key):
            if any([key in COMBO for COMBO in COMBINATIONS]):
                current.add(key)
                if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
                    for callback in CALLBACKS:
                        callback()

        def on_release(key):
            if any([key in COMBO for COMBO in COMBINATIONS]):
                current.remove(key)

        listener = add_early_stopping_key_combination(exit)

        listener.start()
        try:
            rotate(10)
        finally:
            listener.stop()

    main()
