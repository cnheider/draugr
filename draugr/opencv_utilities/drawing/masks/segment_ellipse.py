if __name__ == "__main__":

    def aisjhd():

        from matplotlib import pyplot
        import cv2
        from draugr.opencv_utilities import LineTypeEnum
        import numpy
        import random

        ANGLE_DELTA = 360 // 8

        img = numpy.zeros((700, 700, 3), numpy.uint8)
        img[::] = 255

        for size in range(300, 0, -100):
            for angle in range(0, 360, ANGLE_DELTA):
                r = random.randint(0, 256)
                g = random.randint(0, 256)
                b = random.randint(0, 256)
                cv2.ellipse(
                    img,
                    (350, 350),
                    (size, size),
                    0,
                    angle,
                    angle + ANGLE_DELTA,
                    (r, g, b),
                    LineTypeEnum.filled.value,
                )

        pyplot.gcf().set_size_inches((8, 8))
        pyplot.imshow(img)
        pyplot.show()

    aisjhd()
