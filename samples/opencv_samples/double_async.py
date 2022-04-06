from itertools import cycle

import cv2

from draugr.opencv_utilities import frame_generator
from draugr.opencv_utilities.windows.image import show_image

if __name__ == "__main__":

    def variant0():
        cameras = []
        print("open cam1")
        cameras += ((frame_generator(cv2.VideoCapture(0)), cycle("0")),)
        print("open cam2")
        cameras += ((frame_generator(cv2.VideoCapture(1)), cycle("1")),)
        print("opened")

        a = iter(cycle(cameras))

        class AsyncIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                # await asyncio.sleep(0.1)
                g, i = next(a)
                return next(g), next(i)

        async def run():
            async for image, id in AsyncIterator():
                # await asyncio.sleep(0.1)
                if show_image(image, id, wait=1):
                    break

        import asyncio

        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

    def variant1():
        cameras = []
        print("open cam1")
        # cameras += frame_generator(cv2.VideoCapture(0)),
        print("open cam2")
        cameras += (frame_generator(cv2.VideoCapture(1)),)
        print("opened")

        def show(mat, n=0):
            cv2.imshow(f"{n}", mat)

        while 1:
            list(map(show, map(next, cameras)))
            cv2.waitKey(1)

    def variant1_2():
        cameras = []
        print("open cam1")
        # cameras += frame_generator(cv2.VideoCapture(0)),
        print("open cam2")
        cameras += (frame_generator(cv2.VideoCapture(1)),)
        print("opened")

        def show(mat, n=0):
            cv2.imshow(f"{n}", mat)

        while 1:
            list(
                map(show, *zip(map(next, cameras), range(2)))
            )  # BROKEN, maybe need to run show on main loop
            cv2.waitKey(1)

    def variant2():
        cameras = []
        print("open cam1")
        cameras += (frame_generator(cv2.VideoCapture(0)),)
        print("open cam2")
        cameras += (frame_generator(cv2.VideoCapture(1)),)
        print("opened")

        def show(mat, n):
            cv2.imshow(f"{n}", mat)

        while 1:
            for im, i in zip(map(next, cameras), range(len(cameras))):
                show(im, i)
                cv2.waitKey(1)

    def variant3():
        cameras = []
        print("open cam1")
        cameras += (frame_generator(cv2.VideoCapture(0)),)
        print("open cam2")
        cameras += (frame_generator(cv2.VideoCapture(1)),)
        print("opened")
        from itertools import count

        def show(mat, n):
            cv2.imshow(f"{n}", mat)

        while 1:
            for i, im in count(map(next, cameras)):
                show(im, i)
                cv2.waitKey(1)

    variant3()
