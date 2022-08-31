import queue
import threading
from typing import Sequence, MutableMapping

__all__ = ["Skipper"]


class Skipper:
    """description"""

    def __init__(self, fun):
        self.Q = queue.Queue(1)
        self.fun = fun
        self._stop = False
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        """description"""
        while True:
            if self._stop:
                break
            (args, kwargs) = self.Q.get()
            if self._stop:
                break
            self.fun(*args, **kwargs)

    def stop(self):
        """description"""
        self._stop = True
        try:
            self.Q.put_nowait((None, None))
        except:
            pass
        self.thread.join()

    def __call__(self, *args: Sequence, **kwargs: MutableMapping):
        while not self.Q.empty():
            self.Q.get_nowait()
        self.Q.put((args, kwargs))


if __name__ == "__main__":

    def yhhgsady():
        """description"""
        import time

        def worker(wid, wait_time_sec=1):
            """

            :param wid:
            :type wid:
            """
            print(f"Worker {wid} Start")
            time.sleep(wait_time_sec)
            print(f"Worker {wid} END")

        S = Skipper(worker)
        S(1)
        time.sleep(0.1)
        print("next_job")
        S(2)
        time.sleep(0.1)
        print("next_job")
        S(3)
        time.sleep(4)
        print("next_job")
        S(4)
        time.sleep(0.2)
        print("next_job")
        S(5)
        time.sleep(0.2)
        print("next_job")
        S(6)
        time.sleep(0.2)
        print("next_job")
        S(7)
        time.sleep(4)
        print("Deadline")
        S.stop()

    def yhhgsad2y():
        """description"""
        import time

        def worker(wid, wait_time_sec=1):
            """

            :param wid:
            :type wid:
            """
            print(f"Worker {wid} Start")
            time.sleep(wait_time_sec)
            print(f"Worker {wid} END")

        S = Skipper(worker)
        S(1)
        time.sleep(4)
        print("Deadline")
        S.stop()

    yhhgsady()
    # yhhgsad2y()
