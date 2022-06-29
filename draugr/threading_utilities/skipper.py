import queue
import threading

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

    def __call__(self, *args, **kwargs):
        while not self.Q.empty():
            self.Q.get_nowait()
        self.Q.put((args, kwargs))


if __name__ == "__main__":

    def yhhgsady():
        """description"""
        import time

        def worker(wid):
            """

            :param wid:
            :type wid:
            """
            print(f"Worker {wid} Start")
            time.sleep(1)
            print(f"Worker {wid} END")

        S = Skipper(worker)
        S(1)
        time.sleep(0.1)
        S(2)
        time.sleep(0.1)
        S(3)
        time.sleep(4)
        S(4)
        time.sleep(0.2)
        S(5)
        time.sleep(0.2)
        S(6)
        time.sleep(0.2)
        S(7)
        time.sleep(4)
        S.stop()

    def yhhgsad2y():
        """description"""
        import time

        def worker(wid):
            """

            :param wid:
            :type wid:
            """
            print(f"Worker {wid} Start")
            time.sleep(1)
            print(f"Worker {wid} END")

        S = Skipper(worker)
        S(1)
        time.sleep(4)
        S.stop()

    # yhhgsady()
    yhhgsad2y()
