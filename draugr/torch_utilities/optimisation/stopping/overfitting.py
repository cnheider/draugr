import contextlib

from warg import sink, Number

__all__ = ["ImprovementDetector", "OverfitDetector"]


class ImprovementDetector(contextlib.AbstractContextManager):
    def __init__(
        self,
        patience: int,
        writer: callable = print,
        minimization: bool = True,
        callback: callable = None,
    ):
        """
        NOTE: equality is not checked, only greater or less than

        :param patience:
        :type patience:
        :param writer:
        :type writer:
        :param minimization:
        :type minimization:
        """
        self._patience = patience
        self._writer = writer
        self._count = 0
        self._minimization = minimization  # as opposed to maximization
        self._last_value = None
        self._callback = (lambda: True) if callback is None else callback

    def __call__(self, value: Number) -> bool:
        if self._last_value is None:
            self._last_value = value
            return True

        if self._minimization:
            if value > self._last_value:
                self._count += 1
            else:
                self._count = 0
        else:
            if value < self._last_value:
                self._count += 1
            else:
                self._count = 0

        self._last_value = value
        if self._count >= self._patience:
            self._writer(f"Overfit detected, patience reached")
            return False
        else:
            return self._callback()

    def reset(self) -> None:
        self._count = 0
        self._last_value = None
        self._writer(f"Overfit detector reset")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class OverfitDetector(contextlib.AbstractContextManager):
    def __init__(
        self,
        patience: int,
        writer: callable = print,
        minimization: bool = True,
        callback: callable = None,
    ):
        """
        NOTE: equality is not checked, only greater or less than

        :param patience:
        :type patience:
        :param writer:
        :type writer:
        :param minimization:
        :type minimization:
        """
        self._patience = patience
        self._writer = writer
        self._count = 0
        self._minimization = minimization  # as opposed to maximization
        self._last_value = None
        self._callback = sink if callback is None else callback

    def __call__(self, value: Number) -> bool:
        if self._last_value is None:
            self._last_value = value
            return False

        if self._minimization:
            if value > self._last_value:
                self._count += 1
            else:
                self._count = 0
        else:
            if value < self._last_value:
                self._count += 1
            else:
                self._count = 0

        self._last_value = value
        if self._count >= self._patience:
            self._writer(
                f"Overfit detected, patience reached {self._patience}, loss {value}"
            )
            return True
        else:
            self._callback()
            return False

    def reset(self) -> None:
        self._count = 0
        self._last_value = None
        self._writer(f"Overfit detector reset")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


if __name__ == "__main__":
    with OverfitDetector(patience=3, writer=print) as overfit_detector:
        for i in range(10):
            overfit_detector(i)
        overfit_detector.reset()
        for i in range(10):
            overfit_detector(i)
        overfit_detector.reset()
        for i in range(10):
            overfit_detector(-i)
        print("start training")
        for i in range(2):
            overfit_detector(i)
        overfit_detector(0)
        for i in range(2):
            overfit_detector(i)
        print("start overfitting")
        for i in range(10):
            overfit_detector(i)

    with ImprovementDetector(patience=3, writer=print) as improvement_detector:
        for i in range(10):
            improvement_detector(i)
        improvement_detector.reset()
        for i in range(10):
            improvement_detector(-i)
        print("start training")
        for i in range(2):
            improvement_detector(i)
        improvement_detector(0)
        for i in range(2):
            print(improvement_detector(i))
        print("start overfitting")
        for i in range(10):
            print(improvement_detector(i))
