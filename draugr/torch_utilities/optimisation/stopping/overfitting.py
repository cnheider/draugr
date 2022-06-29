import contextlib

from warg import sink, Number

__all__ = ["ImprovementDetector", "OverfitDetector"]


class ImprovementDetector(contextlib.AbstractContextManager):
    """description"""

    def __init__(
        self,
        patience: int,
        writer: callable = print,
        minimization: bool = True,
        callback: callable = None,
    ):
        """
        NOTE: strictly greater or less than is considered as improvement

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
        self._best_value = None
        self._callback = (lambda: True) if callback is None else callback
        self._best_idx = None

    def __call__(self, value: Number) -> bool:
        if self._best_value is None:
            self._best_value = value
            return True

        if self._minimization:
            if value >= self._best_value:
                self._count += 1
                if self._verbose:
                    self._writer(
                        f"No improvement since last update: {value}>{self._best_value}"
                    )
            else:
                self._count = 0
                self._best_value = value
        else:
            if value <= self._best_value:
                self._count += 1
            else:
                self._count = 0
                self._best_value = value

        if self._count >= self._patience:
            self._writer(f"No improvement detected, patience reached")
            return False
        else:
            return self._callback()

    def reset(self) -> None:
        """description"""
        self._count = 0
        self._best_value = None
        self._writer(f"Improvement detector reset")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class OverfitDetector(contextlib.AbstractContextManager):
    """description"""

    def __init__(
        self,
        patience: int,
        writer: callable = print,
        minimization: bool = True,
        callback: callable = None,
        verbose: bool = False,
    ):
        """
        NOTE: equality, greater or less than

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
        self._best_value = None
        self._callback = sink if callback is None else callback
        self._verbose = verbose

    def __call__(self, value: Number) -> bool:
        if self._best_value is None:
            self._best_value = value
            return False

        if self._minimization:
            if value > self._best_value:
                self._count += 1
                if self._verbose:
                    self._writer(f"Worse than last update: {value}>{self._best_value}")
            else:
                self._count = 0
                self._best_value = value
        else:
            if value < self._best_value:
                self._count += 1
                if self._verbose:
                    self._writer(f"Worse than last update: {value}<{self._best_value}")
            else:
                self._count = 0
                self._best_value = value

        if self._count >= self._patience:
            self._writer(
                f"Overfit detected, patience reached {value}>{self._best_value} with patience {self._patience}"
            )
            return True
        else:
            self._callback()
            return False

    def reset(self) -> None:
        """description"""
        self._count = 0
        self._best_value = None
        self._writer(f"Overfit detector reset")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


if __name__ == "__main__":
    with OverfitDetector(patience=3, writer=print) as is_overfitting:
        for i in range(10):
            is_overfitting(i)
        is_overfitting.reset()
        for i in range(10):
            is_overfitting(-i)

        is_overfitting.reset()
        print("start training")
        for i in range(2):
            is_overfitting(i)
        is_overfitting(0)
        print("start overfitting")
        for i in range(10):
            print(is_overfitting(i), i)

    print("\n\n")

    with ImprovementDetector(patience=3, writer=print) as is_improving:
        for i in range(10):
            is_improving(i)
        is_improving.reset()
        for i in range(10):
            is_improving(-i)
        is_improving.reset()
        print("start training")
        for i in range(2):
            is_improving(i)
        is_improving(0)
        print("start not improving")
        for i in range(10):
            print(is_improving(i), i)
