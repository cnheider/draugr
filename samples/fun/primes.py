from itertools import count
from typing import Iterable, Iterator

depth = 999
import sys

sys.setrecursionlimit(
    depth
)  # RecursionError: maximum recursion depth exceeded eventually


def naturals_generator(
    n: int,
) -> Iterable:  # RecursionError: maximum recursion depth exceeded eventually
    yield n
    yield from naturals_generator(n + 1)


def prime_sieve(
    generator: Iterator,
) -> Iterable:  # RecursionError: maximum recursion depth exceeded eventually
    n = next(generator)
    yield n
    yield from prime_sieve(i for i in generator if i % n != 0)


if __name__ == "__main__":
    for _, prime in zip(range(depth // 2), prime_sieve(count(2))):
        print(prime)
