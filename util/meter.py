# misc imports
import torch
import numpy as np
from abc import abstractmethod
from typing import Union, Iterable

# Define some useful types
Numeric = Union[np.ndarray, torch.Tensor, int, float]
Numerics = Iterable[Numeric]


class Meter:
    def __init__(self, iterable=None):
        if iterable is not None:
            self.addN(iterable)

    @abstractmethod
    def add(self, datum: Numeric):
        pass

    def addN(self, iterable: Numerics):
        for datum in iterable:
            self.add(datum)


class StatsMeter(Meter):
    """
    Auxiliary classs to keep track of online stats including:
        - mean
        - std / variance
    Uses Welford's algorithm to compute sample mean and sample variance incrementally.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    """

    def __init__(self, iterable: Numerics = None):
        """Online Mean and Variance from single samples

        Running stats,
        This is compatible with np.ndarray objects and as long as the

        Keyword Arguments:
            iterable {[iterable} -- Values to initialize (default: {None})
        """
        self.n = 0
        self.mean: Numeric = 0.0
        self.S: Numeric = 0.0
        super().__init__(iterable)

    def verify_meter(self):
        assert self.S >= 0, f"Variance must be non-negative, got {self.S}."
        assert self.n >= 0, f"Number of samples must be non-negative, got {self.n}."

    def add(self, datum: Numeric, n = 1):
        """Add a single datum

        Internals are updated using Welford's method

        Arguments:
            datum  -- Numerical object.
            amount -- Weight of the sample in the running stats.
        """
        self.n += n 
        delta = datum - self.mean
        # Mk = Mk-1+ (xk – Mk-1)/k
        self.mean += delta / self.n
        # Sk = Sk-1 + (xk – Mk-1)*(xk – Mk).
        self.S += delta * (datum - self.mean)
        # update check:
        self.verify_meter()
        
    def addN(self, iterable: Numerics, batch: bool = False):
        """Add N data to the stats

        Arguments:
            iterable {[type]} -- [description]

        Keyword Arguments:
            batch {bool} -- If true, then the mean and std are computed over
            the new array using numpy and then that updates the current stats
        """
        if batch:
            add = self + StatsMeter.from_values(
                len(iterable), np.mean(iterable), np.std(iterable)
            )
            self.n, self.mean, self.S = add.n, add.mean, add.S
        else:
            super().addN(iterable)

    def pop(self, datum: Numeric):
        if self.n == 0:
            raise ValueError("Stats must be non empty")

        self.n -= 1
        delta = datum - self.mean
        # Mk-1 = Mk - (xk - Mk) / (k - 1)
        self.mean -= delta / self.n
        # Sk-1 = Sk - (xk – Mk-1) * (xk – Mk)
        self.S -= (datum - self.mean) * delta

    def popN(self, iterable: Numerics, batch: bool = False):
        if batch:
            raise NotImplementedError
        else:
            for datum in iterable:
                self.pop(datum)

    @property
    def variance(self) -> Numeric:
        # For 2 ≤ k ≤ n, the kth estimate of the variance is s2 = Sk/(k – 1).
        assert self.S >= 0, f"Sum of squares must be non-negative, got {self.S}."
        assert self.n >= 0, f"Num samples must be non-negtative, got {self.n}."
        return self.S / self.n

    @property
    def std(self) -> Numeric:
        assert self.variance >= 0, f"Variance must be positive (or zero), got {self.variance}."
        return np.sqrt(self.variance)

    @staticmethod
    def from_values(n: int, mean: float, std: float) -> "StatsMeter":
        stats = StatsMeter()
        stats.n = n
        stats.mean = mean
        stats.S = std**2 * n
        return stats

    @staticmethod
    def from_raw_values(n: int, mean: float, S: float) -> "StatsMeter":
        stats = StatsMeter()
        stats.n = n
        stats.mean = mean
        stats.S = S
        return stats

    def __str__(self) -> str:
        return f"n={self.n}  mean={self.mean}  std={self.std}"

    def __repr__(self) -> str:
        if self.n == 0:
            return f"{self.__class__.__name__}()"
        return (
            f"{self.__class__.__name__}.from_values("
            + f"n={self.n}, mean={self.mean}, "
            + f"std={self.std})"
        )

    def __add__(self, other: Union[Numeric, "StatsMeter"]) -> "StatsMeter":
        """Adding can be done with int|float or other StatsMeter objects

        For other int|float, it is added to all previous values

        Arguments:
            other {[type]} -- [description]

        Returns:
            StatsMeter -- New instance with the sum.

        Raises:
            TypeError -- If the type is different from int|float|OnlineStas
        """
        if isinstance(other, StatsMeter):
            # Add the means, variances and n_samples of two objects
            n1, n2 = self.n, other.n
            mu1, mu2 = self.mean, other.mean
            S1, S2 = self.S, other.S
            # New stats
            n = n1 + n2
            mu = n1 / n * mu1 + n2 / n * mu2
            S = (S1 + n1 * mu1 * mu1) + (S2 + n2 * mu2 * mu2) - n * mu * mu
            return StatsMeter.from_raw_values(n, mu, S)

        if isinstance(other, (int, float)):
            # Add a fixed amount to all values. Only changes the mean
            return StatsMeter.from_raw_values(self.n, self.mean + other, self.S)
        else:
            raise TypeError("Can only add other groups or numbers")

    def __mul__(self, k: Union[float, int]) -> "StatsMeter":
        # Multiply all values seen by some constant
        return StatsMeter.from_raw_values(self.n, self.mean * k, self.S * k**2)

    def asdict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "n": self.n}

    @property
    def flatmean(self) -> float:
        # for datapoints which are arrays
        return np.mean(self.mean)

    @property
    def flatvariance(self) -> float:
        # for datapoints which are arrays
        return np.mean(self.variance + self.mean**2) - self.flatmean**2

    @property
    def flatstd(self) -> float:
        return np.sqrt(self.flatvariance)


class MeterDict(dict):
    def __init__(self):
        super().__init__()

    def update(self, data):
        for label, value in data.items():
            self[label].add(value)

    def __setitem__(self, key, value):
        if key not in self:
            self[key]
        self[key].add(value)

    def __getitem__(self, key):
        if key not in self:
            super().__setitem__(key, StatsMeter)
        return super().__getitem__(key)

    def collect(self, attr):
        return {label: getattr(meter, attr) for label, meter in self.items()}

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(dict(self))})"

    def add(self, label, value):
        self[label].add(value)