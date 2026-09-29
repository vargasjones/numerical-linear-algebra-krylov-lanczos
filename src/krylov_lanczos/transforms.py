from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass
class ShiftInvertOperator:
    """Callable sparse shift-and-invert operator ``(A - sigma I)^{-1}``."""

    sigma: float
    setup_seconds: float
    solve_seconds: float = 0.0
    solve_count: int = 0
    solve_times: list[float] = field(default_factory=list)

    def __init__(self, A, sigma: float):
        self.sigma = float(sigma)
        n = A.shape[0]
        M = sparse.csc_matrix(A) - self.sigma * sparse.eye(n, format="csc")
        t0 = perf_counter()
        self._lu = splu(M)
        self.setup_seconds = perf_counter() - t0
        self.solve_seconds = 0.0
        self.solve_count = 0
        self.solve_times = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        t0 = perf_counter()
        y = self._lu.solve(x)
        dt = perf_counter() - t0
        self.solve_seconds += dt
        self.solve_count += 1
        self.solve_times.append(dt)
        return y

    def map_back(self, mu):
        mu = np.asarray(mu, dtype=float)
        return self.sigma + 1.0 / mu
