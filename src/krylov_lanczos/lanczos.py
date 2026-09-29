from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import issparse

Reorthogonalization = Literal["none", "full", "selective"]


@dataclass
class LanczosResult:
    """Output of a Lanczos run.

    ``beta`` contains the off-diagonal entries of the final tridiagonal
    matrix ``T_m``. ``beta_next`` is the coefficient multiplying the next
    Lanczos vector in

        A Q_m = Q_m T_m + beta_next q_{m+1} e_m^T.
    """

    Q: NDArray[np.float64]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    beta_next: float
    selective_counts: NDArray[np.int64]

    @property
    def m(self) -> int:
        return self.alpha.size

    def tridiagonal(self) -> NDArray[np.float64]:
        T = np.diag(self.alpha)
        if self.beta.size:
            T += np.diag(self.beta, 1) + np.diag(self.beta, -1)
        return T

    def ritz(self, vectors: bool = False):
        if self.m == 1:
            values = self.alpha.copy()
            if vectors:
                return values, self.Q.copy()
            return values
        values, S = eigh_tridiagonal(self.alpha, self.beta)
        if vectors:
            return values, self.Q @ S
        return values

    def ritz_residual_estimates(self) -> NDArray[np.float64]:
        """Return ``|beta_m e_m^T s_i|`` for all Ritz pairs."""
        if self.m == 1:
            return np.array([abs(self.beta_next)], dtype=float)
        _, S = eigh_tridiagonal(self.alpha, self.beta)
        return abs(self.beta_next) * np.abs(S[-1, :])


def _as_matvec(A) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    if callable(A):
        return A
    if issparse(A):
        return lambda x: np.asarray(A @ x, dtype=float)
    M = np.asarray(A, dtype=float)
    return lambda x: M @ x


def _double_mgs(w: NDArray[np.float64], V: NDArray[np.float64]) -> NDArray[np.float64]:
    """Two modified-Gram--Schmidt sweeps against the columns of ``V``."""
    if V.size == 0:
        return w
    for _ in range(2):
        for k in range(V.shape[1]):
            w = w - float(V[:, k] @ w) * V[:, k]
    return w


def lanczos(
    A,
    v0: ArrayLike,
    m: int,
    *,
    reorthogonalization: Reorthogonalization = "none",
    tol: float = 1e-14,
    selective_threshold: float | None = None,
) -> LanczosResult:
    """Lanczos iteration for a real symmetric linear operator.

    The three-term recurrence computes ``alpha_j = q_j^T A q_j`` before
    subtracting the current and previous Lanczos directions.

    Parameters
    ----------
    A:
        Dense/sparse matrix or callable implementing ``A @ x``.
    v0:
        Nonzero starting vector.
    m:
        Maximum Krylov dimension.
    reorthogonalization:
        ``"none"`` for standard Lanczos, ``"full"`` for two MGS sweeps
        against the whole basis, or ``"selective"`` for a Parlett--Scott
        style selective strategy based on converged Ritz vectors.
    tol:
        Lucky-breakdown threshold.
    selective_threshold:
        ``nu`` in the selective criterion. Defaults to
        ``sqrt(machine epsilon)``.
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if reorthogonalization not in {"none", "full", "selective"}:
        raise ValueError("unknown reorthogonalization strategy")

    matvec = _as_matvec(A)
    q = np.asarray(v0, dtype=float).reshape(-1)
    nq = np.linalg.norm(q)
    if nq == 0:
        raise ValueError("v0 must be nonzero")
    q = q / nq

    n = q.size
    Q = np.zeros((n, m), dtype=float)
    alpha = np.zeros(m, dtype=float)
    beta = np.zeros(max(m - 1, 0), dtype=float)
    selective_counts = np.zeros(m, dtype=np.int64)

    q_prev = np.zeros_like(q)
    beta_prev = 0.0
    beta_next = 0.0
    actual_m = m
    nu = np.sqrt(np.finfo(float).eps) if selective_threshold is None else float(selective_threshold)
    converged_ritz_vectors: list[NDArray[np.float64]] = []

    for j in range(m):
        Q[:, j] = q

        # Three-term Lanczos recurrence.
        w = matvec(q)
        alpha[j] = float(q @ w)
        w = w - alpha[j] * q
        if j > 0:
            w = w - beta_prev * q_prev

        V = Q[:, : j + 1]
        if reorthogonalization == "full":
            w = _double_mgs(w, V)

        elif reorthogonalization == "selective" and j >= 1:
            # Parlett--Scott-style selective criterion:
            # omega_{j,i} = beta_{j+1} |s_{j,i}| < nu ||T_j||.
            raw_beta = float(np.linalg.norm(w))
            theta, S = eigh_tridiagonal(alpha[: j + 1], beta[:j])
            tnorm = max(float(np.max(np.abs(theta))), np.finfo(float).tiny)
            omega = raw_beta * np.abs(S[-1, :])
            active = np.flatnonzero(omega < nu * tnorm)

            # Store newly converged Ritz directions while avoiding duplicates.
            for idx in active:
                y = V @ S[:, idx]
                ny = np.linalg.norm(y)
                if ny == 0:
                    continue
                y = y / ny
                if all(abs(float(y @ old)) < 1.0 - 1e-6 for old in converged_ritz_vectors):
                    converged_ritz_vectors.append(y)

            selective_counts[j] = len(converged_ritz_vectors)
            for y in converged_ritz_vectors:
                w = w - float(y @ w) * y

            # A second pass only over the selected directions is inexpensive
            # and prevents already selected components from being reintroduced
            # by roundoff.
            for y in converged_ritz_vectors:
                w = w - float(y @ w) * y

        beta_next = float(np.linalg.norm(w))
        if beta_next <= tol:
            actual_m = j + 1
            break

        if j < m - 1:
            beta[j] = beta_next
            q_prev = q
            q = w / beta_next
            beta_prev = beta_next

    Q = Q[:, :actual_m]
    alpha = alpha[:actual_m]
    beta = beta[: max(actual_m - 1, 0)]
    selective_counts = selective_counts[:actual_m]
    return LanczosResult(Q, alpha, beta, beta_next, selective_counts)
