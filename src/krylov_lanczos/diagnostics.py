from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .lanczos import LanczosResult


def orthogonality_error(Q: np.ndarray, *, norm: str = "max") -> float:
    G = Q.T @ Q - np.eye(Q.shape[1])
    if norm == "fro":
        return float(np.linalg.norm(G, ord="fro"))
    if norm == "max":
        G = G.copy()
        np.fill_diagonal(G, 0.0)
        return float(np.max(np.abs(G))) if G.size else 0.0
    raise ValueError("norm must be 'max' or 'fro'")


def max_offdiagonal_history(Q: np.ndarray) -> np.ndarray:
    """Return ``max_{i<j}|q_i^T q_j|`` as each new vector is added."""
    if Q.shape[1] <= 1:
        return np.empty(0)
    return np.array(
        [np.max(np.abs(Q[:, :j].T @ Q[:, j])) for j in range(1, Q.shape[1])],
        dtype=float,
    )


def ritz_history(result: LanczosResult):
    history = []
    for k in range(1, result.m + 1):
        d = result.alpha[:k]
        if k == 1:
            vals = d.copy()
        else:
            vals = eigh_tridiagonal(d, result.beta[: k - 1], eigvals_only=True)
        history.append(vals)
    return history


def ritz_pair_at_step(result: LanczosResult, k: int, target: float):
    """Return the Ritz pair at Krylov dimension ``k`` closest to ``target``."""
    if not (1 <= k <= result.m):
        raise ValueError("k must satisfy 1 <= k <= result.m")
    d = result.alpha[:k]
    if k == 1:
        theta = d.copy()
        S = np.ones((1, 1))
    else:
        theta, S = eigh_tridiagonal(d, result.beta[: k - 1])
    idx = int(np.argmin(np.abs(theta - target)))
    y = result.Q[:, :k] @ S[:, idx]
    return float(theta[idx]), y, S[:, idx]


def nearest_ritz_error(ritz_values: np.ndarray, target: float) -> float:
    return float(np.min(np.abs(np.asarray(ritz_values) - target)))


def normalized_ritz_errors(ritz_values: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(spectrum)
    spread = float(spectrum[-1] - spectrum[0])
    return np.array([np.min(np.abs(spectrum - t)) / spread for t in ritz_values])


def absolute_ritz_errors(ritz_values: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(spectrum)
    return np.array([np.min(np.abs(spectrum - t)) for t in ritz_values])


def count_exact_ghosts(ritz_values: np.ndarray, spectrum: np.ndarray, tol: float = 1e-4) -> int:
    """Count duplicate Ritz values close to the same exact eigenvalue.

    This implements the criterion stated in the report: a Ritz value is a
    ghost when it is within ``tol`` of an exact eigenvalue and another Ritz
    value is even closer to that same eigenvalue.  Thus a multiplicity ``r``
    contributes ``r-1`` ghosts.
    """
    ritz = np.asarray(ritz_values)
    spectrum = np.asarray(spectrum)
    nearest = np.argmin(np.abs(ritz[:, None] - spectrum[None, :]), axis=1)
    dist = np.abs(ritz - spectrum[nearest])
    count = 0
    for idx in np.unique(nearest[dist < tol]):
        multiplicity = int(np.sum((nearest == idx) & (dist < tol)))
        count += max(multiplicity - 1, 0)
    return count


def count_heuristic_ghosts(current: np.ndarray, previous: np.ndarray, tol: float = 1e-4) -> int:
    """Heuristic ghost count using only roots of ``T_m`` and ``T_{m-1}``.

    A root of ``T_m`` is called persistent when it lies within ``tol`` of a
    root of ``T_{m-1}``.  Two or more persistent roots that coalesce within
    ``tol`` are counted as duplicate copies.  This mirrors the qualitative
    classifier described in the report without using the exact spectrum.
    """
    current = np.asarray(current)
    previous = np.asarray(previous)
    if previous.size == 0 or current.size < 2:
        return 0
    persistent = np.min(np.abs(current[:, None] - previous[None, :]), axis=1) < tol
    vals = np.sort(current[persistent])
    if vals.size < 2:
        return 0
    return int(np.sum(np.diff(vals) < tol))


def sin_angle(x: np.ndarray, y: np.ndarray) -> float:
    """Sine of the acute angle, computed stably near zero."""
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    c = float(x @ y)
    # ||y - x(x^T y)|| is numerically more reliable than sqrt(1-c^2)
    # when the vectors are almost parallel.
    return float(np.linalg.norm(y - c * x))


def chebyshev_kps_bound(
    m: int,
    lam_min: float,
    lam_2: float,
    lam_max: float,
    start: np.ndarray,
    eigenvector_min: np.ndarray,
) -> float:
    """Exact Chebyshev KPS upper bound for the smallest Ritz value error."""
    v = start / np.linalg.norm(start)
    z = eigenvector_min / np.linalg.norm(eigenvector_min)
    cosang = float(np.clip(abs(v @ z), np.finfo(float).tiny, 1.0))
    tan2 = (1.0 - cosang * cosang) / (cosang * cosang)
    gamma = (lam_2 - lam_min) / (lam_max - lam_2)
    x = 1.0 + 2.0 * gamma
    degree = max(m - 1, 0)
    T = np.cosh(degree * np.arccosh(x))
    return float((lam_max - lam_min) * tan2 / (T * T))


def kps_asymptotic_parameters(lam_min: float, lam_2: float, lam_max: float, C: float):
    gamma = (lam_2 - lam_min) / (lam_max - lam_2)
    kappa = (lam_max - lam_min) / (lam_2 - lam_min)
    rho = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    return float(gamma), float(kappa), float(rho), float(C)


def kps_asymptotic_normalized(m: np.ndarray | int, C: float, rho: float):
    """Simplified normalized curve ``C rho^(2m)`` used in the report."""
    return C * np.power(rho, 2 * np.asarray(m))
