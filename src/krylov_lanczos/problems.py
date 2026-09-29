from __future__ import annotations

import numpy as np
from scipy import sparse


def laplacian_2d(side: int = 32):
    """Five-point 2-D Dirichlet Laplacian without grid-spacing scaling."""
    if side < 2:
        raise ValueError("side must be at least 2")
    main = 2.0 * np.ones(side)
    off = -1.0 * np.ones(side - 1)
    L1 = sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    I = sparse.eye(side, format="csr")
    return sparse.kron(L1, I, format="csr") + sparse.kron(I, L1, format="csr")


def laplacian_2d_eigenvalues(side: int = 32) -> np.ndarray:
    j = np.arange(1, side + 1, dtype=float)
    lam1 = 2.0 - 2.0 * np.cos(np.pi * j / (side + 1))
    vals = (lam1[:, None] + lam1[None, :]).ravel()
    return np.sort(vals)


def _haar_orthogonal(n: int, seed: int) -> np.ndarray:
    """Deterministic Haar-like orthogonal matrix from a Gaussian QR factorization."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs


def clustered_spectrum(n: int = 300) -> np.ndarray:
    """Clustered spectrum used in Experiments 3--5."""
    if n < 30:
        raise ValueError("n must be at least 30")
    lower = np.array([0.10, 0.11, 0.12])
    middle = np.linspace(0.5, 2.0, 10)
    upper_cluster = np.array([4.90, 5.00, 5.01, 5.02])
    remaining = np.linspace(6.0, 9.5, n - lower.size - middle.size - upper_cluster.size)
    return np.sort(np.concatenate([lower, middle, upper_cluster, remaining]))


def clustered_symmetric_matrix(n: int = 300, seed: int = 42):
    """Dense symmetric matrix ``A = Q Lambda Q^T`` used in Experiments 3--5."""
    eigvals = clustered_spectrum(n)
    Q = _haar_orthogonal(n, seed)
    A = (Q * eigvals) @ Q.T
    return A, eigvals, Q


def theory_spectrum(n: int = 400) -> np.ndarray:
    """Spectrum used in Experiments 6a--6b."""
    if n < 50:
        raise ValueError("n must be at least 50")
    eigvals = np.concatenate(
        [
            np.array([0.05]),
            np.linspace(1.0, 2.0, 30),
            np.array([4.0, 4.0 + 1e-6, 4.0 + 2e-6]),
            np.linspace(6.0, 10.0, n - 34),
        ]
    )
    return np.sort(eigvals)


def theory_matrix(n: int = 400, seed: int = 13):
    """Dense symmetric ``A = Q Lambda Q^T`` for the KPS/Davis--Kahan experiments."""
    eigvals = theory_spectrum(n)
    Q = _haar_orthogonal(n, seed)
    A = (Q * eigvals) @ Q.T
    return A, eigvals, Q


def controlled_start_vector(
    eigenvector: np.ndarray,
    *,
    angle_deg: float = 86.6,
    seed: int = 13,
) -> np.ndarray:
    """Construct a unit vector at a prescribed angle from ``eigenvector``.

    The perpendicular component is generated deterministically from ``seed``.
    """
    z = np.asarray(eigenvector, dtype=float)
    z = z / np.linalg.norm(z)
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(z.size)
    p = p - z * float(z @ p)
    p = p / np.linalg.norm(p)
    a = np.deg2rad(angle_deg)
    v = np.cos(a) * z + np.sin(a) * p
    return v / np.linalg.norm(v)
