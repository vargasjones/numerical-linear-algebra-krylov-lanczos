import numpy as np

from krylov_lanczos import (
    absolute_ritz_errors,
    clustered_symmetric_matrix,
    lanczos,
    laplacian_2d,
    laplacian_2d_eigenvalues,
    max_offdiagonal_history,
    ritz_history,
)


def test_lanczos_full_reorthogonalization_preserves_orthogonality():
    A, spectrum, _ = clustered_symmetric_matrix(120, seed=42)
    v0 = np.random.default_rng(42).standard_normal(120)
    result = lanczos(A, v0, 35, reorthogonalization="full")
    off = max_offdiagonal_history(result.Q)
    assert off.max() < 1e-12
    assert absolute_ritz_errors(result.ritz(), spectrum).max() < 0.05


def test_experiment_1_convergence_counts():
    A = laplacian_2d(32)
    exact = laplacian_2d_eigenvalues(32)
    spread = exact[-1] - exact[0]
    v0 = np.random.default_rng(7).standard_normal(A.shape[0])
    result = lanczos(A, v0, 80, reorthogonalization="none")
    hist = ritz_history(result)

    def conv(target):
        errors = [np.min(np.abs(rv - target)) / spread for rv in hist]
        return next((k for k, e in enumerate(errors, 1) if e <= 1e-8), None)

    assert conv(exact[0]) == 65
    assert conv(exact[-1]) == 72
    assert conv(exact[49]) is None
    assert conv(exact[199]) is None
