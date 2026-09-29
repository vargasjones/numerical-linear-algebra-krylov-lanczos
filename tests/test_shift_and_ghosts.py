import numpy as np

from krylov_lanczos import (
    ShiftInvertOperator,
    clustered_symmetric_matrix,
    count_exact_ghosts,
    count_heuristic_ghosts,
    lanczos,
    laplacian_2d,
    laplacian_2d_eigenvalues,
    ritz_history,
)


def test_shift_invert_near_shift_converges_by_three_steps():
    A = laplacian_2d(32)
    exact = laplacian_2d_eigenvalues(32)
    spread = exact[-1] - exact[0]
    target = exact[49]
    v0 = np.random.default_rng(7).standard_normal(A.shape[0])
    op = ShiftInvertOperator(A, target - 1e-4)
    result = lanczos(op, v0, 3)
    mapped = op.map_back(result.ritz())
    assert np.min(np.abs(mapped - target)) / spread <= 1e-8


def test_ghost_classifier_detects_artificial_multiplicity():
    A, spectrum, _ = clustered_symmetric_matrix(300, 42)
    v0 = np.random.default_rng(42).standard_normal(300)
    result = lanczos(A, v0, 150)
    history = ritz_history(result)
    exact_final = count_exact_ghosts(history[-1], spectrum, 1e-4)
    heuristic_final = count_heuristic_ghosts(history[-1], history[-2], 1e-4)
    assert exact_final == 31
    assert heuristic_final == 31
