import numpy as np

from krylov_lanczos import (
    absolute_ritz_errors,
    clustered_symmetric_matrix,
    controlled_start_vector,
    kps_asymptotic_normalized,
    kps_asymptotic_parameters,
    lanczos,
    max_offdiagonal_history,
    ritz_pair_at_step,
    theory_matrix,
)


def test_selective_orthogonalization_is_semi_orthogonal_and_matches_fro_ritz_accuracy():
    A, spectrum, _ = clustered_symmetric_matrix(300, 42)
    v0 = np.random.default_rng(42).standard_normal(300)
    so = lanczos(A, v0, 60, reorthogonalization="selective")
    fro = lanczos(A, v0, 60, reorthogonalization="full")
    sqrt_eps = np.sqrt(np.finfo(float).eps)
    assert max_offdiagonal_history(so.Q).max() <= sqrt_eps
    e_so = absolute_ritz_errors(so.ritz(), spectrum)
    e_fro = absolute_ritz_errors(fro.ritz(), spectrum)
    assert abs(e_so.mean() - e_fro.mean()) < 1e-10
    assert abs(e_so.max() - e_fro.max()) < 1e-10


def test_kps_simplified_bound_is_valid():
    A, spectrum, Q = theory_matrix(400, 13)
    C_target = 287.0
    angle = np.degrees(np.arctan(np.sqrt(C_target)))
    v1 = controlled_start_vector(Q[:, 0], angle_deg=angle, seed=13)
    result = lanczos(A, v1, 110, reorthogonalization="full")
    spread = spectrum[-1] - spectrum[0]
    errors = []
    for m in range(1, result.m + 1):
        theta, _, _ = ritz_pair_at_step(result, m, spectrum[0])
        errors.append(abs(theta - spectrum[0]) / spread)
    errors = np.asarray(errors)
    cosang = abs(float(v1 @ Q[:, 0]))
    C = (1 - cosang**2) / cosang**2
    _, _, rho, _ = kps_asymptotic_parameters(spectrum[0], spectrum[1], spectrum[-1], C)
    bound = kps_asymptotic_normalized(np.arange(1, result.m + 1), C, rho)
    mask = bound > np.finfo(float).eps
    assert np.all(bound[mask] >= errors[mask])
    assert next(i for i, e in enumerate(errors, 1) if e <= 1e-8) == 17
