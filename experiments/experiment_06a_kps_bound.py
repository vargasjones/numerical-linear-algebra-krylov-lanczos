from __future__ import annotations

from pathlib import Path

import numpy as np

from krylov_lanczos import (
    controlled_start_vector,
    kps_asymptotic_normalized,
    kps_asymptotic_parameters,
    lanczos,
    ritz_pair_at_step,
    theory_matrix,
)
from common import first_below, save_summary, setup_matplotlib, single_threaded

N = 400
SEED = 13
M_MAX = 110
TOL = 1e-8
C_TARGET = 287.0
ANGLE_DEG = float(np.degrees(np.arctan(np.sqrt(C_TARGET))))


@single_threaded
def run(output_dir: str | Path = "results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    A, spectrum, Q = theory_matrix(N, SEED)
    z1 = Q[:, 0]
    v1 = controlled_start_vector(z1, angle_deg=ANGLE_DEG, seed=SEED)
    result = lanczos(A, v1, M_MAX, reorthogonalization="full")
    spread = spectrum[-1] - spectrum[0]

    errors = []
    for m in range(1, result.m + 1):
        theta, _, _ = ritz_pair_at_step(result, m, spectrum[0])
        errors.append(abs(theta - spectrum[0]) / spread)
    errors = np.asarray(errors)

    cosang = abs(float(v1 @ z1))
    C = (1.0 - cosang**2) / cosang**2
    gamma, kappa, rho, _ = kps_asymptotic_parameters(spectrum[0], spectrum[1], spectrum[-1], C)
    ms = np.arange(1, result.m + 1)
    bound = kps_asymptotic_normalized(ms, C, rho)
    mask = bound > np.finfo(float).eps
    upper_bound_holds = bool(np.all(bound[mask] >= errors[mask]))
    conv = first_below(errors, TOL)

    summary = {
        "n": N, "seed": SEED, "m_max": result.m, "tol": TOL,
        "lambda_iso": float(spectrum[0]), "lambda_2": float(spectrum[1]), "lambda_n": float(spectrum[-1]),
        "angle_deg": ANGLE_DEG, "C": float(C), "gamma": gamma, "kappa": kappa, "rho": rho,
        "conv_eigenvalue": conv, "upper_bound_holds_until_machine_precision": upper_bound_holds,
    }
    save_summary(output_dir / "experiment_06a.json", summary)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(ms, errors, label="observed normalized error")
    ax.semilogy(ms[mask], bound[mask], linestyle="--", label=r"$C\rho^{2m}$")
    ax.axhline(np.finfo(float).eps, linestyle=":", label="machine epsilon")
    ax.set_xlabel("Krylov dimension m")
    ax.set_ylabel("normalized eigenvalue error")
    ax.set_title("Kaniel–Paige–Saad asymptotic bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "experiment_06a_kps_bound.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
