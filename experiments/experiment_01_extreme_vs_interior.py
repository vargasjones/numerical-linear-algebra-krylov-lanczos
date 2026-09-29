from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh_tridiagonal

from krylov_lanczos import laplacian_2d, laplacian_2d_eigenvalues
from common import first_below, save_summary, setup_matplotlib, single_threaded

SIDE = 32
M_MAX = 80
TOL = 1e-8
SEED = 7


@single_threaded
def run(output_dir: str | Path = "results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    A = laplacian_2d(SIDE)
    exact = laplacian_2d_eigenvalues(SIDE)
    n = A.shape[0]
    spread = exact[-1] - exact[0]
    targets = {
        "lambda_min": exact[0],
        "lambda_max": exact[-1],
        "lambda_50": exact[49],
        "lambda_200": exact[199],
    }

    rng = np.random.default_rng(SEED)
    q = rng.standard_normal(n)
    q /= np.linalg.norm(q)

    Q = np.zeros((n, M_MAX))
    alpha = np.zeros(M_MAX)
    beta = np.zeros(M_MAX - 1)
    Q[:, 0] = q
    cumulative = []
    t0 = perf_counter()

    for j in range(M_MAX):
        w = A @ Q[:, j]
        alpha[j] = float(Q[:, j] @ w)
        w -= alpha[j] * Q[:, j]
        if j > 0:
            w -= beta[j - 1] * Q[:, j - 1]
        b = float(np.linalg.norm(w))
        cumulative.append(perf_counter() - t0)
        if b <= 1e-14:
            alpha = alpha[:j + 1]
            beta = beta[:j]
            Q = Q[:, :j + 1]
            break
        if j < M_MAX - 1:
            beta[j] = b
            Q[:, j + 1] = w / b

    ks = np.arange(3, alpha.size + 1)
    errors = {key: [] for key in targets}
    for k in ks:
        ritz = eigh_tridiagonal(alpha[:k], beta[: k - 1], eigvals_only=True)
        for key, target in targets.items():
            errors[key].append(float(np.min(np.abs(ritz - target)) / spread))

    conv = {key: first_below(vals, TOL, start_index=3) for key, vals in errors.items()}

    # Gap-ratio quantities in the same spirit as Table 4 of the report.
    def lower_gap_ratio(i):
        gap = exact[i + 1] - exact[i]
        gamma = gap / (exact[-1] - exact[i + 1])
        kappa = 1.0 + 1.0 / gamma if gamma > 0 else np.inf
        rho = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1) if np.isfinite(kappa) else np.nan
        return gap, gamma, rho

    gap_min, gamma_min, rho_min = lower_gap_ratio(0)
    # Symmetric formula at the upper end.
    gap_max = exact[-1] - exact[-2]
    gamma_max = gap_max / (exact[-2] - exact[0])
    kappa_max = 1.0 + 1.0 / gamma_max
    rho_max = (np.sqrt(kappa_max) - 1) / (np.sqrt(kappa_max) + 1)
    gap_50, gamma_50, rho_50 = lower_gap_ratio(49)
    gap_200, gamma_200, rho_200 = lower_gap_ratio(199)

    summary = {
        "side": SIDE,
        "n": n,
        "nnz": int(A.nnz),
        "m_max": int(alpha.size),
        "seed": SEED,
        "tol": TOL,
        **{key: float(value) for key, value in targets.items()},
        "conv_lambda_min": conv["lambda_min"],
        "conv_lambda_max": conv["lambda_max"],
        "conv_lambda_50": conv["lambda_50"],
        "conv_lambda_200": conv["lambda_200"],
        "gap_min": float(gap_min), "gamma_min": float(gamma_min), "rho_min": float(rho_min),
        "gap_max": float(gap_max), "gamma_max": float(gamma_max), "rho_max": float(rho_max),
        "gap_50": float(gap_50), "gamma_50": float(gamma_50), "rho_50": float(rho_50),
        "gap_200": float(gap_200), "gamma_200": float(gamma_200), "rho_200": float(rho_200),
        "cpu_seconds_at_end": float(cumulative[-1]),
    }
    save_summary(output_dir / "experiment_01.json", summary)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.asarray(cumulative)[ks - 1]
    for key, vals in errors.items():
        ax.semilogy(x, vals, marker="o", markersize=2.5, label=key)
    ax.axhline(TOL, linestyle="--", label="tolerance")
    ax.set_xlabel("Cumulative CPU time [s]")
    ax.set_ylabel(r"Normalized error $|\theta-\lambda|/(\lambda_{max}-\lambda_{min})$")
    ax.set_title("Extreme vs. interior eigenvalues — standard Lanczos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "experiment_01_extreme_vs_interior.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
