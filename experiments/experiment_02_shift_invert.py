from __future__ import annotations

from pathlib import Path

import numpy as np

from krylov_lanczos import (
    ShiftInvertOperator,
    lanczos,
    laplacian_2d,
    laplacian_2d_eigenvalues,
    ritz_history,
)
from common import first_below, save_summary, setup_matplotlib, single_threaded

SIDE = 32
M_MAX = 80
TOL = 1e-8
SEED = 7
SHIFT_OFFSET = 1e-4


def mapped_error_history(result, operator, target, spread):
    out = []
    for mu in ritz_history(result):
        lam = operator.map_back(mu)
        out.append(float(np.min(np.abs(lam - target)) / spread))
    return np.asarray(out)


def run_case(A, v0, sigma, target, spread):
    op = ShiftInvertOperator(A, sigma)
    result = lanczos(op, v0, M_MAX, reorthogonalization="none")
    errors = mapped_error_history(result, op, target, spread)
    # The report starts its convergence comparison at m=3.
    conv = first_below(errors[2:], TOL, start_index=3)
    solve_to_conv = None
    if conv is not None:
        solve_to_conv = float(sum(op.solve_times[:conv]))
    return op, result, errors, conv, solve_to_conv


@single_threaded
def run(output_dir: str | Path = "results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    A = laplacian_2d(SIDE)
    exact = laplacian_2d_eigenvalues(SIDE)
    spread = exact[-1] - exact[0]
    lam50, lam200 = exact[49], exact[199]
    sigma50 = lam50 - SHIFT_OFFSET
    sigma200 = lam200 - SHIFT_OFFSET
    sigma_far = 0.5 * (lam50 + lam200)

    rng = np.random.default_rng(SEED)
    v0 = rng.standard_normal(A.shape[0])
    v0 /= np.linalg.norm(v0)

    cases = {
        "near_lambda50": run_case(A, v0, sigma50, lam50, spread),
        "near_lambda200": run_case(A, v0, sigma200, lam200, spread),
        "far_for_lambda50": run_case(A, v0, sigma_far, lam50, spread),
    }

    summary = {
        "n": int(A.shape[0]), "nnz": int(A.nnz), "m_max": M_MAX, "seed": SEED, "tol": TOL,
        "lambda_50": float(lam50), "lambda_200": float(lam200),
        "sigma_50": float(sigma50), "sigma_200": float(sigma200), "sigma_far": float(sigma_far),
    }
    for name, (op, result, errors, conv, solve_to_conv) in cases.items():
        summary[f"{name}_conv"] = conv
        summary[f"{name}_setup_seconds"] = float(op.setup_seconds)
        summary[f"{name}_solve_seconds_to_conv"] = solve_to_conv
        summary[f"{name}_final_error"] = float(errors[-1])
    save_summary(output_dir / "experiment_02.json", summary)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, (_, _, errors, _, _) in cases.items():
        ax.semilogy(np.arange(1, len(errors) + 1), errors, label=name)
    ax.axhline(TOL, linestyle="--", label="tolerance")
    ax.set_xlabel("Krylov dimension m")
    ax.set_ylabel("Normalized target-eigenvalue error")
    ax.set_title("Shift-and-invert for interior eigenvalues")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "experiment_02_shift_invert.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
