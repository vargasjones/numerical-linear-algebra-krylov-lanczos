from __future__ import annotations

from pathlib import Path

import numpy as np

from krylov_lanczos import (
    absolute_ritz_errors,
    clustered_symmetric_matrix,
    lanczos,
    max_offdiagonal_history,
)
from common import save_summary, setup_matplotlib, single_threaded

N = 300
SEED = 42
K = 60
SQRT_EPS = np.sqrt(np.finfo(float).eps)


@single_threaded
def run(output_dir: str | Path = "results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    A, spectrum, _ = clustered_symmetric_matrix(N, SEED)
    rng = np.random.default_rng(SEED)
    v0 = rng.standard_normal(N)
    v0 /= np.linalg.norm(v0)

    standard = lanczos(A, v0, K, reorthogonalization="none")
    fro = lanczos(A, v0, K, reorthogonalization="full")

    o_std = max_offdiagonal_history(standard.Q)
    o_fro = max_offdiagonal_history(fro.Q)
    e_std = absolute_ritz_errors(standard.ritz(), spectrum)
    e_fro = absolute_ritz_errors(fro.ritz(), spectrum)
    crossing = next((j for j, x in zip(range(1, K), o_std) if x > SQRT_EPS), None)

    summary = {
        "n": N, "seed": SEED, "k": K,
        "standard_final_orthogonality": float(o_std[-1]),
        "standard_cross_sqrt_eps_step": crossing,
        "standard_ritz_mean_error": float(np.mean(e_std)),
        "standard_ritz_max_error": float(np.max(e_std)),
        "fro_final_orthogonality": float(o_fro[-1]),
        "fro_ritz_mean_error": float(np.mean(e_fro)),
        "fro_ritz_max_error": float(np.max(e_fro)),
    }
    save_summary(output_dir / "experiment_04.json", summary)

    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    steps = np.arange(1, K)
    axes[0].semilogy(steps, o_std, label="standard")
    axes[0].semilogy(steps, o_fro, label="FRO")
    axes[0].axhline(SQRT_EPS, linestyle="--", label=r"$\sqrt{\epsilon}$")
    axes[0].set_xlabel("step j")
    axes[0].set_ylabel(r"$\max_{i<j}|q_i^Tq_j|$")
    axes[0].legend()
    axes[1].semilogy(standard.ritz(), e_std, marker="o", label="standard")
    axes[1].semilogy(fro.ritz(), e_fro, marker="s", label="FRO")
    axes[1].set_xlabel("Ritz value")
    axes[1].set_ylabel("distance to nearest exact eigenvalue")
    axes[1].legend()
    fig.savefig(figures / "experiment_04_standard_vs_fro.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
