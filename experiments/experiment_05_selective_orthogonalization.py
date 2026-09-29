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

    results = {
        "standard": lanczos(A, v0, K, reorthogonalization="none"),
        "selective": lanczos(A, v0, K, reorthogonalization="selective"),
        "fro": lanczos(A, v0, K, reorthogonalization="full"),
    }
    summary = {"n": N, "seed": SEED, "k": K, "sqrt_eps": float(SQRT_EPS)}
    orth = {}
    errors = {}
    for name, result in results.items():
        orth[name] = max_offdiagonal_history(result.Q)
        errors[name] = absolute_ritz_errors(result.ritz(), spectrum)
        summary[f"{name}_final_orthogonality"] = float(orth[name][-1])
        summary[f"{name}_ritz_mean_error"] = float(np.mean(errors[name]))
        summary[f"{name}_ritz_max_error"] = float(np.max(errors[name]))
    summary["selective_is_semi_orthogonal"] = bool(np.max(orth["selective"]) <= SQRT_EPS)
    summary["selective_max_active_vectors"] = int(np.max(results["selective"].selective_counts))
    save_summary(output_dir / "experiment_05.json", summary)

    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    steps = np.arange(1, K)
    for name in results:
        axes[0].semilogy(steps, orth[name], label=name)
    axes[0].axhline(SQRT_EPS, linestyle="--", label=r"$\sqrt{\epsilon}$")
    axes[0].set_xlabel("step j")
    axes[0].set_ylabel(r"$\max_{i<j}|q_i^Tq_j|$")
    axes[0].legend()
    for name, result in results.items():
        axes[1].semilogy(result.ritz(), errors[name], marker="o", label=name)
    axes[1].set_xlabel("Ritz value")
    axes[1].set_ylabel("distance to nearest exact eigenvalue")
    axes[1].legend()
    fig.savefig(figures / "experiment_05_selective_orthogonalization.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
