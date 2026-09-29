from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal

from krylov_lanczos import (
    clustered_symmetric_matrix,
    count_exact_ghosts,
    count_heuristic_ghosts,
    lanczos,
    max_offdiagonal_history,
)
from common import save_summary, setup_matplotlib, single_threaded

N = 300
SEED = 42
K = 150
GHOST_TOL = 1e-4


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
    result = lanczos(A, v0, K, reorthogonalization="none")

    exact_counts, heuristic_counts = [], []
    first_ghost = None
    all_ritz = []
    all_ghost_flags = []
    prev = np.empty(0)
    for k in range(1, result.m + 1):
        if k == 1:
            rv = result.alpha[:1].copy()
        else:
            rv = eigh_tridiagonal(result.alpha[:k], result.beta[: k - 1], eigvals_only=True)
        c = count_exact_ghosts(rv, spectrum, GHOST_TOL)
        h = count_heuristic_ghosts(rv, prev, GHOST_TOL)
        exact_counts.append(c)
        heuristic_counts.append(h)
        if c > 0 and first_ghost is None:
            first_ghost = k

        nearest = np.argmin(np.abs(rv[:, None] - spectrum[None, :]), axis=1)
        dist = np.abs(rv - spectrum[nearest])
        flags = np.zeros(rv.size, dtype=bool)
        for idx in np.unique(nearest[dist < GHOST_TOL]):
            ids = np.flatnonzero((nearest == idx) & (dist < GHOST_TOL))
            if ids.size > 1:
                order = ids[np.argsort(dist[ids])]
                flags[order[1:]] = True
        all_ritz.append(rv)
        all_ghost_flags.append(flags)
        prev = rv

    ortho = max_offdiagonal_history(result.Q)
    summary = {
        "n": N, "seed": SEED, "k": result.m, "ghost_tol": GHOST_TOL,
        "first_exact_ghost_step": first_ghost,
        "final_exact_ghosts": int(exact_counts[-1]),
        "max_exact_ghosts": int(max(exact_counts)),
        "final_heuristic_ghosts": int(heuristic_counts[-1]),
        "max_heuristic_ghosts": int(max(heuristic_counts)),
        "final_max_offdiag_orthogonality": float(ortho[-1]),
    }
    save_summary(output_dir / "experiment_03.json", summary)

    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    for k, (rv, flags) in enumerate(zip(all_ritz, all_ghost_flags), start=1):
        axes[0].scatter(np.full(np.sum(~flags), k), rv[~flags], s=3)
        if np.any(flags):
            axes[0].scatter(np.full(np.sum(flags), k), rv[flags], s=8, marker="x")
    axes[0].set_xlabel("Lanczos step k")
    axes[0].set_ylabel("Ritz value")
    axes[0].set_title("Ritz values and exact-spectrum ghost classification")
    axes[1].plot(range(1, result.m + 1), exact_counts, label="exact-spectrum classifier")
    axes[1].plot(range(1, result.m + 1), heuristic_counts, linestyle="--", label="heuristic classifier")
    axes[1].set_xlabel("Lanczos step k")
    axes[1].set_ylabel("ghost count")
    axes[1].legend()
    fig.savefig(figures / "experiment_03_ghost_eigenvalues.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
