from __future__ import annotations

from pathlib import Path

import numpy as np

from krylov_lanczos import (
    controlled_start_vector,
    lanczos,
    ritz_pair_at_step,
    sin_angle,
    theory_matrix,
)
from common import first_below, save_summary, setup_matplotlib, single_threaded

N = 400
SEED = 13
M_MAX = 110
TOL_EV = 1e-8
TOL_VEC = 1e-10
C_TARGET = 287.0
ANGLE_DEG = float(np.degrees(np.arctan(np.sqrt(C_TARGET))))


@single_threaded
def run(output_dir: str | Path = "results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    A, spectrum, Q = theory_matrix(N, SEED)
    idx_iso = 0
    idx_c = int(np.argmin(np.abs(spectrum - 4.0)))
    z_iso, z_c = Q[:, idx_iso], Q[:, idx_c]
    v1 = controlled_start_vector(z_iso, angle_deg=ANGLE_DEG, seed=SEED)
    result = lanczos(A, v1, M_MAX, reorthogonalization="full")
    spread = spectrum[-1] - spectrum[0]
    delta_iso = float(np.min(np.abs(np.delete(spectrum, idx_iso) - spectrum[idx_iso])))
    delta_c = float(np.min(np.abs(np.delete(spectrum, idx_c) - spectrum[idx_c])))

    err_iso, err_c, sin_iso, sin_c, dk_iso, dk_c = [], [], [], [], [], []
    for m in range(1, result.m + 1):
        ti, yi, _ = ritz_pair_at_step(result, m, spectrum[idx_iso])
        tc, yc, _ = ritz_pair_at_step(result, m, spectrum[idx_c])
        ri = np.linalg.norm(A @ yi - ti * yi)
        rc = np.linalg.norm(A @ yc - tc * yc)
        err_iso.append(abs(ti - spectrum[idx_iso]) / spread)
        err_c.append(abs(tc - spectrum[idx_c]) / spread)
        sin_iso.append(sin_angle(z_iso, yi))
        sin_c.append(sin_angle(z_c, yc))
        dk_iso.append(ri / delta_iso)
        dk_c.append(rc / delta_c)

    err_iso = np.asarray(err_iso); err_c = np.asarray(err_c)
    sin_iso = np.asarray(sin_iso); sin_c = np.asarray(sin_c)
    dk_iso = np.asarray(dk_iso); dk_c = np.asarray(dk_c)
    conv_e_iso = first_below(err_iso, TOL_EV)
    conv_e_c = first_below(err_c, TOL_EV)
    conv_v_iso = first_below(sin_iso, TOL_VEC)
    conv_v_c = first_below(sin_c, TOL_VEC)

    summary = {
        "n": N, "seed": SEED, "m_max": result.m,
        "delta_iso": delta_iso, "delta_c": delta_c,
        "conv_eigenvalue_iso": conv_e_iso, "conv_eigenvector_iso": conv_v_iso,
        "conv_eigenvalue_cluster": conv_e_c, "conv_eigenvector_cluster": conv_v_c,
        "final_sin_iso": float(sin_iso[-1]), "final_sin_cluster": float(sin_c[-1]),
        "cluster_vector_delay": None if conv_e_c is None or conv_v_c is None else int(conv_v_c - conv_e_c),
    }
    save_summary(output_dir / "experiment_06b.json", summary)

    plt = setup_matplotlib()
    ms = np.arange(1, result.m + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].semilogy(ms, sin_iso, label="sin angle")
    axes[0].semilogy(ms, dk_iso, linestyle="--", label=r"$\|r\|/\delta$")
    axes[0].semilogy(ms, err_iso, linestyle=":", label="eigenvalue error")
    axes[0].set_title(f"isolated eigenvalue, delta={delta_iso:.2g}")
    axes[0].set_xlabel("m"); axes[0].legend()
    axes[1].semilogy(ms, sin_c, label="sin angle")
    axes[1].semilogy(ms, np.minimum(dk_c, 10.0), linestyle="--", label=r"$\|r\|/\delta_c$")
    axes[1].semilogy(ms, err_c, linestyle=":", label="eigenvalue error")
    axes[1].set_title(f"clustered eigenvalue, delta={delta_c:.1e}")
    axes[1].set_xlabel("m"); axes[1].legend()
    fig.savefig(figures / "experiment_06b_davis_kahan.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(run())
