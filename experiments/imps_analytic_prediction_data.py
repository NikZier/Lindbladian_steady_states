"""Bundle everything `imps_analytic_prediction_summary.png` is built from into
one pickle, so the figure can be redrawn without touching `_cache/` again.

Mirrors the data assembly in
`experiments/notebooks/analytic_prediction_figures.ipynb` exactly (same
sources, same functions) -- this script performs no TEBD or ED, only re-reads
already-cached runs and re-derives `q` from the fixed seeds in
`renyi2_swssb.draw_samples()`.

Output: experiments/results/imps_analytic_prediction_summary_data.pkl, a dict
    q_at            : {epsilon: [q per sample]},            q = |<11|L''|00>|^2
    inf_values       : {(epsilon, sample): R(r=100)}         iTEBD, infinite system
    finite_values    : {(sample, N): R(N/4, 3N/4)}            |neel>, chi=128
    finite_baseline  : {N: R}                                 L''=0 finite control
    infinite_baseline: dict returned by figs.infinite_baseline() (worst-of-3
                        L''=0 infinite control, full R(r) profile + all_runs)
    samples, finite_n, chi, r_inf, epsilons : the grid constants (figs.* at
                        the time this was built)
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import renyi2_swssb as ex
import imps_summary_figures as figs

OUT_PATH = os.path.join(ex.RESULTS_DIR, "imps_analytic_prediction_summary_data.pkl")


def q_of(eps: float) -> list[float]:
    old = ex.EPSILON
    try:
        ex.EPSILON = eps
        samples = ex.draw_samples()
    finally:
        ex.EPSILON = old
    return [abs(s["L_pp"][3, 0]) ** 2 for s in samples]


def main() -> None:
    results_path = os.path.join(ex.RESULTS_DIR, "imps_eps_init_grids.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)["results"]

    q_at = {eps: q_of(eps) for eps in figs.EPSILONS}

    # sanity check against a cached finite-N run's stored L'' (bit-for-bit)
    check_path = os.path.join(ex.CACHE_DIR, "chi128_sample3_zero_N12_chi128.pkl")
    with open(check_path, "rb") as f:
        q_cached = abs(pickle.load(f)["L_pp"][3, 0]) ** 2
    assert abs(q_cached - q_at[0.20][3]) < 1e-12, "q reconstruction does not match cache"

    inf_values = figs.infinite_values(results)
    finite_values = {
        (s, N): figs.finite_value(s, N)
        for s in figs.SAMPLES for N in figs.FINITE_N
    }
    finite_baseline = {N: figs.finite_baseline(N) for N in figs.FINITE_N}
    infinite_baseline = figs.infinite_baseline()

    data = {
        "q_at": q_at,
        "inf_values": inf_values,
        "finite_values": finite_values,
        "finite_baseline": finite_baseline,
        "infinite_baseline": infinite_baseline,
        "samples": figs.SAMPLES,
        "finite_n": figs.FINITE_N,
        "chi": figs.CHI,
        "r_inf": figs.R_INF,
        "epsilons": figs.EPSILONS,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
