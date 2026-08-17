"""Decay shape of the Renyi-2 correlator in the driven classical circuit.

Reads the grid produced by renyi2_drift_annihilation.py and answers the one
question the raw numbers leave open: R falls with system size, but does it fall
to zero (no SWSSB) or to a nonzero plateau (SWSSB with a long crossover)?

Three candidates are fitted to R(N) per sample, on the SEPARATION N/2 that the
site rule i = N//4, j = 3N//4 makes the correlator span:

    exponential   R = A exp(-N / (2 xi))     -> no long-range order
    power law     R = A N^(-alpha)           -> critical, also no LRO
    plateau       R = R_inf + A exp(-N/(2 xi))  -> SWSSB if R_inf > floor

Selection is by corrected AIC, which penalizes the plateau's extra parameter --
with five sizes a three-parameter fit will otherwise always win on residuals
alone. The verdict is only quoted where it survives the truncation filter.

The truncation filter is the point of this script as much as the fitting.
Trap 1 of CLAUDE.md: a chi-limited run fabricates exactly the exponential decay
being tested for, so any run whose peak bond dimension is within CHI_MARGIN of
the cap is a LOWER BOUND, not a data point, and is excluded from every fit.
Samples are reported with the count of sizes that survived, and a fit over
fewer than MIN_POINTS surviving sizes is refused rather than reported weak.
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import renyi2_swssb as ex

GRID_PATH = os.path.join(ex.RESULTS_DIR, "renyi2_drift_annihilation.pkl")

# Within 10% of the cap counts as binding (CLAUDE.md Trap 1): the truncation is
# already shaping the state well before the bond dimension exactly equals chi.
CHI_MARGIN = 0.90
MIN_POINTS = 3  # a 2-parameter fit on 2 points is interpolation, not evidence


def usable(res: dict, chi_max: int) -> bool:
    """Whether a run is a data point rather than a lower bound."""
    return res is not None and max(res["final_bond_dims"]) < CHI_MARGIN * chi_max


def _aicc(residuals: np.ndarray, k: int) -> float:
    """Corrected AIC for a least-squares fit with k parameters.

    The correction term is what makes this usable at five data points; plain
    AIC barely penalizes the third parameter at this sample size.
    """
    n = len(residuals)
    rss = float(np.sum(residuals**2))
    if n <= k + 1 or rss <= 0:
        return float("inf")
    return n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)


def fit_shapes(sizes: np.ndarray, R: np.ndarray) -> dict:
    """Fit exponential, power-law and plateau forms to R(N). Fits log R.

    Fitting in log space weights the sizes evenly in relative error, which is
    what matters when R spans a decade; a linear-space fit would be dominated
    by the smallest system.

    Input: sizes, the system sizes; R, the correlator at each (all positive).
    Output: dict of model name -> dict with 'params', 'aicc', and for the
        plateau also 'R_inf' and its relative size against R at the largest N.
    """
    from scipy.optimize import curve_fit

    logR = np.log(R)
    out = {}

    # exponential: log R = log A - N / (2 xi)
    slope, intercept = np.polyfit(sizes, logR, 1)
    out["exponential"] = {
        "params": {"A": float(np.exp(intercept)),
                   "xi": float(-1.0 / (2.0 * slope)) if slope < 0 else float("inf")},
        "aicc": _aicc(logR - (slope * sizes + intercept), 2),
    }

    # power law: log R = log A - alpha log N
    s2, i2 = np.polyfit(np.log(sizes), logR, 1)
    out["power_law"] = {
        "params": {"A": float(np.exp(i2)), "alpha": float(-s2)},
        "aicc": _aicc(logR - (s2 * np.log(sizes) + i2), 2),
    }

    # plateau: R = R_inf + A exp(-N / (2 xi)), fitted in log space
    def model(N, R_inf, A, inv_xi):
        return np.log(np.abs(R_inf) + np.abs(A) * np.exp(-inv_xi * N))

    try:
        p0 = (R[-1] * 0.5, R[0], -2.0 * slope)
        popt, _ = curve_fit(model, sizes, logR, p0=p0, maxfev=20000)
        R_inf, A, inv_xi = abs(popt[0]), abs(popt[1]), popt[2]
        out["plateau"] = {
            "params": {"R_inf": float(R_inf), "A": float(A),
                       "xi": float(1.0 / (2.0 * inv_xi)) if inv_xi > 0 else float("inf")},
            "aicc": _aicc(logR - model(sizes, R_inf, A, inv_xi), 3),
            "R_inf": float(R_inf),
            "R_inf_over_R_last": float(R_inf / R[-1]),
        }
    except Exception:
        out["plateau"] = {"params": {}, "aicc": float("inf"), "R_inf": 0.0,
                          "R_inf_over_R_last": 0.0}
    return out


def analyse(grid: dict, init: str = "zero") -> list[dict]:
    """Per-sample decay analysis on the truncation-clean subset of the grid."""
    cfg = grid["config"]
    sizes_all, chi, q = cfg["sizes"], cfg["chi_max"], grid["q"]
    rows = []
    for s in range(cfg["n_samples"]):
        pts = [(N, grid["results"][init][N][s]) for N in sizes_all]
        keep = [(N, r["correlator"]) for N, r in pts if usable(r, chi)]
        dropped = [N for N, r in pts if not usable(r, chi)]
        row = {"sample": s, "q": q[s], "sizes": [N for N, _ in keep],
               "R": [v for _, v in keep], "dropped": dropped, "fits": None,
               "best": None}
        if len(keep) >= MIN_POINTS and all(v > 0 for _, v in keep):
            N = np.array([n for n, _ in keep], dtype=float)
            R = np.array([v for _, v in keep], dtype=float)
            row["fits"] = fit_shapes(N, R)
            row["best"] = min(row["fits"], key=lambda k: row["fits"][k]["aicc"])
        rows.append(row)
    return rows


def report(grid: dict, init: str = "zero") -> None:
    """Print the size scaling, the surviving fits and the verdict."""
    cfg = grid["config"]
    rows = analyse(grid, init)
    floor = 0.0
    if grid["control_Lpp0"]:
        floor = max(abs(min(v for _, v in r["profile"]))
                    for r in grid["control_Lpp0"].values())

    print(f"\n=== decay shape, init |{init}>, p={cfg['model']['p']} ===")
    print(f"control floor {floor:.2e}; runs within {100*(1-CHI_MARGIN):.0f}% "
          f"of chi={cfg['chi_max']} excluded as lower bounds\n")
    print(f"{'s':>2} {'q':>10} {'sizes used':>18} {'best fit':>12} "
          f"{'xi':>8} {'R_inf':>11} {'R_inf/R(max N)':>15}")
    for r in rows:
        if r["fits"] is None:
            print(f"{r['sample']:>2} {r['q']:>10.3e} "
                  f"{str(r['sizes']):>18} {'(too few)':>12}")
            continue
        best = r["best"]
        f = r["fits"][best]
        xi = f["params"].get("xi", float("nan"))
        plat = r["fits"]["plateau"]
        print(f"{r['sample']:>2} {r['q']:>10.3e} {str(r['sizes']):>18} "
              f"{best:>12} {xi:>8.2f} {plat['R_inf']:>11.3e} "
              f"{plat['R_inf_over_R_last']:>14.3f}")

    ok = [r for r in rows if r["fits"] is not None]
    if ok:
        votes = {}
        for r in ok:
            votes[r["best"]] = votes.get(r["best"], 0) + 1
        print(f"\n  best-fit shape over {len(ok)} usable samples: {votes}")
        xis = [r["fits"]["exponential"]["params"]["xi"] for r in ok]
        qs = [r["q"] for r in ok]
        if len(ok) >= 3:
            c = np.corrcoef(np.log(qs), np.log(np.abs(xis)))[0, 1]
            print(f"  corr(log q, log xi_exponential) = {c:+.3f}  "
                  f"(positive = stronger perturbation orders further)")
        print(f"\n  A plateau is only SWSSB if R_inf clears the control floor "
              f"{floor:.2e}.")
    print(f"  Samples excluded at some sizes: "
          f"{[(r['sample'], r['dropped']) for r in rows if r['dropped']]}")


def main() -> None:
    if not os.path.exists(GRID_PATH):
        raise SystemExit(f"grid not found: {GRID_PATH} -- run "
                         f"renyi2_drift_annihilation.py first")
    with open(GRID_PATH, "rb") as f:
        grid = pickle.load(f)
    for init in grid["config"]["inits"]:
        report(grid, init)


if __name__ == "__main__":
    main()
