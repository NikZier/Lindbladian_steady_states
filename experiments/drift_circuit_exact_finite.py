"""SWSSB of the driven classical circuit at finite N, by EXACT sector solves.

The model (identical to renyi2_drift_annihilation.py, the "second model")
-------------------------------------------------------------------------
    |10> -> p |01> + (1-p) |10>      L_R = sqrt(p)   |01><10|
    |01> -> p |01> + (1-p) |10>      L_L = sqrt(1-p) |10><01|
    |00> -> |00>
    |11> -> |00>                     L_A =           |00><11|
    L''  = random on the Z(x)Z commutant, ||L''|| = epsilon

on an OPEN chain, p = 0.8 (charge drifts right), rates 1, the same ten L''
seeds as every other study. N in {4, 6, 8, 10, 12}, epsilon in
{0.20, 0.15, 0.10, 0.05}, plus L''=0 controls and a p = 1/2 (unbiased)
comparison at epsilon = 0.2.

Why exact, not TEBD
-------------------
The finite TEBD grid of this model is known to measure a transient (CLAUDE.md,
"RETRACTED 2026-08-13" and Trap 5): relaxation needs ~12/q = 540-6800 time
units, ~60 h per N = 12 run. lindblad_mps.sector_solver solves L(rho) = 0
directly in the (+,+) sector -- no truncation, no schedule -- so the result
is THE steady state of the open chain. Its error is the GMRES residual
||L rho|| / ||rho||, recorded per run, and it is validated against the dense
exact.sector_steady_state at N = 4, 6 (reported in the output).

Initial states do not exist here, so neither does the zero/neel comparison:
the sector steady state is unique (asserted by the bordered system being
solvable) and both starts at N in {4, 8, 12} lie in this sector. At N = 6, 10
the Neel state has odd parity and would belong to the (-,-) sector -- a
different steady state, not computed.

Outputs (experiments/results/):
    drift_circuit_exact_finite.pkl
    drift_circuit_exact_finite_RvsN.png       R/q vs N, per epsilon and sample
    drift_circuit_exact_finite_profiles.png   full R(i, j) at N = 12, density
    drift_circuit_exact_finite_vs_tebd.png    exact vs the cached TEBD grid
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lindblad_mps import blas, exact, models, observables, sector_solver
import renyi2_swssb as ex

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
P_RIGHT = 0.8
P_UNBIASED = 0.5
HOP_RATE = 1.0
ANNIHILATION_RATE = 1.0
SIZES = [4, 6, 8, 10, 12, 14]
EPSILONS = [0.20, 0.15, 0.10, 0.05]
N_WORKERS = 6            # an N=12 GMRES holds ~restart x 67 MB of Krylov vectors
# At N=14 one Krylov vector is 8192^2 complex = 1.07 GB: restart=60 would need
# ~65 GB, so N >= BIG_N runs GMRES(10) (~12 GB of basis, ~16 GB peak), one
# solve at a time, in BIG_PRIORITY order so the key panels fill in first.
BIG_N = 14
BIG_RESTART = 10
BIG_PRIORITY = [(0.8, 0.20), (0.8, 0.05), (0.5, 0.20), (0.8, 0.10), (0.8, 0.15),
                (0.8, 0.0), (0.5, 0.0)]
GMRES_RTOL = 1e-11
KIND = "driftexact"

# Infinite-system R/q at p=0.8 (imps_drift_eps_grids.py, mean over samples).
R_OVER_Q_INFINITE = {0.20: 3.887955, 0.15: 3.937483, 0.10: 3.973188, 0.05: 3.994284}


def baseline_ops(p: float) -> list[np.ndarray]:
    """Bond jumps of the unperturbed circuit at drift probability p."""
    return models.classical_drift_annihilation_jump_operators(p, HOP_RATE, ANNIHILATION_RATE)


def draw_at(epsilon: float) -> list[dict]:
    """The ten standard L'' draws, rescaled to norm epsilon (same seeds)."""
    original = ex.EPSILON
    try:
        ex.EPSILON = epsilon
        return ex.draw_samples()
    finally:
        ex.EPSILON = original


def creation_amplitude(L_pp: np.ndarray) -> float:
    """q = |<11|L''|00>|^2, the pair-creation rate."""
    return float(abs(L_pp[0b11, 0b00]) ** 2)


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------
def job_key(job: dict) -> str:
    return (f"{KIND}_p{job['p']:.2f}_eps{job['epsilon']:.2f}_{job['label']}"
            f"_N{job['N']}")


def solve(job: dict) -> dict:
    """Exact (+,+) steady state of one (p, epsilon, sample, N) and its diagnostics.

    Output: dict with 'R' (R(N//4, 3N//4)), 'R_matrix' ((N, N), R(i, j) for
        i != j, NaN on the diagonal), 'density' (<n_r>), 'purity', 'lambda_min',
        'negative_weight', 'residual', 'hermiticity', 'iterations', 'seconds'.
    """
    N = job["N"]
    terms = ex.build_L2_terms(job["L_pp"], baseline_ops(job["p"]))
    t0 = time.perf_counter()
    Lv = sector_solver.SectorLiouvillian(terms, N)
    restart = BIG_RESTART if N >= BIG_N else 60
    sol = Lv.steady_state(rtol=GMRES_RTOL, restart=restart)
    del Lv.pos
    rho, cfg = sol["rho"], Lv.configs

    Rm = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(i + 1, N):
            Rm[i, j] = Rm[j, i] = sector_solver.renyi2_xx(rho, cfg, N, i, j)
    i, j = ex.correlator_sites(N)

    pops = np.real(np.diag(rho))
    bits = (cfg[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1
    # Big solves run alone, so the O(n^3) eigensolve may use every core.
    with blas.limit_threads(os.cpu_count() if N >= BIG_N else 1):
        evals = np.linalg.eigvalsh(rho)
    return {
        "R": float(Rm[i, j]), "i": i, "j": j, "R_matrix": Rm,
        "density": pops @ bits,
        "purity": float(np.vdot(rho, rho).real),
        "lambda_min": float(evals.min()),
        "negative_weight": float(-evals[evals < 0].sum()),
        "residual": sol["residual"], "hermiticity": sol["hermiticity"],
        "iterations": sol["iterations"], "gmres_restart": restart,
        "seconds": time.perf_counter() - t0,
    }


def run_job(job: dict) -> dict:
    """Worker entry: cached, exception-safe wrapper around solve()."""
    path = os.path.join(ex.CACHE_DIR, job_key(job) + ".pkl")
    config = {"gmres_rtol": GMRES_RTOL, "hop_rate": HOP_RATE,
              "annihilation_rate": ANNIHILATION_RATE}
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                out = pickle.load(f)
            if out.get("run_config") == config:
                out["cached"] = True
                return out
        except Exception:
            pass
    out = {k: v for k, v in job.items() if k != "L_pp"}
    out["cached"] = False
    out["run_config"] = config
    try:
        with blas.limit_threads(1):
            out["result"] = solve(job)
    except Exception:
        out["error"] = traceback.format_exc()
        return out
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return out


def build_jobs() -> list[dict]:
    jobs = []
    for eps in EPSILONS:
        for s, samp in enumerate(draw_at(eps)):
            for N in SIZES:
                jobs.append({"p": P_RIGHT, "epsilon": eps, "label": f"sample{s}",
                             "sample": s, "L_pp": samp["L_pp"], "N": N,
                             "q": creation_amplitude(samp["L_pp"])})
    for s, samp in enumerate(draw_at(0.20)):
        for N in SIZES:
            jobs.append({"p": P_UNBIASED, "epsilon": 0.20, "label": f"sample{s}",
                         "sample": s, "L_pp": samp["L_pp"], "N": N,
                         "q": creation_amplitude(samp["L_pp"])})
    for p in (P_RIGHT, P_UNBIASED):
        for N in SIZES:
            jobs.append({"p": p, "epsilon": 0.0, "label": "ctrl", "sample": None,
                         "L_pp": None, "N": N, "q": 0.0})
    small = sorted((j for j in jobs if j["N"] < BIG_N), key=lambda j: j["N"], reverse=True)
    big = sorted((j for j in jobs if j["N"] >= BIG_N),
                 key=lambda j: (BIG_PRIORITY.index((j["p"], j["epsilon"])),
                                j["sample"] if j["sample"] is not None else 0))
    return small + big
    return jobs


def dense_validation() -> list[tuple]:
    """Sector solver vs exact.sector_steady_state, all samples, N = 4 and 6."""
    rows = []
    for N in (4, 6):
        for s, samp in enumerate(draw_at(0.20)):
            terms = ex.build_L2_terms(samp["L_pp"], baseline_ops(P_RIGHT))
            ref = exact.sector_steady_state([], [], terms, [], N)["rho"]
            Lv = sector_solver.SectorLiouvillian(terms, N)
            rho = sector_solver.embed_sector(Lv.steady_state()["rho"], Lv.configs, N)
            i, j = ex.correlator_sites(N)
            Rd = observables.renyi2_correlator_dense(ref, models.X, i, j, N)
            Rs = observables.renyi2_correlator_dense(rho, models.X, i, j, N)
            rows.append((N, s, float(np.linalg.norm(rho - ref)), abs(Rs - Rd) / abs(Rd)))
    return rows


# ---------------------------------------------------------------------------
# Assembly, report, plots
# ---------------------------------------------------------------------------
def run_grid() -> dict:
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    print("Validating the sector solver against the dense exact solve ...", flush=True)
    val = dense_validation()
    print(f"  worst |rho - rho_dense| = {max(r[2] for r in val):.2e}, "
          f"worst rel. error in R = {max(r[3] for r in val):.2e} "
          f"({len(val)} cases, N = 4, 6)\n", flush=True)

    jobs = build_jobs()
    print(f"{len(jobs)} exact solves: N < {BIG_N} on {N_WORKERS} workers, "
          f"N >= {BIG_N} one at a time", flush=True)
    res, failures = {}, []
    t0 = time.perf_counter()

    def outcomes():
        small = [j for j in jobs if j["N"] < BIG_N]
        with mp.Pool(N_WORKERS) as pool:
            yield from pool.imap_unordered(run_job, small)
        # A fresh process per solve, so each ~16 GB peak is returned to the OS.
        with mp.Pool(1, maxtasksperchild=1) as pool:
            for j in jobs[len(small):]:
                yield pool.apply(run_job, (j,))

    for k, out in enumerate(outcomes(), 1):
        tag = (f"  [{k:3d}/{len(jobs)}] p={out['p']:.1f} eps={out['epsilon']:.2f} "
               f"{out['label']:>8s} N={out['N']:2d}")
        if "error" in out:
            failures.append(out)
            print(f"{tag} FAILED {out['error'].strip().splitlines()[-1]}", flush=True)
            continue
        r = out["result"]
        res[(out["p"], out["epsilon"], out["sample"], out["N"])] = {**r, "q": out["q"]}
        print(f"{tag} R={r['R']:.6e} res={r['residual']:.1e} it={r['iterations']:3d} "
              f"lmin={r['lambda_min']:+.1e} "
              f"[{'cached' if out['cached'] else '%.0fs' % r['seconds']}, "
              f"{(time.perf_counter()-t0)/60:.1f} min]", flush=True)
    return {"config": {"p": P_RIGHT, "p_unbiased": P_UNBIASED, "sizes": SIZES,
                       "epsilons": EPSILONS, "hop_rate": HOP_RATE,
                       "annihilation_rate": ANNIHILATION_RATE,
                       "base_seed": ex.BASE_SEED, "gmres_rtol": GMRES_RTOL,
                       "site_rule": "i = N//4, j = 3N//4"},
            "validation": val, "results": res,
            "failures": [f["error"] for f in failures]}


def cached_tebd(sample: int, init: str, N: int) -> float | None:
    """R from the old 55.5-unit TEBD grid (eps=0.2, p=0.8, chi=128), if cached."""
    path = os.path.join(ex.CACHE_DIR, f"drift_sample{sample}_{init}_N{N}_chi128.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["result"]["correlator"]


def summarize(grid: dict) -> None:
    res = grid["results"]
    for p, epss in ((P_RIGHT, EPSILONS), (P_UNBIASED, [0.20])):
        for eps in epss:
            print(f"\n=== p={p}, eps={eps}: R(N/4, 3N/4) / q  (infinite p=0.8: "
                  f"{R_OVER_Q_INFINITE.get(eps, float('nan')):.4f}) ===")
            print(f"{'s':>3} {'q':>10} " + " ".join(f"{'N=%d' % N:>9}" for N in SIZES))
            for s in range(10):
                row = [res.get((p, eps, s, N)) for N in SIZES]
                if row[0] is None:
                    continue
                print(f"{s:>3} {row[0]['q']:>10.3e} "
                      + " ".join(f"{r['R'] / r['q']:>9.4f}" if r else f"{'-':>9}"
                                 for r in row))
            means = []
            for N in SIZES:
                v = np.array([res[(p, eps, s, N)]["R"] / res[(p, eps, s, N)]["q"]
                              for s in range(10) if (p, eps, s, N) in res])
                means.append((v.mean(), (v.max() - v.min()) / v.mean()) if v.size
                             else (np.nan, np.nan))
            print(f"{'mean':>14} " + " ".join(f"{m:>9.4f}" for m, _ in means))
            print(f"{'spread':>14} " + " ".join(f"{100*sp:>8.2f}%" for _, sp in means))

    vals = [r for r in res.values()]
    print(f"\nworst residual {max(r['residual'] for r in vals):.1e}, "
          f"worst lambda_min {min(r['lambda_min'] for r in vals):+.1e}")
    for p in (P_RIGHT, P_UNBIASED):
        c = [res[(p, 0.0, None, N)] for N in SIZES if (p, 0.0, None, N) in res]
        print(f"L''=0 control p={p}: max |R| {max(abs(r['R']) for r in c):.1e}, "
              f"purities {[round(r['purity'], 9) for r in c]}")

    print("\nexact vs cached TEBD (eps=0.2, p=0.8, 55.5 units): exact/TEBD")
    for N in (4, 8, 12):
        ratios = [res[(P_RIGHT, 0.2, s, N)]["R"] / t for s in range(10)
                  if (t := cached_tebd(s, "zero", N)) is not None]
        if ratios:
            print(f"  N={N:2d}: {min(ratios):.4f} ... {max(ratios):.4f}")


def plot_all(grid: dict) -> list[str]:
    plt = ex._mpl()
    res = grid["results"]
    cmap = plt.get_cmap("viridis")
    paths = []

    # R/q vs N
    fig, axes = plt.subplots(1, len(EPSILONS) + 1, figsize=(4.2 * (len(EPSILONS) + 1), 4.2),
                             sharey=True)
    panels = [(P_RIGHT, e) for e in EPSILONS] + [(P_UNBIASED, 0.20)]
    for ax, (p, eps) in zip(axes, panels):
        for s in range(10):
            pts = [(N, res[(p, eps, s, N)]["R"] / res[(p, eps, s, N)]["q"])
                   for N in SIZES if (p, eps, s, N) in res]
            ax.plot(*zip(*pts), "-o", color=cmap(s / 9), ms=4, lw=1.2)
        if p == P_RIGHT:
            ax.axhline(R_OVER_Q_INFINITE[eps], ls="--", color="crimson",
                       label="iTEBD (N=inf)")
            ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(SIZES)
        ax.set_xlabel("N")
        ax.set_title(rf"$p={p}$, $\epsilon={eps}$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$R(N/4,3N/4)\,/\,q$")
    fig.suptitle("Driven classical circuit, open chain: exact (+,+) steady state")
    fig.tight_layout()
    paths.append(os.path.join(ex.RESULTS_DIR, "drift_circuit_exact_finite_RvsN.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # Profiles R(i, r) at N=12, eps=0.2, and density
    N = max(n for n in SIZES if all((p, 0.2, s, n) in res
                                     for p in (P_RIGHT, P_UNBIASED) for s in range(10)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for p, ax in ((P_RIGHT, axes[0]), (P_UNBIASED, axes[1])):
        for s in range(10):
            r = res.get((p, 0.2, s, N))
            if r is None:
                continue
            i = r["i"]
            ax.plot(range(i + 1, N), r["R_matrix"][i, i + 1:] / r["q"], "-o",
                    color=cmap(s / 9), ms=4)
        ax.set_yscale("log")
        ax.set_xlabel(f"site r  (i = {N // 4})")
        ax.set_ylabel(r"$R(i,r)/q$")
        ax.set_title(rf"$N={N}$, $p={p}$, $\epsilon=0.2$")
        ax.grid(True, alpha=0.3)
    for p, ls in ((P_RIGHT, "-"), (P_UNBIASED, "--")):
        for s in range(10):
            r = res.get((p, 0.2, s, N))
            if r is not None:
                axes[2].plot(range(N), r["density"] / r["q"], ls, color=cmap(s / 9))
    axes[2].set_yscale("log")
    axes[2].set_xlabel("site r")
    axes[2].set_ylabel(r"$\langle n_r\rangle / q$")
    axes[2].set_title("density (solid p=0.8, dashed p=0.5)")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    paths.append(os.path.join(ex.RESULTS_DIR, "drift_circuit_exact_finite_profiles.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def main() -> None:
    grid = run_grid()
    path = os.path.join(ex.RESULTS_DIR, "drift_circuit_exact_finite.pkl")
    with open(path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\nSaved grid -> {path}", flush=True)
    for name, fn in (("summary", summarize), ("plots", plot_all)):
        try:
            out = fn(grid)
            if out:
                print(f"Saved {name} -> {out}", flush=True)
        except Exception:
            print(f"!! {name} failed:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
