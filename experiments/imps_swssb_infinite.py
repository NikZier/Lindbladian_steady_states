"""The thermodynamic-limit Renyi-2 correlator and correlation length, via iTEBD.

Why
---
Two open problems in the finite-N study (renyi2_swssb*.py), both diagnosed
this session:

  1. Relaxation time grows with N (the long-schedule run confirmed sample 8's
     zero/neel straddle shrinks with more evolved time but doesn't fully
     close even at N=20 with 3.6x the schedule).
  2. Boundary contamination: some of the N=20 profile's fall/rise is edge
     effect entangled with under-relaxation, and a finite chain can't
     separate them.

Both dissolve in the thermodynamic limit: an infinite, translation-invariant
chain has one fixed relaxation time (the Liouvillian gap) independent of any
system size, and no boundary at all. The perturbed model is gapped there
(R = 0.1256*|<11|L''|00>|^2 and the same matrix element sets the gap, both
measured this session across all 10 samples), so this isn't chasing a
gapless target.

This script runs lindblad_mps.itebd.find_steady_state_infinite to the steady
state for three L'' samples, both initial states, at a chi sweep, then
measures R(r) to r=100 and the correlation length via lindblad_mps.iobservables.

Samples: 8 (smallest R, slowest finite-size relaxation), 6 (largest R),
0 (has existing finite chi-convergence data at N=20).
Both inits ('zero', 'neel'): same uniqueness cross-check used everywhere else
in this repo -- a unique steady state in the sector forces the same R from
either start; both fit the 2-site unit cell exactly ('neel' is exactly why a
2-site cell is required at all).

Reading the output
-------------------
xi (correlation length) should PLATEAU as chi grows, not keep increasing --
that is itself a physics check (a genuinely gapped state has finite xi; a
critical/gapless one would show xi growing without bound as chi is relaxed).
R(r) at a fixed chi should also plateau in r for large r (flat, not decaying)
if the sample shows SWSSB, exactly as in the finite-N profiles.

Cache is namespaced 'inf_...', cannot alias the finite grid's 'chi128_...'
cache (different physics, different truncation regime).

Outputs (experiments/results/):
    imps_swssb_infinite.pkl -- the grid, plus the finite N=20 chi=128
        reference points where available.
    imps_swssb_infinite.png -- R(r) vs r per chi, xi vs chi, and one overlay
        against the finite N=20 profile.
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import renyi2_swssb as ex
from lindblad_mps import imps, iobservables, itebd, models

SAMPLES = [8, 6, 0]
INITS = ["zero", "neel"]
CHI_LIST = [64, 128, 256]
R_MAX = 100
DT_SCHEDULE = [0.1, 0.05, 0.02, 0.01, 0.005]  # same as the finite headline run
STEPS_PER_DT = 300
CANONICALIZE_EVERY = 10
KIND = "inf"

FINITE_KIND = "chi128"
FINITE_N = 20
FINITE_CHI = 128

REFERENCE_R = [1, 5, 20, 50, 100]  # r values printed in the chi-convergence table


def build_initial_state(name: str) -> "imps.iMPS":
    """Bond-dim-1 iMPS for the named 2-periodic initial state.

    Input: name, 'zero' (|00...>) or 'neel' (|0101...>).
    Output: an imps.iMPS.
    """
    KET0 = np.array([1, 0], dtype=complex)
    KET1 = np.array([0, 1], dtype=complex)
    if name == "zero":
        return imps.iMPS.pure_product_state(KET0, KET0)
    if name == "neel":
        return imps.iMPS.pure_product_state(KET0, KET1)
    raise ValueError(f"unknown initial state {name!r}")


def job_key(job: dict) -> str:
    """Filename-safe cache key for one (sample, init, chi) job."""
    return f"{KIND}_sample{job['sample']}_{job['init']}_chi{job['chi_max']}"


def run_config() -> dict:
    """Settings that change a run's result but are not covered by job_key()."""
    return {
        "dt_schedule": list(DT_SCHEDULE),
        "steps_per_dt": STEPS_PER_DT,
        "canonicalize_every": CANONICALIZE_EVERY,
        "cutoff": ex.CUTOFF,
        "r_max": R_MAX,
    }


def run_job(job: dict) -> dict:
    """Execute one iTEBD steady-state run (worker entry point).

    Must be module-level to survive pickling to worker processes (Windows
    spawn). Every completed job is cached before returning; a cached job with
    a matching run_config is returned without recomputing.

    Input: job dict with 'sample', 'L_pp', 'init', 'chi_max'.
    Output: job dict augmented with 'result' (dict of correlator/correlation-
        length/diagnostics, no live iMPS) and 'seconds', or 'error'.
    """
    path = os.path.join(ex.CACHE_DIR, job_key(job) + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                out = pickle.load(f)
            if out.get("run_config") == run_config():
                out["cached"] = True
                return out
        except Exception:
            pass

    t0 = time.perf_counter()
    out = dict(job)
    out["cached"] = False
    out["run_config"] = run_config()
    try:
        L2_terms = ex.build_L2_terms(job["L_pp"])
        state, history = itebd.find_steady_state_infinite(
            H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[],
            dt_schedule=DT_SCHEDULE, steps_per_dt=STEPS_PER_DT,
            chi_max=job["chi_max"], cutoff=ex.CUTOFF,
            canonicalize_every=CANONICALIZE_EVERY,
            initial_state=build_initial_state(job["init"]),
        )
        profile = iobservables.correlator_profile(state, models.X, r_max=R_MAX)
        xi_diag = iobservables.correlation_length(state)
        out["result"] = {
            "profile": profile,
            "xi": xi_diag["xi"],
            "eta1": xi_diag["eta1"],
            "eta2": xi_diag["eta2"],
            "bond_dims": dict(state.bond_dims),
            "max_discarded_weight": max(history["discarded_weight"], default=0.0),
            "final_eigenvalue_drift": (
                history["eigenvalue_drift"][-1] if history["eigenvalue_drift"] else float("nan")
            ),
        }
    except Exception:
        out["error"] = traceback.format_exc()
    out["seconds"] = time.perf_counter() - t0

    if "error" not in out:
        os.makedirs(ex.CACHE_DIR, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f)
        os.replace(tmp, path)
    return out


def build_jobs(samples: list[dict]) -> list[dict]:
    """Enumerate the SAMPLES x INITS x CHI_LIST grid, longest job first.

    Input: samples, from ex.draw_samples().
    Output: list of job dicts for run_job(), sorted by descending estimated
        cost (chi**3, since iTEBD cost per canonicalize()/step is chi^3-ish
        regardless of "system size" -- there is none).
    """
    jobs = [
        {"sample": s, "L_pp": samples[s]["L_pp"], "init": init, "chi_max": chi}
        for s in SAMPLES
        for init in INITS
        for chi in CHI_LIST
    ]
    jobs.sort(key=lambda j: j["chi_max"] ** 3, reverse=True)
    return jobs


def load_finite_reference(sample: int, init: str) -> dict | None:
    """Load the existing finite N=20, chi=128 result for the same sample/init.

    Output: the cached result dict, or None if that run isn't cached.
    """
    path = os.path.join(
        ex.CACHE_DIR, f"{FINITE_KIND}_sample{sample}_{init}_N{FINITE_N}_chi{FINITE_CHI}.pkl"
    )
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["result"]


def run_grid() -> dict:
    """Run every (sample, init, chi) job on a worker pool.

    Output: pickle-ready dict with the grid and finite-N reference points.
    """
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()
    jobs = build_jobs(samples)

    grid = {
        "config": {
            "samples": SAMPLES, "inits": INITS, "chi_list": CHI_LIST, "r_max": R_MAX,
            "dt_schedule": DT_SCHEDULE, "steps_per_dt": STEPS_PER_DT,
            "canonicalize_every": CANONICALIZE_EVERY, "epsilon": ex.EPSILON,
        },
        "descriptions": {s: samples[s]["description"] for s in SAMPLES},
        "runs": {},  # (sample, init, chi) -> result
        "finite_reference": {
            (s, init): load_finite_reference(s, init) for s in SAMPLES for init in INITS
        },
    }

    print(f"{len(jobs)} iTEBD runs queued: samples {SAMPLES} x inits {INITS} x "
          f"chi in {CHI_LIST}, r_max={R_MAX}.", flush=True)
    print("Expect chi=256 runs to take a while (canonicalize() is ARPACK-"
          "dominated) -- exact per-call cost not yet measured at production "
          "chi; watch the timing on the first few and extrapolate.\n", flush=True)

    t0 = time.perf_counter()
    done, failures = 0, []
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_job, jobs):
            done += 1
            tag = f"  [{done:2d}/{len(jobs)}] sample{out['sample']} {out['init']:>4s} chi={out['chi_max']:3d}"
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  {out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            res = out["result"]
            key = (out["sample"], out["init"], out["chi_max"])
            grid["runs"][key] = res
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
            xi_str = f"{res['xi']:.2f}" if res["xi"] is not None else "None"
            maxbond = max(res["bond_dims"].values())
            print(f"{tag}  R(1)={res['profile'][0][1]:.4e}  R({R_MAX})={res['profile'][-1][1]:.4e}  "
                  f"xi={xi_str}  bond={maxbond:3d}/{out['chi_max']}"
                  f"{'!' if maxbond >= out['chi_max'] else ' '}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    grid["failures"] = [{k: v for k, v in f.items() if k != "L_pp"} for f in failures]
    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t0)/60:.1f} min.", flush=True)
    for f in failures:
        print(f"  FAILED: {job_key(f)}", flush=True)
    return grid


def summarize(grid: dict) -> None:
    """Print the chi-convergence table (R at reference r, xi) per sample/init."""
    cfg = grid["config"]
    print(f"\n{'='*90}\niTEBD thermodynamic-limit grid\n{'='*90}")
    for s in cfg["samples"]:
        for init in cfg["inits"]:
            rows = [grid["runs"].get((s, init, chi)) for chi in cfg["chi_list"]]
            if all(r is None for r in rows):
                continue
            print(f"\nsample {s}, init |{init}>:")
            header = f"  {'chi':>5}  " + "  ".join(f"R(r={r:3d})" for r in REFERENCE_R) + "      xi      bond"
            print(header)
            for chi, res in zip(cfg["chi_list"], rows):
                if res is None:
                    print(f"  {chi:>5}  (missing)")
                    continue
                prof = dict(res["profile"])
                r_vals = "  ".join(f"{prof.get(r, float('nan')):.3e}" for r in REFERENCE_R)
                xi_str = f"{res['xi']:7.2f}" if res["xi"] is not None else "   None"
                maxbond = max(res["bond_dims"].values())
                flag = "!" if maxbond >= chi else " "
                print(f"  {chi:>5}  {r_vals}  {xi_str}  {maxbond:4d}/{chi}{flag}")

            ref = grid["finite_reference"].get((s, init))
            if ref is not None:
                i, j = ref["i"], ref["j"]
                print(f"  (finite N={FINITE_N}, chi={FINITE_CHI} reference: "
                      f"R({i},{j}) = {ref['correlator']:.4e} at separation {j-i} -- "
                      f"not expected to match exactly, different truncation/"
                      f"relaxation error sources)")

    print("\nxi should plateau (not grow) with chi if the sample is genuinely "
          "gapped; R(r) should plateau in r for a sample showing SWSSB.",
          flush=True)


def plot_grid(grid: dict) -> str:
    """Plot R(r) vs r per chi, xi vs chi, and one finite-vs-infinite overlay."""
    plt = ex._mpl()
    cfg = grid["config"]
    cmap = plt.get_cmap("viridis")
    n_chi = len(cfg["chi_list"])
    panels = [(s, init) for s in cfg["samples"] for init in cfg["inits"]]

    fig, axes = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 9), squeeze=False)
    for col, (s, init) in enumerate(panels):
        ax0, ax1 = axes[0][col], axes[1][col]
        xis, chis_with_xi = [], []
        for k, chi in enumerate(cfg["chi_list"]):
            res = grid["runs"].get((s, init, chi))
            if res is None:
                continue
            r = [p[0] for p in res["profile"]]
            v = [p[1] for p in res["profile"]]
            ax0.plot(r, v, "-", color=cmap(k / max(n_chi - 1, 1)), lw=1.8, label=f"chi={chi}")
            if res["xi"] is not None:
                xis.append(res["xi"])
                chis_with_xi.append(chi)
        ax0.set_yscale("log")
        ax0.set_xlabel("separation r")
        ax0.set_ylabel(r"$R(0,r)$")
        ax0.set_title(f"sample {s}, |{init}>")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=7)

        ax1.plot(chis_with_xi, xis, "-o", color="crimson")
        ax1.set_xlabel("bond dimension chi")
        ax1.set_ylabel(r"correlation length $\xi$")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("plateau = gapped, growth = gapless")

    fig.suptitle(rf"iTEBD thermodynamic limit ($\epsilon={cfg['epsilon']}$), "
                 f"samples {cfg['samples']}")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_swssb_infinite.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_grid()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_swssb_infinite.pkl")
    with open(path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\nSaved grid -> {path}", flush=True)

    try:
        summarize(grid)
    except Exception:
        print(f"!! summary failed (data is still saved):\n{traceback.format_exc()}", flush=True)
    try:
        print(f"Saved plot -> {plot_grid(grid)}", flush=True)
    except Exception:
        print(f"!! plot failed (data is still saved):\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
