"""The thermodynamic-limit iTEBD grid on a much longer schedule.

Why
---
The standard-schedule grid (imps_swssb_infinite.py, 1500 steps -- the same
schedule the finite-N study already found insufficient for sample 8 even at
N=20) came back visibly unconverged: R(r) profiles are wildly oscillatory
rather than plateauing, don't collapse onto each other as chi grows from 64
to 256, and dip negative in several places (a positivity violation, the same
diagnostic the finite study uses to flag non-convergence). Correlation
lengths don't stabilize with chi either -- sample 0's |neel> case climbs
xi = 17 -> 24 -> 53 as chi grows, the explicit "still relaxing" signature.
correlator_profile itself is independently verified exact on hand-computable
product-state cases (tests/test_imps.py), so this reads as under-relaxation,
not a correctness bug -- the direct infinite-system analogue of the
finite-N under-relaxation problem this whole iTEBD effort was built to avoid.

The fix is the same one that worked there: more evolved time. iTEBD turned
out to be far cheaper than budgeted (the entire 18-job standard grid, up to
chi=256, finished in 2.3 minutes total; canonicalize() costs roughly 1s at
chi=256, not the tens of seconds guessed before measuring), so a schedule
several times longer than even the finite study's is still cheap.

The schedule
------------
Repeats each dt rather than lengthening the fine tail, mirroring
renyi2_swssb_longrun.py's reasoning: evolved time is dt * steps, so time is
cheapest at large dt, while the fine stages exist only to remove the O(dt^2)
Trotter bias of the converged state, not to do the bulk of relaxing.
LONG_DT_SCHEDULE runs 31 stages (9300 steps) for 472.5 time units, versus the
standard schedule's 5 stages (1500 steps) for 55.5 time units -- 6.2x the
cost, 8.5x the evolved time.

Cache is namespaced 'inflong_...', distinct from both the standard infinite
grid ('inf_...') and the finite grids -- these are the same (sample, init,
chi) triples under a different schedule, and aliasing the cache would
silently return the old, under-converged runs.

Outputs (experiments/results/):
    imps_swssb_infinite_longrun.pkl -- the long-schedule grid plus the
        standard-schedule grid for comparison.
    imps_swssb_infinite_longrun.png -- R(r) vs r per chi (long schedule),
        xi vs chi, and long-vs-standard overlays.
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

import imps_swssb_infinite as ex_inf
import renyi2_swssb as ex
from lindblad_mps import iobservables, models

KIND = "inflong"

# Each entry is one stage of ex_inf.STEPS_PER_DT (300) steps. See the module
# docstring: repeats of a dt, not a longer fine tail.
LONG_DT_SCHEDULE = [0.1] * 10 + [0.05] * 8 + [0.02] * 6 + [0.01] * 4 + [0.005] * 3

REFERENCE_R = ex_inf.REFERENCE_R


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    """Evolved time at the end of each stage of a schedule."""
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def run_job_long(job: dict) -> dict:
    """Worker entry point: pin the long schedule and namespace, then run normally.

    Windows spawns workers that import this module fresh (no inherited
    parent state), so the override has to happen inside the worker itself --
    exactly the pattern renyi2_swssb_longrun.py uses for the finite study.

    Input: job, a dict for ex_inf.run_job().
    Output: ex_inf.run_job()'s output.
    """
    ex_inf.DT_SCHEDULE = LONG_DT_SCHEDULE
    ex_inf.KIND = KIND
    return ex_inf.run_job(job)


def reference_result(sample: int, init: str, chi: int) -> dict | None:
    """Load the standard-schedule result for the same (sample, init, chi).

    Output: the cached result dict, or None if that run isn't cached.
    """
    path = os.path.join(ex.CACHE_DIR, f"inf_sample{sample}_{init}_chi{chi}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["result"]


def run_all() -> dict:
    """Run every (sample, init, chi) job on the long schedule.

    Output: pickle-ready dict with the long-schedule grid, the config, and
        the standard-schedule results for comparison.
    """
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()
    jobs = ex_inf.build_jobs(samples)
    for job in jobs:
        job["chi_max"] = job["chi_max"]  # unchanged; kept for clarity

    n_steps = len(LONG_DT_SCHEDULE) * ex_inf.STEPS_PER_DT
    old_steps = len(ex_inf.DT_SCHEDULE) * ex_inf.STEPS_PER_DT
    total_time = sum(LONG_DT_SCHEDULE) * ex_inf.STEPS_PER_DT
    old_time = sum(ex_inf.DT_SCHEDULE) * ex_inf.STEPS_PER_DT

    grid = {
        "config": {
            "samples": ex_inf.SAMPLES, "inits": ex_inf.INITS, "chi_list": ex_inf.CHI_LIST,
            "r_max": ex_inf.R_MAX, "dt_schedule": LONG_DT_SCHEDULE,
            "steps_per_dt": ex_inf.STEPS_PER_DT,
            "canonicalize_every": ex_inf.CANONICALIZE_EVERY, "epsilon": ex.EPSILON,
            "evolved_time": total_time, "n_steps": n_steps,
            "reference_dt_schedule": ex_inf.DT_SCHEDULE,
            "reference_evolved_time": old_time,
        },
        "descriptions": {s: samples[s]["description"] for s in ex_inf.SAMPLES},
        "runs": {},
        "reference": {},
    }

    print(f"{len(jobs)} long-schedule iTEBD runs queued: samples {ex_inf.SAMPLES} x "
          f"inits {ex_inf.INITS} x chi in {ex_inf.CHI_LIST}.", flush=True)
    print(f"schedule: {len(LONG_DT_SCHEDULE)} stages, {n_steps} steps, "
          f"{total_time:.1f} time units ({n_steps/old_steps:.1f}x the cost and "
          f"{total_time/old_time:.1f}x the evolved time of the standard "
          f"{old_steps}-step schedule).\n", flush=True)

    t0 = time.perf_counter()
    done, failures = 0, []
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_job_long, jobs):
            done += 1
            key = (out["sample"], out["init"], out["chi_max"])
            tag = f"  [{done:2d}/{len(jobs)}] sample{out['sample']} {out['init']:>4s} chi={out['chi_max']:3d}"
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  {out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            res = out["result"]
            grid["runs"][key] = res
            grid["reference"][key] = reference_result(*key)
            ref = grid["reference"][key]
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
            xi_str = f"{res['xi']:.2f}" if res["xi"] is not None else "None"
            xi_change = ""
            if ref is not None and ref["xi"] is not None and res["xi"] is not None:
                xi_change = f"  (was {ref['xi']:.2f})"
            maxbond = max(res["bond_dims"].values())
            print(f"{tag}  R(1)={res['profile'][0][1]:.4e}  R({ex_inf.R_MAX})={res['profile'][-1][1]:.4e}  "
                  f"xi={xi_str}{xi_change}  bond={maxbond:3d}/{out['chi_max']}"
                  f"{'!' if maxbond >= out['chi_max'] else ' '}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    grid["failures"] = [{k: v for k, v in f.items() if k != "L_pp"} for f in failures]
    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t0)/60:.1f} min.", flush=True)
    for f in failures:
        print(f"  FAILED: {ex_inf.job_key(f)}", flush=True)
    return grid


def summarize(grid: dict) -> None:
    """Print the long-vs-standard chi-convergence table per sample/init."""
    cfg = grid["config"]
    print(f"\n{'='*90}\niTEBD thermodynamic-limit grid -- long schedule "
          f"({cfg['evolved_time']:.0f} time units vs standard "
          f"{cfg['reference_evolved_time']:.0f})\n{'='*90}")
    for s in cfg["samples"]:
        for init in cfg["inits"]:
            rows = [grid["runs"].get((s, init, chi)) for chi in cfg["chi_list"]]
            if all(r is None for r in rows):
                continue
            print(f"\nsample {s}, init |{init}>:")
            header = (f"  {'chi':>5}  {'schedule':>8}  " +
                      "  ".join(f"R(r={r:3d})" for r in REFERENCE_R) + "      xi      bond")
            print(header)
            for chi, res in zip(cfg["chi_list"], rows):
                ref = grid["reference"].get((s, init, chi))
                for which, r in (("standard", ref), ("long", res)):
                    if r is None:
                        print(f"  {chi:>5}  {which:>8}  (missing)")
                        continue
                    prof = dict(r["profile"])
                    r_vals = "  ".join(f"{prof.get(rr, float('nan')):.3e}" for rr in REFERENCE_R)
                    xi_str = f"{r['xi']:7.2f}" if r["xi"] is not None else "   None"
                    maxbond = max(r["bond_dims"].values())
                    flag = "!" if maxbond >= chi else " "
                    print(f"  {chi:>5}  {which:>8}  {r_vals}  {xi_str}  {maxbond:4d}/{chi}{flag}")
            print()

    print("A converged sample shows R(r) flattening (not oscillating) at large r, "
          "AND xi stabilizing across chi (not still drifting).", flush=True)


def plot_grid(grid: dict) -> str:
    """Plot R(r) long-vs-standard overlay per (sample, init) at chi=256, and xi vs chi."""
    plt = ex._mpl()
    cfg = grid["config"]
    cmap = plt.get_cmap("viridis")
    panels = [(s, init) for s in cfg["samples"] for init in cfg["inits"]]
    chi_focus = cfg["chi_list"][-1]  # chi=256, the production point

    fig, axes = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 9), squeeze=False)
    for col, (s, init) in enumerate(panels):
        ax0, ax1 = axes[0][col], axes[1][col]

        res = grid["runs"].get((s, init, chi_focus))
        ref = grid["reference"].get((s, init, chi_focus))
        if res is not None:
            r = [p[0] for p in res["profile"]]
            v = [p[1] for p in res["profile"]]
            ax0.plot(r, v, "-", color="crimson", lw=1.8, label="long schedule")
        if ref is not None:
            r = [p[0] for p in ref["profile"]]
            v = [p[1] for p in ref["profile"]]
            ax0.plot(r, v, "--", color="grey", lw=1.2, alpha=0.7, label="standard schedule")
        ax0.set_yscale("log")
        ax0.set_xlabel("separation r")
        ax0.set_ylabel(r"$R(0,r)$")
        ax0.set_title(f"sample {s}, |{init}>, chi={chi_focus}")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=7)

        xis_long, xis_std, chis = [], [], []
        for chi in cfg["chi_list"]:
            r_long = grid["runs"].get((s, init, chi))
            r_std = grid["reference"].get((s, init, chi))
            chis.append(chi)
            xis_long.append(r_long["xi"] if r_long and r_long["xi"] is not None else np.nan)
            xis_std.append(r_std["xi"] if r_std and r_std["xi"] is not None else np.nan)
        ax1.plot(chis, xis_long, "-o", color="crimson", label="long")
        ax1.plot(chis, xis_std, "--s", color="grey", alpha=0.7, label="standard")
        ax1.set_xlabel("bond dimension chi")
        ax1.set_ylabel(r"correlation length $\xi$")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7)
        ax1.set_title("plateau = converged")

    fig.suptitle(f"Long vs standard schedule ({cfg['evolved_time']:.0f} vs "
                 f"{cfg['reference_evolved_time']:.0f} time units), "
                 rf"$\epsilon={cfg['epsilon']}$")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_swssb_infinite_longrun.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_all()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_swssb_infinite_longrun.pkl")
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
