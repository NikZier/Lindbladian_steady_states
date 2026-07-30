"""All three samples, |neel> only, on sample 8's extended-timescale methodology.

Why
---
Sample 8's 8700-unit run (imps_sample8_timescale_long.py) showed that a
single terminal snapshot is a poor estimator (individual stages carry large
spikes, up to ~20x the typical value) but that the MEDIAN over the
trajectory is stable and physically sensible where the raw plot looked
noisy -- most clearly for R(0,100) from |neel>, whose median drifted only 4%
across three-way-split windows of the run and sat within 10-25% of the
established finite-N answer (2.21e-4) throughout, not just at the end.

This applies the identical methodology -- same schedule, same chi, same
fine-grained stage_callback trajectory, same rolling mean/median analysis --
to samples 6 and 0 as well, so all three samples get the same treatment
sample 8 did. Only |neel> is run this time (not |zero>): the zero/neel
agreement check that motivated running both has already been established as
a useful diagnostic in the shorter runs (renyi2_swssb*.py, the earlier
imps_swssb_infinite* grids), and this pass is specifically about getting a
clean, well-resolved trajectory for each sample rather than re-doing that
cross-check -- running one init per sample instead of two roughly halves
the cost for the same wall-clock budget.

The three samples run in PARALLEL (mp.Pool(3) -- N_WORKERS=12 available,
comfortably enough for 3 concurrent chi=256 iTEBD jobs).

Schedule: identical to imps_sample8_timescale_long.py -- chi=256,
STEPS_PER_DT=50, 2175 stages, 108750 steps, 8700 evolved time units. Reusing
the same schedule (not a fresh guess per sample) keeps trajectories directly
comparable across samples, and sample 8 -- the slowest-relaxing of the three
by construction (smallest pair-creation rate, see CLAUDE.md) -- already
needed this much time, so 6 and 0 (faster relaxers) get at least as much
margin.

Outputs (experiments/results/):
    imps_all_samples_neel_timescale.pkl -- per-sample trajectory + robust
        (median-by-thirds) summary.
    imps_all_samples_neel_timescale.png -- rolling mean/median R(r) vs t,
        one row per sample, one column per reference r, y-axis scaled to
        the rolling quantities (not the raw spikes) -- same style as
        imps_sample8_rolling_average.py.
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
from lindblad_mps import iobservables, itebd, models

SAMPLES = [8, 6, 0]
INIT = "neel"
CHI = 128  # bond dimension peaked at 22-32 with discarded weight ~1e-16 in every
           # run so far -- 256 was never remotely binding, 128 is still ~4x margin
R_MAX = 100
REFERENCE_R = [1, 5, 20, 50, 100]

STEPS_PER_DT = 50  # same as imps_sample8_timescale_long.py, for direct comparability
DT_SCHEDULE = [0.1] * 500 + [0.05] * 120 + [0.02] * 100 + [0.01] * 60 + [0.005] * 40
# 2940 evolved time units, down from 8700. The old length was sized to fight
# oscillations that turned out to be a measurement artifact (see
# iobservables.py's docstring). Post-fix, imps_schedule_length_check.py measured
# the actual plateau times: sample 6 at t=105, sample 0 at t=770, sample 8 still
# moving at t=1219 -- so ~2940 gives the slowest sample >2x margin at ~3x less cost.
CANONICALIZE_EVERY = 10
ROLLING_WINDOW = 200.0  # evolved-time units, same as imps_sample8_rolling_average.py

KIND = "allsamplests"


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def job_key(sample: int) -> str:
    return f"{KIND}_sample{sample}_{INIT}_chi{CHI}"


def run_config() -> dict:
    return {
        "dt_schedule": DT_SCHEDULE, "steps_per_dt": STEPS_PER_DT,
        "canonicalize_every": CANONICALIZE_EVERY, "cutoff": ex.CUTOFF, "r_max": R_MAX,
    }


def run_one(job: dict) -> dict:
    """Run one sample from |neel>, recording a stage-by-stage trajectory.

    Module-level (not a closure) so it survives pickling to a worker process
    (Windows spawn). Cached under its own key; a cache hit with a matching
    run_config is returned without recomputing.

    Input: job, dict with 'sample', 'L_pp'.
    Output: dict with 'sample', 'trajectory', 'final_profile', 'cached', 'seconds'.
    """
    sample = job["sample"]
    path = os.path.join(ex.CACHE_DIR, job_key(sample) + ".pkl")
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
    trajectory: list[dict] = []
    times = elapsed_times(DT_SCHEDULE, STEPS_PER_DT)

    def stage_callback(stage: int, dt: float, state) -> None:
        profile = iobservables.correlator_profile(state, models.X, r_max=max(REFERENCE_R))
        prof_map = dict(profile)
        xi_diag = iobservables.correlation_length(state)
        trajectory.append({
            "t": times[stage],
            "R": {r: prof_map.get(r, float("nan")) for r in REFERENCE_R},
            "xi": xi_diag["xi"],
            "reason": xi_diag["reason"],
            "bond": max(state.bond_dims.values()),
        })

    L2_terms = ex.build_L2_terms(job["L_pp"])
    state, history = itebd.find_steady_state_infinite(
        H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[],
        dt_schedule=DT_SCHEDULE, steps_per_dt=STEPS_PER_DT,
        chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
        initial_state=ex_inf.build_initial_state(INIT),
        stage_callback=stage_callback,
    )
    final_profile = iobservables.correlator_profile(state, models.X, r_max=R_MAX)

    out = {
        "sample": sample,
        "trajectory": trajectory,
        "final_profile": final_profile,
        "max_discarded_weight": max(history["discarded_weight"], default=0.0),
        "run_config": run_config(),
        "cached": False,
        "seconds": time.perf_counter() - t0,
    }
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return out


def rolling_mean_median(t: np.ndarray, v: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-windowed rolling mean and median -- see imps_sample8_rolling_average.py."""
    half = window / 2.0
    mean = np.empty_like(v)
    median = np.empty_like(v)
    lo = hi = 0
    n = len(t)
    for i in range(n):
        while lo < n and t[lo] < t[i] - half:
            lo += 1
        while hi < n and t[hi] <= t[i] + half:
            hi += 1
        window_vals = v[lo:hi]
        mean[i] = np.mean(window_vals)
        median[i] = np.median(window_vals)
    return mean, median


def median_by_thirds(trajectory: list[dict], r: int) -> list[float]:
    """Median of R(r) in each of three equal-count windows of the trajectory
    -- the drift-detection check that caught R(0,50)|zero> still moving in
    the sample-8 run and confirmed R(0,100)|neel> was genuinely stable."""
    v = np.array([pt["R"][r] for pt in trajectory])
    n = len(v)
    return [float(np.median(v[lo:hi])) for lo, hi in [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]]


def run_all() -> dict:
    """Run all three samples in parallel.

    Output: pickle-ready dict with per-sample trajectory and robust summary.
    """
    samples = ex.draw_samples()
    jobs = [{"sample": s, "L_pp": samples[s]["L_pp"]} for s in SAMPLES]

    n_steps = len(DT_SCHEDULE) * STEPS_PER_DT
    total_time = sum(DT_SCHEDULE) * STEPS_PER_DT
    print(f"samples {SAMPLES}, init=|{INIT}>, chi={CHI}: {len(DT_SCHEDULE)} stages, "
          f"{n_steps} steps, {total_time:.1f} evolved time units, all in parallel.\n", flush=True)

    grid = {
        "config": {
            "samples": SAMPLES, "init": INIT, "chi": CHI, "dt_schedule": DT_SCHEDULE,
            "steps_per_dt": STEPS_PER_DT, "canonicalize_every": CANONICALIZE_EVERY,
            "reference_r": REFERENCE_R, "r_max": R_MAX, "epsilon": ex.EPSILON,
            "evolved_time": total_time, "n_steps": n_steps,
        },
        "descriptions": {s: samples[s]["description"] for s in SAMPLES},
        "runs": {},
        "median_by_thirds": {},
    }

    t0 = time.perf_counter()
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            s = out["sample"]
            grid["runs"][s] = out
            grid["median_by_thirds"][s] = {r: median_by_thirds(out["trajectory"], r) for r in REFERENCE_R}
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
            last = out["trajectory"][-1]
            xi_str = f"{last['xi']:.2f}" if last["xi"] is not None else f"None({last['reason']})"
            print(f"  sample{s}  done  t_final={last['t']:.1f}  R(1)={last['R'][1]:.4e}  "
                  f"xi={xi_str}  bond={last['bond']:3d}/{CHI}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    return grid


def summarize(grid: dict) -> None:
    """Print the median-by-thirds drift table for every sample/reference r."""
    cfg = grid["config"]
    print(f"\n{'='*100}\nall samples, init=|{cfg['init']}>, chi={cfg['chi']}, "
          f"{cfg['evolved_time']:.0f} time units -- median-by-thirds drift\n{'='*100}")
    print(f"\n{'sample':>6} {'r':>4}  {'1st third':>11} {'2nd third':>11} {'3rd third':>11}  {'drift 1st->3rd':>14}")
    for s in cfg["samples"]:
        for r in REFERENCE_R:
            thirds = grid["median_by_thirds"][s][r]
            drift = (thirds[2] / thirds[0] - 1) if thirds[0] else float("nan")
            print(f"{s:>6} {r:>4}  {thirds[0]:11.3e} {thirds[1]:11.3e} {thirds[2]:11.3e}  {drift:+13.0%}")
        print()
    print("Small |drift| (a few %) means the median has genuinely settled, not just landed in a "
          "calm patch of a still-fluctuating trajectory -- see the sample-8 write-up for why this "
          "check matters more than either a single terminal value or eyeballing the raw trace.",
          flush=True)


def plot(grid: dict) -> str:
    """Plot rolling mean/median R(r) vs t, one row per sample, one column per reference r."""
    plt = ex._mpl()
    cfg = grid["config"]

    fig, axes = plt.subplots(len(cfg["samples"]), len(REFERENCE_R),
                              figsize=(4.2 * len(REFERENCE_R), 4.0 * len(cfg["samples"])), squeeze=False)
    for row, s in enumerate(cfg["samples"]):
        traj = grid["runs"][s]["trajectory"]
        t = np.array([pt["t"] for pt in traj])
        for col, r in enumerate(REFERENCE_R):
            ax = axes[row][col]
            v = np.array([pt["R"][r] for pt in traj])
            mean, median = rolling_mean_median(t, v, ROLLING_WINDOW)

            ax.plot(t, v, "-", color="tab:red", lw=0.4, alpha=0.15)
            ax.plot(t, mean, "-", color="tab:red", lw=1.4, alpha=0.85, label="rolling mean")
            ax.plot(t, median, "--", color="tab:red", lw=1.6, alpha=0.9, label="rolling median")

            lo, hi = np.percentile(np.concatenate([mean, median]), [1, 99])
            pad = 0.15 * max(hi - lo, 1e-8)
            ax.set_ylim(lo - pad, hi + pad)
            ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
            ax.set_xlabel("evolved time")
            ax.set_ylabel(rf"$R(0,{r})$")
            ax.set_title(f"sample {s}, r = {r}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.suptitle(rf"All samples, |{cfg['init']}> only: rolling mean/median of $R(r)$ "
                 f"(window = {ROLLING_WINDOW:.0f} time units), faint line = raw trajectory")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_all_samples_neel_timescale.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_all()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_all_samples_neel_timescale.pkl")
    with open(path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\nSaved trajectory -> {path}", flush=True)

    try:
        summarize(grid)
    except Exception:
        print(f"!! summary failed (data is still saved):\n{traceback.format_exc()}", flush=True)
    try:
        print(f"Saved plot -> {plot(grid)}", flush=True)
    except Exception:
        print(f"!! plot failed (data is still saved):\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
