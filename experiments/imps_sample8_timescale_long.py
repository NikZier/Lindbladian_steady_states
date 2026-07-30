"""Sample 8's iTEBD convergence timescale, extended: 5x imps_sample8_timescale.py.

Why
---
The first timescale run (imps_sample8_timescale.py, 1740 evolved time units)
showed a real but incomplete signal: the MEDIAN of R(r) over the run's later
stages (t>1000) agrees between the zero and neel starts to within a few
percent at r=20/50/100 (5.01e-4 vs 4.84e-4 at r=20, etc.), landing within
~2x of the established finite-N converged value (1.92e-4) -- but any SINGLE
stage is dominated by large occasional spikes (r=20 alone ranged from
-1.1e-2 to +6.4e-3, ~20x the median), and r=1/r=5 stayed considerably
noisier than r>=20 even in the median. Both open questions -- does the
spread shrink with more time, and do r=1/r=5 eventually settle the way
r>=20 already has -- need more evolved time to answer, not more analysis of
the same data.

This script is exactly the same setup (sample 8, both starts, chi=256, same
STEPS_PER_DT=50 for direct comparability of the trajectories) at 5x the
schedule length: 8700 evolved time units. The two starts now run in
PARALLEL (mp.Pool(2)) rather than sequentially, since a run this long is
worth not waiting on twice.

Robust summary
---------------
Reports MEDIAN and interquartile spread of R(r) over the last 30% of stages
(not just the terminal value) for both starts, plus their agreement -- the
single-snapshot terminal value used elsewhere in this study is a poor
estimator here specifically because of the spike behavior just described;
median-of-late-stages is a much more stable one. This should probably be
retrofitted into imps_swssb_infinite.py / imps_swssb_infinite_longrun.py's
reporting too, once validated here.

Cache is namespaced 'sample8ts2_...', distinct from the first (shorter)
timescale run -- both are kept on disk for direct before/after comparison.

Outputs (experiments/results/):
    imps_sample8_timescale_long.pkl -- the extended trajectory for both starts.
    imps_sample8_timescale_long.png -- R(ref_r) vs t and xi vs t, zero and
        neel overlaid, plus the median/IQR robust summary printed to stdout.
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

SAMPLE = 8
INITS = ["zero", "neel"]
CHI = 256
R_MAX = 100
REFERENCE_R = [1, 5, 20, 50, 100]

STEPS_PER_DT = 50  # same as the first timescale run, for a directly comparable trajectory
DT_SCHEDULE = [0.1] * 1500 + [0.05] * 400 + [0.02] * 150 + [0.01] * 75 + [0.005] * 50
CANONICALIZE_EVERY = 10
LATE_STAGE_FRACTION = 0.3  # fraction of stages (from the end) used for the robust summary

KIND = "sample8ts2"


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    """Evolved time at the end of each stage of a schedule."""
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def job_key(init: str) -> str:
    return f"{KIND}_sample{SAMPLE}_{init}_chi{CHI}"


def run_config() -> dict:
    return {
        "dt_schedule": DT_SCHEDULE, "steps_per_dt": STEPS_PER_DT,
        "canonicalize_every": CANONICALIZE_EVERY, "cutoff": ex.CUTOFF, "r_max": R_MAX,
    }


def run_one(job: dict) -> dict:
    """Run sample 8 from one initial state, recording a stage-by-stage trajectory.

    Module-level (not a closure) so it survives pickling to a worker process
    (Windows spawn). Cached under its own key; a cache hit with a matching
    run_config is returned without recomputing.

    Input: job, dict with 'init', 'L_pp'.
    Output: dict with 'init', 'trajectory' (list of per-stage dicts: 't', 'R'
        (reference-r dict), 'xi', 'reason', 'bond'), 'final_profile'
        (full r_max=100 profile at the end), 'cached', 'seconds'.
    """
    init = job["init"]
    path = os.path.join(ex.CACHE_DIR, job_key(init) + ".pkl")
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
        initial_state=ex_inf.build_initial_state(init),
        stage_callback=stage_callback,
    )
    final_profile = iobservables.correlator_profile(state, models.X, r_max=R_MAX)

    out = {
        "init": init,
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


def robust_summary(trajectory: list[dict]) -> dict:
    """Median and interquartile spread of R(ref_r) over the last LATE_STAGE_FRACTION of stages.

    The terminal (last-stage) value used elsewhere in this study is a poor
    estimator for this run: individual stages carry large occasional spikes
    (measured in the shorter run: r=20 ranged -1.1e-2..+6.4e-3, ~20x its own
    median), so the median over many late stages is what should be compared
    across starts/samples, not any single snapshot.

    Output: dict mapping r -> {'median', 'q25', 'q75', 'n'}.
    """
    n = len(trajectory)
    cut = trajectory[int(n * (1 - LATE_STAGE_FRACTION)):]
    out = {}
    for r in REFERENCE_R:
        vals = np.array([pt["R"][r] for pt in cut])
        out[r] = {
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "n": len(vals),
        }
    return out


def run_all() -> dict:
    """Run both starts in parallel.

    Output: pickle-ready dict with both trajectories, robust summaries, and config.
    """
    samples = ex.draw_samples()
    L_pp = samples[SAMPLE]["L_pp"]
    jobs = [{"init": init, "L_pp": L_pp} for init in INITS]

    n_steps = len(DT_SCHEDULE) * STEPS_PER_DT
    total_time = sum(DT_SCHEDULE) * STEPS_PER_DT
    print(f"sample {SAMPLE}, chi={CHI}: {len(DT_SCHEDULE)} stages, {n_steps} steps, "
          f"{total_time:.1f} evolved time units, both starts, in parallel.\n", flush=True)

    grid = {
        "config": {
            "sample": SAMPLE, "chi": CHI, "dt_schedule": DT_SCHEDULE,
            "steps_per_dt": STEPS_PER_DT, "canonicalize_every": CANONICALIZE_EVERY,
            "reference_r": REFERENCE_R, "r_max": R_MAX, "epsilon": ex.EPSILON,
            "evolved_time": total_time, "n_steps": n_steps,
            "late_stage_fraction": LATE_STAGE_FRACTION,
        },
        "description": samples[SAMPLE]["description"],
        "runs": {},
        "robust": {},
    }

    t0 = time.perf_counter()
    with mp.Pool(processes=2) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            init = out["init"]
            grid["runs"][init] = out
            grid["robust"][init] = robust_summary(out["trajectory"])
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
            last = out["trajectory"][-1]
            xi_str = f"{last['xi']:.2f}" if last["xi"] is not None else f"None({last['reason']})"
            print(f"  {init:>4s}  done  t_final={last['t']:.1f}  R(1)={last['R'][1]:.4e}  "
                  f"xi={xi_str}  bond={last['bond']:3d}/{CHI}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    return grid


def summarize(grid: dict) -> None:
    """Print the robust (median/IQR) late-stage summary and the zero/neel agreement."""
    cfg = grid["config"]
    print(f"\n{'='*100}\nsample {cfg['sample']} robust summary "
          f"(median of last {cfg['late_stage_fraction']:.0%} of stages), "
          f"chi={cfg['chi']}, {cfg['evolved_time']:.0f} time units\n{'='*100}")

    print(f"\n{'r':>5}  {'median zero':>13} {'IQR zero':>21}  "
          f"{'median neel':>13} {'IQR neel':>21}  {'gap':>8}")
    for r in REFERENCE_R:
        rz = grid["robust"]["zero"][r]
        rn = grid["robust"]["neel"][r]
        iqr_z = f"[{rz['q25']:.2e}, {rz['q75']:.2e}]"
        iqr_n = f"[{rn['q25']:.2e}, {rn['q75']:.2e}]"
        gap = (rn["median"] / rz["median"] - 1) if rz["median"] else float("nan")
        print(f"{r:>5}  {rz['median']:13.4e} {iqr_z:>21}  "
              f"{rn['median']:13.4e} {iqr_n:>21}  {gap:+8.1%}")

    print(f"\n(reference: finite N<=12, chi=128 converged value for sample 8 was 2.21e-4; "
          f"the smaller separations used here, r=20-100 on an infinite chain, are not "
          f"directly the same quantity but should be the same order of magnitude if "
          f"converged.)")
    print("\nCompare median/IQR against the first (1740-unit) timescale run: shrinking IQR "
          "and closer zero/neel medians mean the extra time is helping; if both are "
          "similar to the first run, sample 8 may need a fundamentally different approach "
          "(larger chi, or the spikes are a genuine physical feature, not truncation noise).",
          flush=True)


def plot(grid: dict) -> str:
    """Plot R(ref_r) vs t and xi vs t, zero and neel overlaid."""
    plt = ex._mpl()
    cfg = grid["config"]
    colors = {"zero": "tab:blue", "neel": "tab:red"}

    fig, axes = plt.subplots(1, len(REFERENCE_R) + 1, figsize=(4.2 * (len(REFERENCE_R) + 1), 4.5))
    for col, r in enumerate(REFERENCE_R):
        ax = axes[col]
        for init in INITS:
            traj = grid["runs"][init]["trajectory"]
            t = [pt["t"] for pt in traj]
            v = [pt["R"][r] for pt in traj]
            ax.plot(t, v, "-", color=colors[init], lw=1.0, alpha=0.7, label=f"|{init}>")
            med = grid["robust"][init][r]["median"]
            ax.axhline(med, ls="--", color=colors[init], lw=1.5, alpha=0.9)
        ax.set_xlabel("evolved time")
        ax.set_ylabel(rf"$R(0,{r})$")
        ax.set_title(f"r = {r}")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    ax = axes[-1]
    for init in INITS:
        traj = grid["runs"][init]["trajectory"]
        t = [pt["t"] for pt in traj if pt["xi"] is not None]
        v = [pt["xi"] for pt in traj if pt["xi"] is not None]
        ax.plot(t, v, "-o", ms=2, color=colors[init], lw=1.0, alpha=0.85, label=f"|{init}>")
    ax.set_xlabel("evolved time")
    ax.set_ylabel(r"correlation length $\xi$ (only where resolved)")
    ax.set_title("xi vs t")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Sample {cfg['sample']} extended convergence trajectory, chi={cfg['chi']} "
                 rf"($\epsilon={cfg['epsilon']}$), dashed = late-stage median")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_sample8_timescale_long.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_all()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_sample8_timescale_long.pkl")
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
