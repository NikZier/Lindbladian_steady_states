"""Sample 8's iTEBD convergence timescale: R(r)/xi vs evolved time, both starts.

Why
---
Sample 8 was still visibly unconverged after 472.5 time units (8.5x the
standard schedule) in imps_swssb_infinite_longrun.py -- noisy, non-plateauing
R(r), no stable xi. It is *expected* to be the slowest-relaxing of the three
samples run there: it has the smallest pair-creation matrix element
|<11|L''|00>|^2 of all 10 samples in the study (established this session),
and both R and the Liouvillian gap scale with that same quantity, so it sets
the longest relaxation time by construction, not by accident.

Rather than guess a schedule long enough for sample 8 and re-run the whole
3-sample grid at that length, this script runs ONLY sample 8, both initial
states, at ONE chi (256 -- bond dimension has stayed well below chi_max at
every chi tried so far, so chi is not the limiting factor here, only time
is), on a schedule with FINE time resolution via stage_callback: R(r) at a
few reference separations and xi are recorded at every stage, not just at
the end. That gives a genuine R(t)/xi(t) trajectory instead of one number,
so we can read off the timescale directly -- from the zero and neel starts
converging to the SAME value (the same diagnostic the finite-N study used
throughout: a unique steady state forces agreement, and the point where the
two curves meet IS the relaxation timescale) and from R(r)/xi themselves
flattening in time. That timescale is then the basis for how long the
OTHER (faster-relaxing, since their creation rates are larger) samples
plausibly need, without having to run everyone this long to find out.

The schedule
------------
STEPS_PER_DT = 50 (not the usual 300) specifically to get many stage_callback
samples -- resolution in the trajectory, not just cost, is the point of this
run. 435 stages, 21750 steps, 1740 evolved time units: 3.7x the 472.5 units
already tried and unconverged, sized from the gap argument above (sample 8's
creation rate is ~3.2x smaller than sample 0's, whose 472.5-unit run showed
much better -- if imperfect -- convergence, so a relaxation time a few times
longer is the expected ballpark, not a guess pulled from nowhere).

correlation_length() now reports xi=None with reason='near_degenerate' when
the profile is too flat to resolve a finite decay length numerically rather
than returning a blown-up, non-repeatable number (fixed this session after
the long-schedule grid's sample 6 came back with xi jumping 750/11299/455
across chi -- noise from a near-degenerate eigenvalue ratio, not real
variation) -- the trajectory printed below carries that reason through.

Outputs (experiments/results/):
    imps_sample8_timescale.pkl -- the (t, R(ref_r), xi, reason, bond) trajectory
        for both starts, plus the final full r_max=100 profile.
    imps_sample8_timescale.png -- R(ref_r) vs t and xi vs t, zero and neel
        overlaid -- the point where they meet is the answer.
"""

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

STEPS_PER_DT = 50  # small on purpose: resolution in time, see module docstring
DT_SCHEDULE = [0.1] * 300 + [0.05] * 80 + [0.02] * 30 + [0.01] * 15 + [0.005] * 10
CANONICALIZE_EVERY = 10

KIND = "sample8ts"


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


def run_one(init: str, L_pp: np.ndarray) -> dict:
    """Run sample 8 from one initial state, recording a stage-by-stage trajectory.

    Cached under its own key; a cache hit with a matching run_config is
    returned without recomputing.

    Output: dict with 'trajectory' (list of per-stage dicts: 't', 'R'
        (reference-r dict), 'xi', 'reason', 'bond'), 'final_profile'
        (full r_max=100 profile at the end) and 'seconds'.
    """
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

    def stage_callback(stage: int, dt: float, state) -> None:
        profile = iobservables.correlator_profile(state, models.X, r_max=max(REFERENCE_R))
        prof_map = dict(profile)
        xi_diag = iobservables.correlation_length(state)
        trajectory.append({
            "t": elapsed_times(DT_SCHEDULE, STEPS_PER_DT)[stage],
            "R": {r: prof_map.get(r, float("nan")) for r in REFERENCE_R},
            "xi": xi_diag["xi"],
            "reason": xi_diag["reason"],
            "bond": max(state.bond_dims.values()),
        })

    L2_terms = ex.build_L2_terms(L_pp)
    state, history = itebd.find_steady_state_infinite(
        H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[],
        dt_schedule=DT_SCHEDULE, steps_per_dt=STEPS_PER_DT,
        chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
        initial_state=ex_inf.build_initial_state(init),
        stage_callback=stage_callback,
    )
    final_profile = iobservables.correlator_profile(state, models.X, r_max=R_MAX)

    out = {
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


def run_all() -> dict:
    """Run both starts sequentially (only 2 runs -- not worth a worker pool).

    Output: pickle-ready dict with both trajectories and config.
    """
    samples = ex.draw_samples()
    L_pp = samples[SAMPLE]["L_pp"]

    n_steps = len(DT_SCHEDULE) * STEPS_PER_DT
    total_time = sum(DT_SCHEDULE) * STEPS_PER_DT
    print(f"sample {SAMPLE}, chi={CHI}: {len(DT_SCHEDULE)} stages, {n_steps} steps, "
          f"{total_time:.1f} evolved time units, both starts.\n", flush=True)

    grid = {
        "config": {
            "sample": SAMPLE, "chi": CHI, "dt_schedule": DT_SCHEDULE,
            "steps_per_dt": STEPS_PER_DT, "canonicalize_every": CANONICALIZE_EVERY,
            "reference_r": REFERENCE_R, "r_max": R_MAX, "epsilon": ex.EPSILON,
            "evolved_time": total_time, "n_steps": n_steps,
        },
        "description": samples[SAMPLE]["description"],
        "runs": {},
    }

    t0 = time.perf_counter()
    for init in INITS:
        out = run_one(init, L_pp)
        grid["runs"][init] = out
        timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
        last = out["trajectory"][-1]
        xi_str = f"{last['xi']:.2f}" if last["xi"] is not None else f"None({last['reason']})"
        print(f"  {init:>4s}  done  t_final={last['t']:.1f}  R(1)={last['R'][1]:.4e}  "
              f"xi={xi_str}  bond={last['bond']:3d}/{CHI}  "
              f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    return grid


def summarize(grid: dict) -> None:
    """Print the R(r)/xi trajectory for both starts, and where they converge."""
    cfg = grid["config"]
    print(f"\n{'='*100}\nsample {cfg['sample']} convergence trajectory, chi={cfg['chi']}, "
          f"{cfg['evolved_time']:.0f} time units\n{'='*100}")

    traj_zero = grid["runs"]["zero"]["trajectory"]
    traj_neel = grid["runs"]["neel"]["trajectory"]
    print(f"\n{'t':>8}  {'R(1) zero':>11} {'R(1) neel':>11} {'gap':>8}  "
          f"{'R(20) zero':>11} {'R(20) neel':>11} {'gap':>8}  "
          f"{'xi zero':>10} {'xi neel':>10}  {'bond z/n':>10}")
    # print a subsample of stages (every ~N-th) so the table stays readable
    n = len(traj_zero)
    stride = max(1, n // 40)
    for i in range(0, n, stride):
        z, ne = traj_zero[i], traj_neel[i]
        gap1 = (ne["R"][1] / z["R"][1] - 1) if z["R"][1] else float("nan")
        gap20 = (ne["R"][20] / z["R"][20] - 1) if z["R"][20] else float("nan")
        xi_z = f"{z['xi']:.2f}" if z["xi"] is not None else "None"
        xi_n = f"{ne['xi']:.2f}" if ne["xi"] is not None else "None"
        print(f"{z['t']:8.1f}  {z['R'][1]:11.3e} {ne['R'][1]:11.3e} {gap1:+8.1%}  "
              f"{z['R'][20]:11.3e} {ne['R'][20]:11.3e} {gap20:+8.1%}  "
              f"{xi_z:>10} {xi_n:>10}  {z['bond']:4d}/{ne['bond']:<4d}")

    print("\nLook for: R(r) from zero and neel converging to the SAME value (gap -> 0%), "
          "AND that value stabilizing in t (not still drifting) -- that point in t is "
          "the relaxation timescale for this sample.", flush=True)


def plot(grid: dict) -> str:
    """Plot R(ref_r) vs t and xi vs t, zero and neel overlaid."""
    plt = ex._mpl()
    cfg = grid["config"]
    colors = {"zero": "tab:blue", "neel": "tab:red"}
    cmap = plt.get_cmap("plasma")

    fig, axes = plt.subplots(1, len(REFERENCE_R) + 1, figsize=(4.2 * (len(REFERENCE_R) + 1), 4.5))
    for col, r in enumerate(REFERENCE_R):
        ax = axes[col]
        for init in INITS:
            traj = grid["runs"][init]["trajectory"]
            t = [pt["t"] for pt in traj]
            v = [pt["R"][r] for pt in traj]
            ax.plot(t, v, "-", color=colors[init], lw=1.3, alpha=0.85, label=f"|{init}>")
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

    fig.suptitle(f"Sample {cfg['sample']} convergence trajectory, chi={cfg['chi']} "
                 rf"($\epsilon={cfg['epsilon']}$)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_sample8_timescale.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_all()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_sample8_timescale.pkl")
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
    main()
