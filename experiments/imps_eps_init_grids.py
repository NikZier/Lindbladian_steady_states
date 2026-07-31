"""Three iTEBD grids: sample-8 convergence, epsilon=0.15, and |0...0> at epsilon=0.2.

Why these three, together
-------------------------
All three are iTEBD-only (no finite-N work) and independent job-for-job, so
they share one pool rather than three sequential launches. Splitting them into
separate scripts would serialise the tail: the single longest job here (sample
8 at 15015 time units) runs ~2 h, and the other 20 jobs fit inside its shadow
on 12 workers.

1. **sample 8, epsilon=0.2, |neel>, 15015 time units** (5.1x the standard
   2940-unit schedule). Sample 8 has the smallest pair-creation matrix element
   in the ensemble, |<11|L''|00>|^2 = 0.00177, 12.5x below sample 6. Since the
   Liouvillian gap is of order that creation rate, sample 8 relaxes ~12.5x
   slower than sample 6, which plateaus at t~105 -- putting sample 8's plateau
   near t~1300 in the most optimistic reading. It was still drifting (-16% to
   -26% median-by-thirds) at 2940, so this run buys ~5x more time to settle the
   question of whether that drift is unconverged relaxation or something real.

2. **all 10 samples, epsilon=0.15, |neel>, 5234 time units** (1.78x the
   standard schedule). The scale factor is not a guess -- it comes from the
   gap. L'' is rescaled to operator norm epsilon, so the pair-creation matrix
   element <11|L''|00> scales LINEARLY in epsilon and the rate (hence the gap)
   as epsilon^2. Relaxation time tau ~ 1/gap ~ 1/epsilon^2, so

       tau(0.15)/tau(0.2) = (0.2/0.15)^2 = 16/9 = 1.778

   and 2940 * 1.778 = 5228, rounded to the 5234-unit schedule below. Running
   the same 2940 units at the smaller epsilon would have given every sample
   LESS relaxation than the epsilon=0.2 grid had, which is exactly backwards.

   This grid also tests a sharp prediction rather than filling in an open
   parameter: CLAUDE.md establishes R = 0.1256 * |<11|L''|00>|^2, so

       R(epsilon=0.15) / R(epsilon=0.2) = (0.15/0.2)^2 = 0.5625

   sample by sample. Verified numerically at setup time (assert below): the
   rescaling preserves the L'' direction and changes only the norm, so all ten
   draws give 0.5625 to machine precision in the matrix element. Whether the
   measured R follows is the actual test.

3. **all 10 samples, epsilon=0.2, |0...0>, 2940 time units.** The |neel> grid
   exists; this is its partner. |0...0> is a DARK state of the baseline, so
   baseline R = 0 identically and the state approaches the steady state from
   BELOW, while |neel> carries real correlations and comes down from above. A
   unique steady state forces both to the same R, so agreement is convergence
   from two directions and disagreement brackets the truth -- the same
   diagnostic that exposed the finite-N under-relaxation at N=20 (see
   CLAUDE.md, "Worked example -- sample 8"). Same 2940-unit schedule as the
   |neel> grid so the two are directly comparable.

Cache safety
------------
job_key() does NOT encode epsilon, and grids 2 and 3 differ from the existing
|neel> epsilon=0.2 runs in ways the key alone would not catch. Two guards:
each grid gets its own KIND namespace, AND epsilon is stored in run_config so
a mismatch forces a recompute rather than silently returning the wrong run.
Both sample sets are drawn in the PARENT (renyi2_swssb.EPSILON is module
state, and workers re-import the module fresh under Windows spawn), and the
resulting L'' matrices are passed by value in the job dict -- so no worker
ever reads EPSILON.

Outputs (experiments/results/):
    imps_eps_init_grids.pkl -- every trajectory, keyed (kind, sample)
    imps_eps015_prediction.png -- measured R(0.15)/R(0.2) vs the predicted 0.5625
    imps_zero_vs_neel_infinite.png -- the two-initial-state uniqueness check
    imps_sample8_long.png -- sample 8's 15015-unit trajectory
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

CHI = 128
R_MAX = 100
REFERENCE_R = [1, 5, 20, 50, 100]
STEPS_PER_DT = 50
CANONICALIZE_EVERY = 10
ALL_SAMPLES = list(range(10))

# The established epsilon=0.2 schedule: 2940 evolved time units.
SCHEDULE_BASE = [0.1] * 500 + [0.05] * 120 + [0.02] * 100 + [0.01] * 60 + [0.005] * 40
# x1.78 for epsilon=0.15, from tau ~ 1/epsilon^2 (see docstring): 5234 units.
SCHEDULE_EPS015 = [0.1] * 890 + [0.05] * 214 + [0.02] * 178 + [0.01] * 107 + [0.005] * 71
# x5.1 for sample 8, whose gap is 12.5x below sample 6's: 15015 units.
SCHEDULE_S8LONG = [0.1] * 2800 + [0.05] * 300 + [0.02] * 200 + [0.01] * 100 + [0.005] * 60
# 1.78 x 15015 = 26790, for sample 8 at epsilon=0.15 -- the slowest sample at
# the smaller epsilon, so it takes the long schedule AND the 1/epsilon^2 factor.
SCHEDULE_EPS015LONG = [0.1] * 5000 + [0.05] * 530 + [0.02] * 350 + [0.01] * 180 + [0.005] * 100

# Samples whose 2940-unit |zero> runs were still building at the end (drift
# +3%, +85%, +230% and 3-6.5% spread against |neel>). Samples 7 and 8 hold the
# two smallest pair-creation rates in the ensemble, so the gap argument puts
# them last to relax; sample 3 is mid-pack but drifted enough to be worth
# redoing rather than quoting.
ZERO_SLOW = [3, 7, 8]


def scale_schedule(schedule: list[float], factor: int) -> list[float]:
    """Repeat each dt stage `factor` times: same dt anneal, `factor` x the time.

    Lengthening by adding stages (rather than by raising steps_per_dt or
    coarsening dt) keeps the dt annealing profile identical, so the Trotter
    floor at the end of the run -- r = 0.0202 dt^2 -- is unchanged and the
    longer runs remain directly comparable to the shorter ones.
    """
    out: list[float] = []
    for dt in schedule:
        out.extend([dt] * factor)
    return out


# tau ~ 1/gap ~ 1/epsilon^2, so relative to epsilon=0.2:
#   epsilon=0.10 -> (0.2/0.10)^2 =  4x
#   epsilon=0.05 -> (0.2/0.05)^2 = 16x
SCHEDULE_EPS010 = scale_schedule(SCHEDULE_BASE, 4)        # 11760 units
SCHEDULE_EPS005 = scale_schedule(SCHEDULE_BASE, 16)       # 47040 units
SCHEDULE_EPS010LONG = scale_schedule(SCHEDULE_S8LONG, 4)  # 60060 units
SCHEDULE_EPS005LONG = scale_schedule(SCHEDULE_S8LONG, 16)  # 240240 units

GRIDS = {
    "s8long":    {"epsilon": 0.20, "init": "neel", "samples": [8],
                  "schedule": SCHEDULE_S8LONG},
    "eps015neel": {"epsilon": 0.15, "init": "neel", "samples": ALL_SAMPLES,
                   "schedule": SCHEDULE_EPS015},
    "eps02zero": {"epsilon": 0.20, "init": "zero", "samples": ALL_SAMPLES,
                  "schedule": SCHEDULE_BASE},
    # --- follow-ups for the runs the first pass left unconverged ---
    "zerolong":   {"epsilon": 0.20, "init": "zero", "samples": ZERO_SLOW,
                   "schedule": SCHEDULE_S8LONG},
    "eps015long": {"epsilon": 0.15, "init": "neel", "samples": [8],
                   "schedule": SCHEDULE_EPS015LONG},
    # --- the epsilon sweep: R ~ epsilon^2 over a 4x range in epsilon, i.e.
    #     16x in R. Sample 8 takes the long schedule from the start, being the
    #     only sample that needed it at BOTH 0.2 and 0.15.
    "eps010neel": {"epsilon": 0.10, "init": "neel", "samples": ALL_SAMPLES,
                   "schedule": SCHEDULE_EPS010},
    "eps010long": {"epsilon": 0.10, "init": "neel", "samples": [8],
                   "schedule": SCHEDULE_EPS010LONG},
    "eps005neel": {"epsilon": 0.05, "init": "neel", "samples": ALL_SAMPLES,
                   "schedule": SCHEDULE_EPS005},
    "eps005long": {"epsilon": 0.05, "init": "neel", "samples": [8],
                   "schedule": SCHEDULE_EPS005LONG},
}

# Where a sample was run twice, the longer schedule supersedes the shorter.
# Sample 8 is the worked example: its 2940-unit |neel> value (4.1475e-04) sat
# at 1.86x the finite-N law and made it look like an outlier, while the
# 15015-unit value (4.4365e-04) sits at 1.991x, exactly with the other nine.
SUPERSEDES = {("eps02zero", s): ("zerolong", s) for s in ZERO_SLOW}
SUPERSEDES[("eps015neel", 8)] = ("eps015long", 8)
SUPERSEDES[("eps010neel", 8)] = ("eps010long", 8)
SUPERSEDES[("eps005neel", 8)] = ("eps005long", 8)

# Every epsilon in the sweep, against the epsilon=0.2 reference.
EPS_GRID_KINDS = {0.15: "eps015neel", 0.10: "eps010neel", 0.05: "eps005neel"}

# Record ~1000 trajectory points regardless of schedule length. The stage
# callback measures a 100-point correlator profile AND an Arnoldi correlation
# length, ~0.1-0.2 s; at 55360 stages (epsilon=0.05, sample 8) measuring every
# stage would cost hours of pure diagnostics on top of the evolution. The
# threshold keeps every schedule run so far at every-stage sampling, so
# existing trajectories stay exactly as they were.
MEASURE_TARGET_POINTS = 1000
MEASURE_EVERY_THRESHOLD = 4000


def measure_every(schedule: list[float]) -> int:
    """Stage-sampling stride, a pure function of the schedule (hence of run_config)."""
    n = len(schedule)
    if n <= MEASURE_EVERY_THRESHOLD:
        return 1
    return max(1, n // MEASURE_TARGET_POINTS)
ROLLING_WINDOW = 200.0
PREDICTED_EPS_RATIO = (0.15 / 0.20) ** 2  # 0.5625


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def job_key(job: dict) -> str:
    return f"{job['kind']}_sample{job['sample']}_{job['init']}_chi{CHI}"


def run_config(job: dict) -> dict:
    """Settings that change the result but are NOT covered by job_key.

    epsilon is in here deliberately: the key cannot express it, and a stale
    cache entry from a different epsilon would be silently wrong rather than
    merely stale.
    """
    return {
        "dt_schedule": list(job["schedule"]), "steps_per_dt": STEPS_PER_DT,
        "canonicalize_every": CANONICALIZE_EVERY, "cutoff": ex.CUTOFF,
        "r_max": R_MAX, "epsilon": job["epsilon"], "init": job["init"],
    }


def run_one(job: dict) -> dict:
    """Run one (kind, sample) iTEBD job, recording a stage-by-stage trajectory.

    Module-level so it survives pickling to a worker (Windows spawn). Uses
    only job['L_pp'], never the module-level EPSILON, so the worker's freshly
    imported copy of renyi2_swssb cannot contaminate the run.

    Input: job, dict with 'kind', 'sample', 'init', 'epsilon', 'schedule', 'L_pp'.
    Output: dict with trajectory, final_profile, cached, seconds.
    """
    path = os.path.join(ex.CACHE_DIR, job_key(job) + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                out = pickle.load(f)
            if out.get("run_config") == run_config(job):
                out["cached"] = True
                return out
        except Exception:
            pass

    t0 = time.perf_counter()
    trajectory: list[dict] = []
    times = elapsed_times(job["schedule"], STEPS_PER_DT)

    stride = measure_every(job["schedule"])
    last_stage = len(job["schedule"]) - 1

    def stage_callback(stage: int, dt: float, state) -> None:
        if stage % stride and stage != last_stage:
            return
        prof_map = dict(iobservables.correlator_profile(state, models.X, r_max=max(REFERENCE_R)))
        xi_diag = iobservables.correlation_length(state)
        trajectory.append({
            "t": times[stage],
            "R": {r: prof_map.get(r, float("nan")) for r in REFERENCE_R},
            "xi": xi_diag["xi"], "reason": xi_diag["reason"],
            "bond": max(state.bond_dims.values()),
        })

    L2_terms = ex.build_L2_terms(job["L_pp"])
    state, history = itebd.find_steady_state_infinite(
        H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[],
        dt_schedule=job["schedule"], steps_per_dt=STEPS_PER_DT,
        chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
        initial_state=ex_inf.build_initial_state(job["init"]),
        stage_callback=stage_callback,
    )

    out = {
        "kind": job["kind"], "sample": job["sample"], "init": job["init"],
        "epsilon": job["epsilon"],
        "trajectory": trajectory,
        "final_profile": iobservables.correlator_profile(state, models.X, r_max=R_MAX),
        "max_discarded_weight": max(history["discarded_weight"], default=0.0),
        "run_config": run_config(job), "cached": False,
        "seconds": time.perf_counter() - t0,
    }
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return out


def build_jobs() -> list[dict]:
    """Draw L'' at each epsilon in the PARENT and attach it to every job.

    Asserts the epsilon rescaling behaves as the tau ~ 1/epsilon^2 and
    R ~ epsilon^2 arguments both assume: same L'' direction, norm set to
    epsilon, so the pair-creation matrix element scales by exactly (0.15/0.2)^2.
    """
    by_eps = {}
    original = ex.EPSILON
    try:
        for eps in sorted({g["epsilon"] for g in GRIDS.values()}):
            ex.EPSILON = eps
            by_eps[eps] = ex.draw_samples()
    finally:
        ex.EPSILON = original

    if 0.15 in by_eps and 0.20 in by_eps:
        for s in ALL_SAMPLES:
            c20 = abs(by_eps[0.20][s]["L_pp"][3, 0]) ** 2
            c15 = abs(by_eps[0.15][s]["L_pp"][3, 0]) ** 2
            assert abs(c15 / c20 - PREDICTED_EPS_RATIO) < 1e-9, (
                f"sample {s}: creation-rate ratio {c15/c20} != {PREDICTED_EPS_RATIO}; "
                "the epsilon rescaling is not norm-only, so R ~ epsilon^2 does not follow")

    jobs = []
    for kind, g in GRIDS.items():
        for s in g["samples"]:
            jobs.append({"kind": kind, "sample": s, "init": g["init"],
                         "epsilon": g["epsilon"], "schedule": g["schedule"],
                         "L_pp": by_eps[g["epsilon"]][s]["L_pp"],
                         "creation_rate": abs(by_eps[g["epsilon"]][s]["L_pp"][3, 0]) ** 2})
    # Longest first: the 15015-unit sample-8 job must start immediately or it
    # alone sets the wall clock at the end.
    jobs.sort(key=lambda j: -sum(j["schedule"]))
    return jobs


def plateau(traj: list[dict], r: int, frac: float = 0.3) -> float:
    """Median of R(r) over the last `frac` of evolved time.

    Median not mean: individual stages carry occasional large spikes (up to
    ~20x), which a mean tracks and a median does not.
    """
    t = np.array([p["t"] for p in traj])
    v = np.array([p["R"][r] for p in traj])
    return float(np.nanmedian(v[t > (1 - frac) * t.max()]))


def pick(results: dict, kind: str, sample: int) -> dict | None:
    """The result for (kind, sample), preferring a longer re-run if one exists.

    Quoting a short run when a long one is available is not a small error
    here: sample 8's 2940-unit value was 6.5% low, which was enough to make it
    read as an outlier both against the finite-N law and against R ~ epsilon^2.
    """
    long_key = SUPERSEDES.get((kind, sample))
    if long_key is not None and long_key in results:
        return results[long_key]
    return results.get((kind, sample))


def median_by_thirds(traj: list[dict], r: int) -> list[float]:
    v = np.array([p["R"][r] for p in traj])
    n = len(v)
    return [float(np.nanmedian(v[lo:hi]))
            for lo, hi in [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]]


def rolling(t: np.ndarray, v: np.ndarray, window: float) -> np.ndarray:
    half, n = window / 2.0, len(t)
    out = np.empty_like(v)
    lo = hi = 0
    for i in range(n):
        while lo < n and t[lo] < t[i] - half:
            lo += 1
        while hi < n and t[hi] <= t[i] + half:
            hi += 1
        out[i] = np.nanmean(v[lo:hi])
    return out


def load_neel_eps02(results: dict | None = None) -> dict:
    """The epsilon=0.2 |neel> reference, one entry per sample.

    Sample 8 is taken from THIS script's 15015-unit run when available, not
    from the 2940-unit grid: that sample is the slowest relaxer in the
    ensemble and its short run is 6.5% low (see pick()).
    """
    out = {}
    for s in ALL_SAMPLES:
        path = os.path.join(ex.CACHE_DIR, f"allsamplests_sample{s}_neel_chi{CHI}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    out[s] = pickle.load(f)
            except Exception:
                pass
    if results and ("s8long", 8) in results:
        out[8] = results[("s8long", 8)]
    return out


def main() -> None:
    jobs = build_jobs()
    total_units = sum(sum(j["schedule"]) * STEPS_PER_DT for j in jobs)
    print(f"{len(jobs)} iTEBD jobs on {ex.N_WORKERS} workers, chi={CHI}", flush=True)
    for kind, g in GRIDS.items():
        n = len(g["samples"])
        print(f"  {kind:>11}: {n:>2} sample(s), eps={g['epsilon']}, |{g['init']}>, "
              f"{sum(g['schedule'])*STEPS_PER_DT:.0f} time units each", flush=True)
    print(f"  total evolved time across all jobs: {total_units:.0f} units\n", flush=True)

    results: dict = {}
    t0 = time.perf_counter()
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            results[(out["kind"], out["sample"])] = out
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
            last = out["trajectory"][-1]
            print(f"  {out['kind']:>11} s{out['sample']}  t={last['t']:.0f}  "
                  f"R(100)={last['R'][100]:.4e}  bond={last['bond']:3d}/{CHI}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_eps_init_grids.pkl")
    with open(path, "wb") as f:
        pickle.dump({"config": {k: {kk: vv for kk, vv in v.items() if kk != "schedule"}
                                for k, v in GRIDS.items()},
                     "chi": CHI, "results": results}, f)
    print(f"\nSaved -> {path}", flush=True)

    for fn in (report_law, report_eps015, report_zero_vs_neel, report_sample8):
        try:
            fn(results)
        except Exception:
            print(f"!! {fn.__name__} failed (data is saved):\n"
                  f"{traceback.format_exc()}", flush=True)


FINITE_LAW = 0.1256  # R = FINITE_LAW * |<11|L''|00>|^2, calibrated at N=12 (+-0.9%)


def report_law(results: dict) -> None:
    """R_iTEBD against the finite-N law -- i.e. the factor of ~2, sample by sample.

    CLAUDE.md establishes R = 0.1256 * |<11|L''|00>|^2 at N=12 across the ten
    samples with +-0.9% spread. The infinite-system value sits at ~2x that, a
    discrepancy traced (this session) to the denominator Tr[rho^dag rho]: the R
    numerator agrees with exact dynamics to 4.4% while the purity is ~half,
    which is the algebraic signature of an equal mixture of two orthogonal
    states that A = X_i X_j maps into each other -- purity halves, numerator is
    unchanged, so R doubles. That is the structure of a symmetry-broken pair.

    The tightness of this ratio across independent L'' draws is the evidence
    that matters: a universal constant points at structure, a scattered one at
    per-sample convergence error.
    """
    base = load_neel_eps02(results)
    samples = ex.draw_samples()  # epsilon=0.2, module default
    print(f"\n{'='*88}\nR_iTEBD vs the finite-N law R = {FINITE_LAW} * "
          f"|<11|L''|00>|^2\n{'='*88}")
    print(f"{'s':>2} {'rate':>10} {'law*rate':>12} {'R_iTEBD':>12} {'ratio':>8}")
    ratios = []
    for s in ALL_SAMPLES:
        if s not in base:
            continue
        rate = abs(samples[s]["L_pp"][3, 0]) ** 2
        v = plateau(base[s]["trajectory"], 100)
        pred = FINITE_LAW * rate
        ratios.append(v / pred)
        print(f"{s:>2} {rate:>10.6f} {pred:>12.4e} {v:>12.4e} {v/pred:>8.4f}")
    if ratios:
        a = np.array(ratios)
        print(f"\nmean {a.mean():.4f} +- {a.std():.4f}  "
              f"(min {a.min():.4f}, max {a.max():.4f}, spread {a.std()/a.mean():.2%})")


def report_eps015(results: dict) -> None:
    """Test R(0.15)/R(0.2) = 0.5625 sample by sample."""
    base = load_neel_eps02(results)
    plt = ex._mpl()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {0.15: "tab:red", 0.10: "tab:blue", 0.05: "tab:green"}

    for eps, kind in sorted(EPS_GRID_KINDS.items(), reverse=True):
        predicted = (eps / 0.20) ** 2
        rows = []
        for s in ALL_SAMPLES:
            r_eps = pick(results, kind, s)
            if r_eps is None or s not in base:
                continue
            v, v20 = plateau(r_eps["trajectory"], 100), plateau(base[s]["trajectory"], 100)
            th = median_by_thirds(r_eps["trajectory"], 100)
            rows.append((s, v20, v, v / v20 if v20 else float("nan"),
                         (th[2] / th[0] - 1) if th[0] else float("nan")))
        if not rows:
            continue

        print(f"\n{'='*88}\nepsilon = {eps} vs 0.20, |neel>: prediction R ~ epsilon^2 "
              f"=> ratio {predicted:.4f}\n{'='*88}")
        print(f"{'s':>2} {'R(eps=0.20)':>13} {f'R(eps={eps})':>14} {'ratio':>9} "
              f"{'vs pred':>9}  {'drift(3rds)':>11}")
        for s, v20, v, ratio, drift in rows:
            print(f"{s:>2} {v20:>13.4e} {v:>14.4e} {ratio:>9.5f} "
                  f"{ratio/predicted:>8.3f}x {drift:>+10.0%}")
        a = np.array([r[3] for r in rows])
        print(f"\nmean ratio {a.mean():.5f} +- {a.std():.5f}  (predicted {predicted:.4f}, "
              f"mean/predicted = {a.mean()/predicted:.4f}, spread {a.std()/a.mean():.2%})")

        # Plot ratio NORMALIZED to its own prediction, so all three epsilons
        # share one axis at 1.0 -- otherwise 0.5625 and 0.0625 cannot be
        # compared by eye and the small-epsilon scatter is invisible.
        ax.plot([r[0] for r in rows], a / predicted, "o", ms=8, color=colors.get(eps),
                label=rf"$\epsilon={eps}$ (predicted {predicted:.4f})")

    ax.axhline(1.0, color="black", ls="--", lw=1.6, label=r"exact $\epsilon^2$ scaling")
    ax.set_xlabel("sample")
    ax.set_ylabel(r"measured ratio $/\ (\epsilon/0.2)^2$")
    ax.set_title(r"Testing $R \propto \epsilon^2$ in the thermodynamic limit ($r=100$)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(ex.RESULTS_DIR, "imps_eps015_prediction.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved -> {p}", flush=True)


def report_zero_vs_neel(results: dict) -> None:
    """|0...0> vs |neel> at epsilon=0.2: the steady-state uniqueness check."""
    base = load_neel_eps02(results)
    print(f"\n{'='*88}\nepsilon = 0.20: |0...0> vs |neel> (unique steady state => equal)"
          f"\n{'='*88}")
    print(f"{'s':>2} {'R |neel>':>13} {'R |zero>':>13} {'zero/neel':>10} "
          f"{'spread':>8}  {'drift(3rds)':>11}")
    zs, ns, xs = [], [], []
    for s in ALL_SAMPLES:
        rz = pick(results, "eps02zero", s)
        if rz is None or s not in base:
            continue
        vz, vn = plateau(rz["trajectory"], 100), plateau(base[s]["trajectory"], 100)
        th = median_by_thirds(rz["trajectory"], 100)
        drift = (th[2] / th[0] - 1) if th[0] else float("nan")
        print(f"{s:>2} {vn:>13.4e} {vz:>13.4e} {vz/vn if vn else float('nan'):>10.4f} "
              f"{abs(vz-vn)/max(abs(vz+vn)/2,1e-300):>7.1%} {drift:>+10.0%}")
        zs.append(vz); ns.append(vn); xs.append(s)

    plt = ex._mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ns, "o", ms=9, color="tab:blue", label=r"$|neel\rangle$ (from above)")
    ax.plot(xs, zs, "s", ms=8, color="tab:red", mfc="none", label=r"$|0\ldots0\rangle$ (from below)")
    ax.set_yscale("log")
    ax.set_xlabel("sample"); ax.set_ylabel("$R(0,100)$")
    ax.set_title(r"Steady-state uniqueness, infinite chain ($\epsilon=0.2$, $r=100$)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    p = os.path.join(ex.RESULTS_DIR, "imps_zero_vs_neel_infinite.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved -> {p}", flush=True)


def report_sample8(results: dict) -> None:
    """Did sample 8 finally settle at 15015 units?"""
    out = results.get(("s8long", 8))
    if out is None:
        return
    traj = out["trajectory"]
    t = np.array([p["t"] for p in traj])
    print(f"\n{'='*88}\nsample 8, epsilon=0.2, |neel>, {t.max():.0f} time units "
          f"(5.1x the standard schedule)\n{'='*88}")
    print(f"{'r':>4} {'1st third':>12} {'2nd third':>12} {'3rd third':>12} "
          f"{'drift':>8} {'last 30%':>12}")
    for r in REFERENCE_R:
        th = median_by_thirds(traj, r)
        drift = (th[2] / th[0] - 1) if th[0] else float("nan")
        print(f"{r:>4} {th[0]:>12.4e} {th[1]:>12.4e} {th[2]:>12.4e} "
              f"{drift:>+7.0%} {plateau(traj, r):>12.4e}")
    print("\nAt 2940 units this sample drifted -16% to -26%. If the drift here is a few\n"
          "percent it has settled; if it is still tens of percent, the gap argument\n"
          "(smallest creation rate => slowest relaxation) needs revisiting rather than\n"
          "more time.")

    plt = ex._mpl()
    fig, axes = plt.subplots(1, len(REFERENCE_R), figsize=(4.2 * len(REFERENCE_R), 4.0),
                             squeeze=False)
    for col, r in enumerate(REFERENCE_R):
        ax = axes[0][col]
        v = np.array([p["R"][r] for p in traj])
        roll = rolling(t, v, ROLLING_WINDOW)
        ax.plot(t, v, "-", color="tab:red", lw=0.4, alpha=0.15)
        ax.plot(t, roll, "-", color="tab:red", lw=1.5, label=f"rolling mean ({ROLLING_WINDOW:.0f})")
        ax.axvline(2940, color="grey", ls=":", lw=1.3, label="old schedule end")
        lo, hi = np.nanpercentile(roll, [1, 99])
        ax.set_ylim(lo - 0.15 * (hi - lo + 1e-12), hi + 0.15 * (hi - lo + 1e-12))
        ax.set_xscale("log")
        ax.set_xlabel("evolved time"); ax.set_ylabel(rf"$R(0,{r})$")
        ax.set_title(f"sample 8, r = {r}"); ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)
    fig.suptitle(rf"Sample 8, $\epsilon=0.2$, $|neel\rangle$, {t.max():.0f} time units")
    fig.tight_layout()
    p = os.path.join(ex.RESULTS_DIR, "imps_sample8_long.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved -> {p}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
