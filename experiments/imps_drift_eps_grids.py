"""The driven classical circuit in the THERMODYNAMIC LIMIT: iTEBD at p = 0.8,
epsilon in {0.20, 0.15, 0.10, 0.05}.

Why this run exists
-------------------
The finite-N grid for this model (renyi2_drift_annihilation.py) concluded
"R decays exponentially, no SWSSB, xi = 4-9 sites", against the first model's
flat R and R_iTEBD = q/4 in the infinite system. That comparison was never
apples-to-apples: no iTEBD run of THIS model existed, so a finite-size result
was being read against an infinite-size one for a different Lindbladian.

A pilot (2026-08-12) makes the gap concrete and inverts the expected reading.
Sample 6 from |0...0> at epsilon=0.2 relaxes to

    R(1) = 8.4291e-02 ... R(40) = 8.4213e-02      flat to 4 digits

i.e. long-range order, in the very model whose finite-N profile decays with a
constant step ratio of 0.79. The suspect is evolved TIME, not size: the finite
grid's schedule is 300 * (0.1+0.05+0.02+0.01+0.005) = 55.5 time units, while
sample 6 needed ~155 units here (3.4/q) and sample 8 was still visibly rising
at 600 units (1.06/q). The first model tolerated 55.5 units because its
baseline rates are 4-16 rather than 1, buying ~16x more relaxation per unit
time; the drift model's rates are 1, so the same schedule is ~16x shorter in
units of the dynamics.

The zero/neel spread cannot see this failure mode, which is why the finite grid
read as clean (<=0.68% at N=20). Both starts equilibrate locally within a few
time units and then crawl along the SAME slow manifold at rate ~q; they agree
with each other while both sit far from the fixed point. Bracketing the truth
requires the two starts to approach from opposite sides of a value they have
both nearly reached -- an assumption, not a measurement.

What this grid measures
-----------------------
Per epsilon, all ten L'' samples, from |0...0>:

  1. **Flatness of R(r)** -- R(100)/R(1) and R(20)/R(1). No fitting and no
     model selection (see CLAUDE.md Trap 4), and no boundary to avoid: the
     infinite chain settles the SWSSB question directly.
  2. **R/q against epsilon**, q = |<11|L''|00>|^2. The first model gives
     R = q/4 exactly in this limit; the finite drift model gives R = 2q/gamma_A
     as epsilon -> 0. Whether the infinite drift model has its own such law,
     and whether R ~ epsilon^2 survives, is the quantitative payload.
  3. **xi** from the profile's log-slope and from the transfer operator's
     sub-leading eigenvalue (iobservables.correlation_length), which returns
     None rather than a fabricated number when the two leading eigenvalues are
     near-degenerate -- exactly the SWSSB case.

Sizing the schedule: tau ~ 1/q, per sample
------------------------------------------
Relaxation is set by the pair-creation rate q (CLAUDE.md: the same matrix
element sets both R and the Liouvillian gap), and q spans 12.5x across the
ensemble, so a schedule uniform in time units over-runs sample 6 by an order
of magnitude while leaving sample 8 unrelaxed. Each job therefore gets

    target = TARGET_GAP_TIMES / q,    factor = ceil(target / 2940 units)

evolved time, as an integer number of copies of the standard 2940-unit shape.
Repeating whole stages (rather than raising steps_per_dt or coarsening dt)
keeps the dt anneal -- hence the Trotter floor -- identical across every run in
the grid, so the long and short runs stay directly comparable. Since q ~
epsilon^2, this reproduces the established 1/epsilon^2 schedule scaling
automatically instead of hard-coding it per epsilon.

TARGET_GAP_TIMES = 12 is ~3.5x the convergence time measured in the pilot
(sample 6 flat to 4-5 digits by 3.4/q); the per-trajectory drift-by-thirds
table is what actually certifies each run, not this constant.

Why |0...0> and not |neel>
--------------------------
Cost, by two orders of magnitude at small epsilon. |0...0> is a dark state of
the drift baseline, so at small epsilon the state never leaves the
low-entanglement neighbourhood of the vacuum: measured at epsilon=0.05,
bond 20 and 34 ms per time unit from |0...0> against bond 128 (capped!) and
3623 ms from |neel>, whose violent transient also drove R NEGATIVE
(truncation destroying positivity) before it settled. |neel> is kept as the
uniqueness cross-check at epsilon=0.2 only, where it costs the same as
|0...0>, on the three samples that bracket the ensemble in q.

Controls
--------
  * **L'' = 0**, both inits: the exact steady state is the pure dark vacuum, so
    R must return ~0. This is the false-positive floor of the infinite
    pipeline for this model (the finite one measures 7.09e-10).
  * **3x-length re-run**, sample 4 at epsilon=0.2: a sample whose production
    factor is 1, re-run at factor 3. If R moves, TARGET_GAP_TIMES is too small
    and every factor-1 job in the grid is suspect.

Cache safety
------------
Namespaced 'driftinf_...', and run_config carries epsilon, the init, the model
parameters (p, hop_rate, annihilation_rate) and the schedule shape + factor --
none of which the key expresses. Two epsilons, two models or two schedule
lengths therefore cannot alias onto one entry. The schedule is stored as
(shape, factor) rather than expanded: the pair determines it exactly, and the
expanded list runs to 30k entries at epsilon=0.05.

Outputs (experiments/results/):
    imps_drift_eps_grids.pkl        every trajectory + final profile
    imps_drift_flatness.png         R(r)/R(1) per sample, one panel per epsilon
    imps_drift_R_vs_epsilon.png     R/q vs epsilon, and R vs q per epsilon
    imps_drift_finite_vs_infinite.png   the profile the finite N=20 grid saw,
                                    against the infinite one
"""

import math
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

# ---------------------------------------------------------------------------
# Model -- identical to the finite grid's config (renyi2_drift_annihilation.py)
# ---------------------------------------------------------------------------
P_RIGHT = 0.8
HOP_RATE = 1.0
ANNIHILATION_RATE = 1.0
MODEL = {"name": "classical_drift_annihilation", "p": P_RIGHT,
         "hop_rate": HOP_RATE, "annihilation_rate": ANNIHILATION_RATE}

EPSILONS = [0.20, 0.15, 0.10, 0.05]
ALL_SAMPLES = list(range(10))

CHI = 128
R_MAX = 100
REFERENCE_R = [1, 2, 5, 10, 20, 50, 100]
STEPS_PER_DT = 50
CANONICALIZE_EVERY = 10

# The standard 2940-time-unit shape, as used by the first model's iTEBD grids.
BASE_SHAPE = [0.1] * 500 + [0.05] * 120 + [0.02] * 100 + [0.01] * 60 + [0.005] * 40
BASE_UNITS = sum(BASE_SHAPE) * STEPS_PER_DT  # 2940.0

# Evolved time in units of 1/q (the pair-creation rate, hence ~1/gap). See the
# module docstring: measured convergence is ~3.4/q, this is ~3.5x that.
TARGET_GAP_TIMES = 12.0

# |neel> is only affordable at epsilon=0.2 (see docstring). These three samples
# bracket the ensemble: 8 the smallest q (slowest), 6 the largest, 3 mid-pack.
NEEL_SAMPLES = [8, 3, 6]
# A factor-1 production sample, re-run 3x longer to test TARGET_GAP_TIMES.
LONG_CONTROL_SAMPLE = 4
LONG_CONTROL_MULTIPLE = 3

# Record ~1000 trajectory points regardless of schedule length: each measurement
# is a 100-point correlator profile plus an Arnoldi solve, and the epsilon=0.05
# schedules run to ~30k stages.
MEASURE_TARGET_POINTS = 1000

# R(r) log-slope window for xi. Starts past the first few sites (where the
# profile carries non-asymptotic structure) and stops well inside r_max.
XI_FIT_RANGE = (10, 50)
# R(100)/R(1) above this counts as flat, i.e. long-range order rather than a
# resolvable xi. Deliberately loose: the pilot's flat sample sits at 0.999.
FLAT_RATIO = 0.5

KIND_PREFIX = "driftinf"
FINITE_PICKLE = "renyi2_drift_annihilation.pkl"  # for the finite-vs-infinite figure
FINITE_N = 20


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def schedule_factor(q: float) -> int:
    """Number of BASE_SHAPE copies needed to evolve for TARGET_GAP_TIMES / q.

    Input: q, the sample's pair-creation rate |<11|L''|00>|^2 at this epsilon.
    Output: int >= 1, the repeat count (see expand_schedule).
    """
    if q <= 0.0:
        return 1
    return max(1, math.ceil(TARGET_GAP_TIMES / q / BASE_UNITS))


def expand_schedule(factor: int) -> list[float]:
    """Repeat each BASE_SHAPE stage `factor` times: same dt anneal, factor x time.

    Lengthening by repeating stages -- rather than by raising steps_per_dt or
    coarsening dt -- leaves the final-stage Trotter floor unchanged, so runs of
    different length in this grid remain directly comparable.

    Input: factor, the repeat count from schedule_factor().
    Output: the dt schedule, a list of length factor * len(BASE_SHAPE).
    """
    out: list[float] = []
    for dt in BASE_SHAPE:
        out.extend([dt] * factor)
    return out


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    """Cumulative evolved time at the end of each dt stage.

    Input: dt_schedule; steps_per_dt.
    Output: list of floats, one per stage.
    """
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def measure_every(n_stages: int) -> int:
    """Stage-sampling stride, a pure function of the schedule length.

    Input: n_stages, len(dt_schedule).
    Output: int >= 1 stride giving ~MEASURE_TARGET_POINTS trajectory samples.
    """
    return max(1, n_stages // MEASURE_TARGET_POINTS)


# ---------------------------------------------------------------------------
# Job plumbing
# ---------------------------------------------------------------------------
def job_key(job: dict) -> str:
    """Filename-safe cache key for one job."""
    return f"{KIND_PREFIX}_{job['kind']}_{job['label']}_{job['init']}_chi{CHI}"


def run_config(job: dict) -> dict:
    """Settings that change the result but are NOT covered by job_key().

    epsilon, the init, the model parameters and the schedule all live here: the
    key expresses none of them, and a stale entry from a different epsilon or a
    shorter schedule would be silently wrong rather than merely stale. The
    schedule is recorded as (shape, factor), which determines it exactly.

    Input: job dict.
    Output: dict for comparison against a cache entry's stored copy.
    """
    return {
        "model": MODEL,
        "epsilon": job["epsilon"],
        "init": job["init"],
        "schedule_shape": list(BASE_SHAPE),
        "schedule_factor": job["factor"],
        "steps_per_dt": STEPS_PER_DT,
        "canonicalize_every": CANONICALIZE_EVERY,
        "cutoff": ex.CUTOFF,
        "chi_max": CHI,
        "r_max": R_MAX,
    }


def run_one(job: dict) -> dict:
    """Run one iTEBD job, recording a strided stage-by-stage trajectory.

    Module-level so it survives pickling to a worker (Windows spawn). Reads
    job['L_pp'] only, never ex.EPSILON, so a worker's freshly imported copy of
    renyi2_swssb cannot contaminate the run. Exceptions are returned, not
    raised: one failure must not abort the pool.

    Input: job dict with 'kind', 'label', 'sample', 'init', 'epsilon', 'q',
        'factor', 'L_pp' (None for the L''=0 control).
    Output: the job dict (minus L_pp) augmented with 'trajectory',
        'final_profile', 'xi', 'bond_dims', 'seconds', 'cached' -- or 'error'.
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
    out = {k: v for k, v in job.items() if k != "L_pp"}
    out["cached"] = False
    out["run_config"] = run_config(job)

    try:
        schedule = expand_schedule(job["factor"])
        times = elapsed_times(schedule, STEPS_PER_DT)
        stride = measure_every(len(schedule))
        last_stage = len(schedule) - 1
        trajectory: list[dict] = []

        def stage_callback(stage: int, dt: float, state) -> None:
            if stage % stride and stage != last_stage:
                return
            prof = dict(iobservables.correlator_profile(state, models.X, r_max=R_MAX))
            xi_diag = iobservables.correlation_length(state)
            trajectory.append({
                "t": times[stage],
                "R": {r: prof.get(r, float("nan")) for r in REFERENCE_R},
                "xi": xi_diag["xi"], "reason": xi_diag["reason"],
                "bond": max(state.bond_dims.values()),
            })

        baseline = models.classical_drift_annihilation_jump_operators(
            P_RIGHT, HOP_RATE, ANNIHILATION_RATE)
        L2_terms = ex.build_L2_terms(job["L_pp"], baseline)

        state, history = itebd.find_steady_state_infinite(
            H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[],
            dt_schedule=schedule, steps_per_dt=STEPS_PER_DT,
            chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
            initial_state=ex_inf.build_initial_state(job["init"]),
            stage_callback=stage_callback,
        )

        xi_diag = iobservables.correlation_length(state)
        out.update({
            "trajectory": trajectory,
            "final_profile": iobservables.correlator_profile(state, models.X, r_max=R_MAX),
            "xi": xi_diag["xi"], "xi_reason": xi_diag["reason"],
            "eta1": xi_diag["eta1"], "eta2": xi_diag["eta2"],
            "bond_dims": dict(state.bond_dims),
            "max_discarded_weight": max(history["discarded_weight"], default=0.0),
            "final_eigenvalue_drift": (history["eigenvalue_drift"][-1]
                                       if history["eigenvalue_drift"] else float("nan")),
            "evolved_time": times[-1],
        })
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


def draw_at(epsilon: float) -> list[dict]:
    """The ten L'' samples at a given epsilon, drawn in the PARENT process.

    ex.EPSILON is module state and workers re-import the module fresh under
    Windows spawn, so every L'' is drawn here and passed by value.

    Input: epsilon, target operator norm of L''.
    Output: list of ex.draw_samples() dicts.
    """
    original = ex.EPSILON
    try:
        ex.EPSILON = epsilon
        return ex.draw_samples()
    finally:
        ex.EPSILON = original


def build_jobs() -> list[dict]:
    """Enumerate every job in the grid, longest first.

    Asserts that rescaling L'' to a smaller epsilon is norm-only, so that
    q ~ epsilon^2 and hence the 1/epsilon^2 schedule scaling both follow (the
    same guard imps_eps_init_grids.py uses).

    Output: list of job dicts for run_one(), sorted by descending evolved time
        so the longest job starts immediately and does not set the tail.
    """
    by_eps = {eps: draw_at(eps) for eps in EPSILONS}
    q_of = {eps: [abs(s["L_pp"][3, 0]) ** 2 for s in by_eps[eps]] for eps in EPSILONS}

    ref = max(EPSILONS)
    for eps in EPSILONS:
        predicted = (eps / ref) ** 2
        for s in ALL_SAMPLES:
            got = q_of[eps][s] / q_of[ref][s]
            assert abs(got / predicted - 1.0) < 1e-9, (
                f"sample {s}: q(eps={eps})/q(eps={ref}) = {got}, expected {predicted}; "
                "the epsilon rescaling is not norm-only, so q ~ epsilon^2 fails")

    def make(kind, label, sample, init, eps, L_pp, q, factor):
        return {"kind": kind, "label": label, "sample": sample, "init": init,
                "epsilon": eps, "L_pp": L_pp, "q": q, "factor": factor}

    jobs = []
    for eps in EPSILONS:
        kind = f"eps{round(eps*100):03d}"
        for s in ALL_SAMPLES:
            q = q_of[eps][s]
            jobs.append(make(kind, f"sample{s}", s, "zero", eps,
                             by_eps[eps][s]["L_pp"], q, schedule_factor(q)))

    for s in NEEL_SAMPLES:
        q = q_of[ref][s]
        jobs.append(make(f"eps{round(ref*100):03d}n", f"sample{s}", s, "neel", ref,
                         by_eps[ref][s]["L_pp"], q, schedule_factor(q)))

    for init in ("zero", "neel"):
        jobs.append(make("ctrl", "baseline", None, init, 0.0, None, 0.0, 1))

    s = LONG_CONTROL_SAMPLE
    q = q_of[ref][s]
    jobs.append(make("long", f"sample{s}", s, "zero", ref, by_eps[ref][s]["L_pp"], q,
                     schedule_factor(q) * LONG_CONTROL_MULTIPLE))

    jobs.sort(key=lambda j: -j["factor"])
    return jobs


# ---------------------------------------------------------------------------
# Trajectory / profile readers
# ---------------------------------------------------------------------------
def plateau(traj: list[dict], r: int, frac: float = 0.3) -> float:
    """Median of R(r) over the last `frac` of evolved time.

    Median, not mean: individual stages carry occasional large spikes that a
    mean tracks and a median does not.

    Input: traj, a run's trajectory list; r, separation; frac.
    Output: float.
    """
    t = np.array([p["t"] for p in traj])
    v = np.array([p["R"][r] for p in traj])
    if t.size == 0:
        return float("nan")
    return float(np.nanmedian(v[t > (1 - frac) * t.max()]))


def median_by_thirds(traj: list[dict], r: int) -> list[float]:
    """Median of R(r) over each third of the trajectory (the drift diagnostic).

    Input: traj; r.
    Output: [first, second, third] medians.
    """
    v = np.array([p["R"][r] for p in traj])
    n = len(v)
    if n == 0:
        return [float("nan")] * 3
    return [float(np.nanmedian(v[lo:hi]))
            for lo, hi in [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]]


def drift(traj: list[dict], r: int) -> float:
    """Relative change in median R(r) from the first third to the last.

    Input: traj; r.
    Output: float (nan if the first third is zero).
    """
    th = median_by_thirds(traj, r)
    return (th[2] / th[0] - 1.0) if th[0] else float("nan")


def profile_xi(profile: list[tuple[int, float]], lo: int = None, hi: int = None) -> dict:
    """Correlation length from the log-slope of R(r) over a window.

    Reads the profile directly rather than fitting a decay model to R(N) --
    see CLAUDE.md Trap 4, where the latter inverted a conclusion.

    xi is None whenever a specific finite value is not a meaningful thing to
    report from the data -- the window holds a non-positive R (truncation can
    produce one), the fitted slope is not a decay, or the profile is flat by
    FLAT_RATIO. The last case matters: a flat profile leaves the slope at
    numerical noise, which the formula happily turns into xi = 4e+06. That is
    the SWSSB signal, not a correlation length, and it is reported as 'flat'.
    Same policy as iobservables.correlation_length on a near-degenerate
    transfer spectrum.

    Input: profile, list of (r, R(0,r)); lo, hi, the fit window (defaults to
        XI_FIT_RANGE).
    Output: dict with 'xi', 'r2', 'flat' (R(r_max)/R(1) > FLAT_RATIO), 'ratio'.
    """
    lo = XI_FIT_RANGE[0] if lo is None else lo
    hi = XI_FIT_RANGE[1] if hi is None else hi
    d = dict(profile)
    r1 = d.get(1, float("nan"))
    r_last = max(d)
    ratio = d[r_last] / r1 if r1 else float("nan")

    rs = np.array([r for r in range(lo, hi + 1) if r in d], dtype=float)
    vs = np.array([d[int(r)] for r in rs])
    out = {"xi": None, "r2": float("nan"), "flat": bool(ratio > FLAT_RATIO),
           "ratio": float(ratio)}
    if out["flat"] or rs.size < 3 or np.any(vs <= 0) or np.any(~np.isfinite(vs)):
        return out

    slope, intercept = np.polyfit(rs, np.log(vs), 1)
    pred = slope * rs + intercept
    ss_res = float(np.sum((np.log(vs) - pred) ** 2))
    ss_tot = float(np.sum((np.log(vs) - np.mean(np.log(vs))) ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if slope < 0:
        out["xi"] = float(-1.0 / slope)
    return out


def key_of(out: dict) -> tuple:
    """(kind, label, init) identity of a result, used as the results-dict key."""
    return (out["kind"], out["label"], out["init"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    jobs = build_jobs()
    total_units = sum(j["factor"] * BASE_UNITS for j in jobs)
    print(f"{len(jobs)} iTEBD jobs on {ex.N_WORKERS} workers, chi={CHI}, "
          f"model {MODEL['name']} p={P_RIGHT} (hop {HOP_RATE}, ann {ANNIHILATION_RATE})",
          flush=True)
    print(f"  schedules sized as {TARGET_GAP_TIMES:.0f}/q, rounded up to whole "
          f"{BASE_UNITS:.0f}-unit copies", flush=True)
    for j in jobs:
        print(f"  {j['kind']:>8} {j['label']:>9} |{j['init']:>4}>  eps={j['epsilon']:.2f}  "
              f"q={j['q']:.3e}  x{j['factor']:>3}  {j['factor']*BASE_UNITS:>9.0f} units",
              flush=True)
    print(f"  total evolved time: {total_units:.0f} units\n", flush=True)

    results: dict = {}
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            done += 1
            tag = (f"  [{done:2d}/{len(jobs)}] {out['kind']:>8} {out['label']:>9} "
                   f"|{out['init']:>4}> eps={out['epsilon']:.2f}")
            if "error" in out:
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                results[key_of(out)] = out
                continue
            results[key_of(out)] = out
            prof = dict(out["final_profile"])
            fit = profile_xi(out["final_profile"])
            xi_str = f"{fit['xi']:6.2f}" if fit["xi"] is not None else "  flat"
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.1f}min"
            print(f"{tag}  t={out['evolved_time']:.0f}  R(1)={prof[1]:.4e}  "
                  f"R(100)={prof[R_MAX]:.4e}  R({R_MAX})/R(1)={fit['ratio']:8.5f}  "
                  f"xi={xi_str}  bond={max(out['bond_dims'].values()):3d}/{CHI}  "
                  f"drift={drift(out['trajectory'], 1):+7.1%}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_drift_eps_grids.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": MODEL, "epsilons": EPSILONS, "chi": CHI,
                     "target_gap_times": TARGET_GAP_TIMES, "r_max": R_MAX,
                     "results": results}, f)
    print(f"\nSaved -> {path}", flush=True)

    for fn in (report_flatness, report_law, report_controls, report_zero_vs_neel,
               plot_flatness, plot_law, plot_finite_vs_infinite):
        try:
            fn(results)
        except Exception:
            print(f"!! {fn.__name__} failed (data is saved):\n"
                  f"{traceback.format_exc()}", flush=True)


def _eps_kind(eps: float, neel: bool = False) -> str:
    return f"eps{round(eps*100):03d}" + ("n" if neel else "")


def _get(results: dict, eps: float, sample: int, init: str = "zero") -> dict | None:
    key = (_eps_kind(eps, init == "neel"), f"sample{sample}", init)
    out = results.get(key)
    return None if out is None or "error" in out else out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def report_flatness(results: dict) -> None:
    """The SWSSB verdict: is R(r) flat in the thermodynamic limit?

    R(100)/R(1) needs no fit and no model selection. The finite N=20 grid for
    this model reported a constant step ratio of ~0.79 per site, i.e.
    R(12)/R(1) ~ 0.08 -- so a flat infinite profile means the finite decay was
    not the thermodynamic-limit behaviour.
    """
    print(f"\n{'='*100}\nFlatness of R(r): R({R_MAX})/R(1) and R(20)/R(1), "
          f"|0...0>, per epsilon\n{'='*100}")
    for eps in EPSILONS:
        print(f"\nepsilon = {eps}")
        print(f"{'s':>2} {'q':>11} {'t':>9} {'R(1)':>12} {f'R({R_MAX})':>12} "
              f"{'R(100)/R(1)':>12} {'R(20)/R(1)':>11} {'xi_prof':>8} {'xi_tm':>8} "
              f"{'bond':>5} {'drift':>8}")
        for s in ALL_SAMPLES:
            out = _get(results, eps, s)
            if out is None:
                print(f"{s:>2}  (missing)")
                continue
            d = dict(out["final_profile"])
            fit = profile_xi(out["final_profile"])
            xi_p = f"{fit['xi']:8.2f}" if fit["xi"] is not None else "    flat"
            xi_t = f"{out['xi']:8.2f}" if out["xi"] is not None else "    None"
            print(f"{s:>2} {out['q']:>11.4e} {out['evolved_time']:>9.0f} "
                  f"{d[1]:>12.4e} {d[R_MAX]:>12.4e} {fit['ratio']:>12.6f} "
                  f"{d[20]/d[1] if d[1] else float('nan'):>11.6f} {xi_p} {xi_t} "
                  f"{max(out['bond_dims'].values()):>5d} {drift(out['trajectory'], 1):>+8.1%}")
    print(f"\nA ratio near 1 is long-range order (SWSSB). Compare the finite N={FINITE_N} "
          f"grid,\nwhose profile fell by a constant factor ~0.79 per site.", flush=True)


def report_law(results: dict) -> None:
    """R/q per epsilon: is there an amplitude law, and does R ~ epsilon^2 hold?

    The first model gives R = q/4 exactly in this limit. The finite drift model
    gives R = 2q/gamma_A as epsilon -> 0 at p = 1/2, and F != 2 at p = 0.8.
    """
    print(f"\n{'='*100}\nR/q at r = {R_MAX} and r = 1, |0...0>\n{'='*100}")
    print(f"{'eps':>6} {'n':>3} {'mean R(1)/q':>13} {'spread':>8} "
          f"{f'mean R({R_MAX})/q':>15} {'spread':>8}  {'vs eps=0.2':>10}")
    ref = None
    for eps in EPSILONS:
        rows = [(_get(results, eps, s)) for s in ALL_SAMPLES]
        rows = [o for o in rows if o is not None]
        if not rows:
            continue
        a1 = np.array([plateau(o["trajectory"], 1) / o["q"] for o in rows])
        a2 = np.array([plateau(o["trajectory"], R_MAX) / o["q"] for o in rows])
        if ref is None:
            ref = a1.mean()
        print(f"{eps:>6.2f} {len(rows):>3} {a1.mean():>13.6f} {a1.std()/a1.mean():>7.2%} "
              f"{a2.mean():>15.6f} {a2.std()/abs(a2.mean()) if a2.mean() else float('nan'):>7.2%}"
              f"  {a1.mean()/ref:>10.4f}")
    print("\nR/q constant across epsilon means R ~ epsilon^2 (q ~ epsilon^2 by "
          "construction);\nthe residual is the O(epsilon^2) correction, which in the "
          "first model was -0.036 eps^2.", flush=True)


def report_controls(results: dict) -> None:
    """L''=0 floor, and whether the 3x-length re-run moved R."""
    print(f"\n{'='*100}\nControls\n{'='*100}")
    for init in ("zero", "neel"):
        out = results.get(("ctrl", "baseline", init))
        if out is None or "error" in out:
            print(f"  L''=0 |{init}>: missing")
            continue
        d = dict(out["final_profile"])
        print(f"  L''=0 |{init:>4}>: R(1)={d[1]:.3e}  R({R_MAX})={d[R_MAX]:.3e}  "
              f"bond={max(out['bond_dims'].values())}  "
              f"(exact answer: the pure dark vacuum, R = 0)")

    s = LONG_CONTROL_SAMPLE
    short, long = _get(results, max(EPSILONS), s), results.get(("long", f"sample{s}", "zero"))
    if short is not None and long is not None and "error" not in long:
        for r in (1, 20, R_MAX):
            a, b = plateau(short["trajectory"], r), plateau(long["trajectory"], r)
            print(f"  sample {s} eps=0.2, r={r:>3}: {short['evolved_time']:.0f} units "
                  f"-> {a:.6e},  {long['evolved_time']:.0f} units -> {b:.6e}   "
                  f"({(b/a-1) if a else float('nan'):+.2%})")
        print(f"  A few percent means TARGET_GAP_TIMES = {TARGET_GAP_TIMES:.0f} is enough; "
              f"tens of percent means every factor-1 job needs re-running longer.")


def report_zero_vs_neel(results: dict) -> None:
    """|0...0> vs |neel> at epsilon=0.2: the uniqueness / two-sided check.

    Note what this test can and cannot do here. Both starts equilibrate
    locally within a few time units and then crawl along the same slow
    manifold at rate ~q, so agreement is only evidence of convergence once
    both have plateaued -- which the drift-by-thirds column is what certifies.
    """
    ref = max(EPSILONS)
    print(f"\n{'='*100}\nepsilon = {ref}: |0...0> vs |neel> (unique steady state => equal)"
          f"\n{'='*100}")
    print(f"{'s':>2} {'r':>4} {'R |zero>':>13} {'R |neel>':>13} {'spread':>8} "
          f"{'drift z':>9} {'drift n':>9}")
    for s in NEEL_SAMPLES:
        z, n = _get(results, ref, s, "zero"), _get(results, ref, s, "neel")
        if z is None or n is None:
            print(f"{s:>2}  (missing)")
            continue
        for r in (1, 20, R_MAX):
            vz, vn = plateau(z["trajectory"], r), plateau(n["trajectory"], r)
            spread = abs(vz - vn) / max(abs(vz + vn) / 2, 1e-300)
            print(f"{s:>2} {r:>4} {vz:>13.5e} {vn:>13.5e} {spread:>7.2%} "
                  f"{drift(z['trajectory'], r):>+9.1%} {drift(n['trajectory'], r):>+9.1%}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_flatness(results: dict) -> None:
    """R(r)/R(1) vs r, one panel per epsilon, all ten samples."""
    plt = ex._mpl()
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(4.6 * len(EPSILONS), 4.4),
                             squeeze=False, sharey=True)
    for col, eps in enumerate(EPSILONS):
        ax = axes[0][col]
        for s in ALL_SAMPLES:
            out = _get(results, eps, s)
            if out is None:
                continue
            d = dict(out["final_profile"])
            rs = sorted(d)
            v0 = d[1]
            ax.plot(rs, [d[r] / v0 for r in rs], "-", lw=1.6,
                    color=cmap(s / (len(ALL_SAMPLES) - 1)), label=f"s{s}")
        ax.set_yscale("log")
        ax.set_xlabel("separation r")
        if col == 0:
            ax.set_ylabel("$R(0,r)\\,/\\,R(0,1)$")
            ax.legend(fontsize=6, ncol=2)
        ax.set_title(rf"$\epsilon={eps}$")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(rf"Infinite chain, driven circuit $p={P_RIGHT}$: flat = SWSSB "
                 f"(finite $N={FINITE_N}$ fell ~0.79 per site)")
    fig.tight_layout()
    p = os.path.join(ex.RESULTS_DIR, "imps_drift_flatness.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved -> {p}", flush=True)


def plot_law(results: dict) -> None:
    """R vs q per epsilon, and R/q vs epsilon."""
    plt = ex._mpl()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {0.20: "tab:red", 0.15: "tab:orange", 0.10: "tab:blue", 0.05: "tab:green"}
    for eps in EPSILONS:
        rows = [o for o in (_get(results, eps, s) for s in ALL_SAMPLES) if o is not None]
        if not rows:
            continue
        q = np.array([o["q"] for o in rows])
        R = np.array([plateau(o["trajectory"], 1) for o in rows])
        ax0.plot(q, R, "o", ms=7, color=colors.get(eps), label=rf"$\epsilon={eps}$")
        ax1.plot([eps] * len(rows), R / q, "o", ms=7, color=colors.get(eps), alpha=0.75)
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax0.set_xlabel(r"$q = |\langle 11|L''|00\rangle|^2$")
    ax0.set_ylabel("$R(0,1)$")
    ax0.set_title("amplitude law")
    ax0.grid(True, alpha=0.3, which="both"); ax0.legend(fontsize=8)
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel("$R(0,1)\\,/\\,q$")
    ax1.set_title(r"constant $\Leftrightarrow$ $R \propto \epsilon^2$")
    ax1.grid(True, alpha=0.3)
    fig.suptitle(rf"Driven circuit $p={P_RIGHT}$, infinite chain")
    fig.tight_layout()
    p = os.path.join(ex.RESULTS_DIR, "imps_drift_R_vs_epsilon.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved -> {p}", flush=True)


def plot_finite_vs_infinite(results: dict) -> None:
    """The finite N=20 profile against the infinite one, per sample.

    Both normalized to their own value at separation 1, since the finite grid's
    reference site is i = N//4 rather than 0 -- the shapes are the claim, not
    the absolute values.
    """
    path = os.path.join(ex.RESULTS_DIR, FINITE_PICKLE)
    if not os.path.exists(path):
        print(f"  (no {FINITE_PICKLE}; skipping finite-vs-infinite figure)", flush=True)
        return
    with open(path, "rb") as f:
        fin = pickle.load(f)

    plt = ex._mpl()
    ref = max(EPSILONS)
    show = [s for s in NEEL_SAMPLES if _get(results, ref, s) is not None]
    fig, axes = plt.subplots(1, len(show), figsize=(4.8 * max(len(show), 1), 4.4),
                             squeeze=False)
    for col, s in enumerate(show):
        ax = axes[0][col]
        out = _get(results, ref, s)
        d = dict(out["final_profile"])
        rs = [r for r in sorted(d) if r <= 20]
        ax.plot(rs, [d[r] / d[1] for r in rs], "-o", ms=4, color="tab:red",
                label=r"infinite (iTEBD)")

        entry = fin["results"]["zero"][FINITE_N][s]
        if entry is not None:
            i = entry["i"]
            prof = [(r - i, v) for r, v in entry["profile"]]
            v1 = dict(prof).get(1)
            if v1:
                ax.plot([r for r, _ in prof], [v / v1 for _, v in prof], "-s", ms=4,
                        color="tab:blue", label=rf"finite $N={FINITE_N}$")
        ax.set_yscale("log")
        ax.set_xlabel("separation r")
        if col == 0:
            ax.set_ylabel("$R(r)\\,/\\,R(1)$")
        ax.set_title(f"sample {s},  q = {out['q']:.2e}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(rf"Driven circuit $p={P_RIGHT}$, $\epsilon={ref}$: the finite grid's "
                 f"decay against the thermodynamic limit")
    fig.tight_layout()
    p = os.path.join(ex.RESULTS_DIR, "imps_drift_finite_vs_infinite.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved -> {p}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
