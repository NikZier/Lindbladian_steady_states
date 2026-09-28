"""HAMILTONIAN perturbations of the dark steady state, in the thermodynamic
limit: iTEBD at epsilon in {0.20, 0.15, 0.10, 0.05}, ten random draws each.

What this is, against what already exists
-----------------------------------------
Every SWSSB result in this project so far perturbs the baseline

    L  = X_a X_{a+1} (1 - Z_a Z_{a+1})          L' = X_a X_{a+1} (1-Z_a)(1-Z_b)

with a random Z_2-symmetric DISSIPATIVE L'', and finds

    R_iTEBD = q/4,    q = |<11|L''|00>|^2,    R ~ epsilon^2

(CLAUDE.md, "The infinite system (iTEBD) -- FIRST model"). This script asks the
same question of a random Z_2-symmetric HERMITIAN perturbation applied as a
nearest-neighbour Hamiltonian, H = sum_a h_{a,a+1}, ||h|| = epsilon. Same
baseline, same |0...0> start, same correlator, same 1/epsilon^2 schedule
discipline. iTEBD only -- there is no finite-size grid here and none is needed:
the question is long-range order and the epsilon exponent, and the
thermodynamic limit answers both without a finite-size caveat.

Coherent, not incoherent -- and why epsilon^2 is still the expectation
---------------------------------------------------------------------
h commutes with Z(x)Z, so its even-parity block connects |00> <-> |11> with
amplitude c = <11|h|00> (models.pair_creation_element). The dark vacuum is
therefore destabilized COHERENTLY at amplitude |c| ~ epsilon, where L''
destabilizes it INCOHERENTLY at rate q ~ epsilon^2. Adiabatically eliminating
the |00>-|11> coherence against the baseline annihilation rate gamma_A = 16
(L' = 4|00><11|) turns the coherent drive back into an effective incoherent
pair-creation rate ~ 4|c|^2/gamma_A, i.e. ~ epsilon^2 again -- so R ~ epsilon^2
is the prediction, with a different prefactor. The measured exponent is the
deliverable; report_epsilon_scaling() fits it FREE rather than assuming 2.

Sizing the schedule -- and the trap that nearly set it 1000x too short
-----------------------------------------------------------------------
CLAUDE.md Trap 5 cost this project a retracted conclusion: a grid ran on a
schedule never sized from the relaxation rate, agreed with itself from two
initial states, and sat 19x below its fixed point. So the rate is established
BEFORE the run, from an exact solve.

The obvious route does not work. exact.sector_steady_state reports the slowest
mode of the N-site sector, but at N = 4 and 6 that is the BASELINE diffusive
mode -- measured 3.0626 and 1.2785, independent of epsilon to 4 digits across
epsilon in {0.2, 0.1, 0.05}. A + A -> 0 is diffusion-limited and critical, and
a 4-site chain has no room for the diffusive slowing that makes the infinite
chain slow. Sizing from that number would give factor 1 everywhere.

What does work is the exact N=4 CORRELATOR. Measured on the (+,+) sector state:

    R_exact(N=4) = 0.023106 |c|^2     spread 0.00% over 10 draws at epsilon=0.01
                                      (0.08% at epsilon=0.2, an O(epsilon^2) term)

That one matrix element fixes R to four digits across independent draws, which
is exactly the statement that h acts through a single effective incoherent
pair-creation channel. Running the established finite-N law R = q/8 backwards
converts the exact correlator into that channel's rate:

    q_eff = 8 R_exact(N=4) = 0.18485 |c|^2

and each job gets TARGET_GAP_TIMES / q_eff evolved time, as whole copies of the
standard 2940-unit shape -- exactly as imps_drift_eps_grids.py does with q.
Since |c|^2 ~ epsilon^2 by construction, this reproduces the 1/epsilon^2
schedule scaling automatically rather than hard-coding it per epsilon.

q_eff is a calibrated proxy, not a theorem, so it is not trusted on its own:
the 3x-length control is what tests it, and the per-run drift-by-thirds table
is what certifies each individual job.

Controls
--------
  * **h = 0 from |0...0>**: the vacuum is dark for the unperturbed baseline, so
    R must come back at ~0. The pipeline's false-positive floor.
  * **3x-length re-run** of a factor-1 sample at epsilon=0.2. If R moves,
    TARGET_GAP_TIMES (hence q_eff) is too small and every factor-1 job in the
    grid is suspect. This is the control that tests the sizing argument above.
  * **|neel> cross-check** at epsilon=0.2 on three samples bracketing the
    ensemble in |c|. Read as consistency only -- Trap 5 is precisely that two
    starts agreeing is NOT evidence of relaxation.
  * **Positivity**: min eigenvalue / purity of the true 8-site window RDM
    (iobservables.reduced_density_matrix). The Hamiltonian part of the
    generator is norm-preserving rather than contractive, so truncation error
    is less damped here than in a purely dissipative flow; Trap 3 found R
    unaffected by non-positivity in the first study, but that is measured
    again rather than assumed.

Cache safety
------------
Namespaced 'haminf_...', and run_config carries the perturbation type, epsilon,
the init and the schedule (shape, factor) -- none of which the key expresses.
A Hamiltonian run can therefore never alias onto a dissipative one, nor a long
schedule onto a short one. All draws happen in the PARENT and the matrices are
passed by value: ex.EPSILON is module state and workers re-import the module
fresh under Windows spawn.

Outputs (experiments/results/):
    imps_ham_eps_grid.pkl        every trajectory + final profile + the sizing
    imps_ham_flatness.png        R(r)/R(1) per sample, one panel per epsilon
    imps_ham_R_vs_epsilon.png    R vs epsilon with a FREE-exponent fit, and
                                 R/|c|^2 vs epsilon extrapolated to epsilon -> 0
    imps_ham_trajectories.png    every sample x every epsilon, raw + rolling

Usage:
    uv run python experiments/imps_ham_eps_grid.py --preflight   # sizing only
    uv run python experiments/imps_ham_eps_grid.py               # the grid
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
from lindblad_mps import exact, iobservables, itebd, models, observables

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERTURBATION = "hamiltonian"
EPSILONS = [0.20, 0.15, 0.10, 0.05]
ALL_SAMPLES = list(range(10))
# Distinct from renyi2_swssb.BASE_SEED: the Hermitian ensemble is an
# independent draw, not a Hermitized version of the dissipative one.
HAM_BASE_SEED = 20260919

CHI = 128
R_MAX = 100
REFERENCE_R = [1, 2, 5, 10, 20, 50, 100]
STEPS_PER_DT = 50
CANONICALIZE_EVERY = 10

# The standard 2940-time-unit shape, shared with every other iTEBD grid here.
BASE_SHAPE = [0.1] * 500 + [0.05] * 120 + [0.02] * 100 + [0.01] * 60 + [0.005] * 40
BASE_UNITS = sum(BASE_SHAPE) * STEPS_PER_DT  # 2940.0

# Evolved time in units of 1/q_eff (see the module docstring). Same constant the
# drift grid used against its q, and tested here by the 3x-length control.
TARGET_GAP_TIMES = 12.0

EXACT_N = 4              # the size whose exact sector solve calibrates q_eff
FINITE_LAW_DENOM = 8.0   # R = q/8 at finite N, run backwards to get q_eff

LONG_CONTROL_MULTIPLE = 3
POSITIVITY_WINDOW = 8

MEASURE_TARGET_POINTS = 1000
ROLLING_WINDOW_FRAC = 0.02  # rolling-mean width as a fraction of evolved time

KIND_PREFIX = "haminf"

# Every trajectory summary (plateau, drift, the epsilon fit) reads R at R_MAX,
# but a trajectory only stores the separations in REFERENCE_R. Asserted rather
# than assumed: the mismatch stays silent until a KeyError deep inside a report,
# by which point the evolution has already been paid for.
assert R_MAX in REFERENCE_R, "R_MAX must be one of the separations in REFERENCE_R"
R_MID = REFERENCE_R[len(REFERENCE_R) // 2]  # middle column of the flatness table


# ---------------------------------------------------------------------------
# The perturbation ensemble and its calibrated rate
# ---------------------------------------------------------------------------
def draw_at(epsilon: float) -> list[dict]:
    """The ten Hermitian bond perturbations at one epsilon, drawn in the PARENT.

    Each sample has its own seeded Generator, so rescaling epsilon preserves the
    direction and changes only the norm -- which is what makes |c|^2 ~ epsilon^2
    exact, and "R ~ epsilon^2" a statement about R rather than about the draw
    (asserted in build_jobs()).

    Input: epsilon, the target operator norm of h.
    Output: list of dicts with 'seed', 'h', 'c', 'c2', 'description'.
    """
    out = []
    for s in range(len(ALL_SAMPLES)):
        seed = HAM_BASE_SEED + 1 + s
        rng = np.random.default_rng(seed)
        h = models.random_zz_commuting_hermitian_operator(epsilon, rng)
        assert models.commutes_with_zz(h), "sampled h must commute with Z(x)Z"
        assert np.allclose(h, h.conj().T), "sampled h must be Hermitian"
        c = models.pair_creation_element(h)
        out.append({
            "seed": seed, "h": h, "c": c, "c2": float(abs(c) ** 2),
            "description": models.describe_operator(h, epsilon, seed=seed),
        })
    return out


def exact_reference(h: np.ndarray, N: int = EXACT_N) -> dict:
    """Exact (+,+)-sector correlator and gap at size N, for one h.

    R here is the absolute reference the schedule is calibrated against (see the
    module docstring). 'gap_rate' is returned too, but only so the preflight can
    display the fact that it is epsilon-INDEPENDENT and therefore useless for
    sizing -- do not size a schedule from it.

    Input: h, the (4, 4) Hermitian bond perturbation; N, chain size (dense).
    Output: dict with 'R', 'gap_rate', 'gap_singular', 'purity'.
    """
    i, j = ex.correlator_sites(N)
    out = exact.sector_steady_state([(h, 1.0)], [], ex.build_L2_terms(None), [], N)
    rho = out["rho"]
    return {
        "R": float(observables.renyi2_correlator_dense(rho, models.X, i, j, N)),
        "gap_rate": out["gap_rate"],
        "gap_singular": out["gap_singular"],
        "purity": float(np.trace(rho @ rho).real),
    }


def effective_creation_rate(R_exact: float) -> float:
    """q_eff = 8 * R_exact(N=4): the incoherent pair-creation rate h acts through.

    The finite-N law R = q/8 (CLAUDE.md, established over ten dissipative draws
    and derived analytically) run backwards. Legitimate because the exact N=4
    correlator is proportional to |<11|h|00>|^2 to 0.00% across independent
    draws, which IS the statement that the coherent drive reduces to a single
    incoherent channel.

    Input: R_exact, the exact (+,+)-sector correlator at N = EXACT_N.
    Output: float, a relaxation rate -- schedules are TARGET_GAP_TIMES / this.
    """
    return FINITE_LAW_DENOM * R_exact


def schedule_factor(q_eff: float) -> int:
    """Number of BASE_SHAPE copies needed for TARGET_GAP_TIMES / q_eff units."""
    if q_eff <= 0.0:
        return 1
    return max(1, math.ceil(TARGET_GAP_TIMES / q_eff / BASE_UNITS))


def expand_schedule(factor: int) -> list[float]:
    """Repeat each BASE_SHAPE stage `factor` times: same dt anneal, factor x time.

    Lengthening by repeating stages -- rather than raising steps_per_dt or
    coarsening dt -- leaves the final-stage Trotter floor unchanged, so runs of
    different length stay directly comparable.
    """
    out: list[float] = []
    for dt in BASE_SHAPE:
        out.extend([dt] * factor)
    return out


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    """Cumulative evolved time at the end of each dt stage."""
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def measure_every(n_stages: int) -> int:
    """Stage-sampling stride giving ~MEASURE_TARGET_POINTS trajectory samples.

    Each measurement is a 100-point correlator profile plus an Arnoldi solve;
    the epsilon=0.05 schedules run to tens of thousands of stages, where
    measuring every stage would cost more than the evolution itself.
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

    The perturbation type is in here as well as epsilon and the schedule: the
    key cannot express any of them, and a dissipative entry silently reused for
    a Hamiltonian run would be wrong rather than merely stale. The schedule is
    stored as (shape, factor), which determines it exactly -- the expanded list
    runs to tens of thousands of entries at epsilon=0.05.
    """
    return {
        "perturbation": PERTURBATION,
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
    job['h'] only, never module state, so a worker's freshly imported copy of
    this module cannot contaminate the run. Exceptions are returned, not raised:
    one failed job must not abort the pool.

    Input: job dict with 'kind', 'label', 'sample', 'init', 'epsilon', 'c2',
        'q_eff', 'factor', 'h' (None for the h = 0 control).
    Output: the job dict (minus 'h') augmented with 'trajectory',
        'final_profile', 'positivity', 'bond_dims', 'seconds', 'cached'
        -- or 'error'.
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
    out = {k: v for k, v in job.items() if k != "h"}
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

        H2_terms = [] if job["h"] is None else [(job["h"], 1.0)]
        state, history = itebd.find_steady_state_infinite(
            H2_terms=H2_terms, H1_terms=[],
            L2_terms=ex.build_L2_terms(None), L1_terms=[],
            dt_schedule=schedule, steps_per_dt=STEPS_PER_DT,
            chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
            initial_state=ex_inf.build_initial_state(job["init"]),
            stage_callback=stage_callback,
        )

        xi_diag = iobservables.correlation_length(state)
        # Positivity of the TRUE window RDM, not of a tiled finite chain: the
        # Hamiltonian part of the generator does not damp truncation error the
        # way a dissipator does, so this is measured rather than assumed.
        try:
            rdm = iobservables.reduced_density_matrix(state, POSITIVITY_WINDOW)
            positivity = iobservables.rdm_diagnostics(rdm)
            positivity.pop("eigenvalues", None)  # keep the pickle small
        except Exception:
            positivity = None

        out.update({
            "trajectory": trajectory,
            "final_profile": iobservables.correlator_profile(state, models.X, r_max=R_MAX),
            "xi": xi_diag["xi"], "xi_reason": xi_diag["reason"],
            "eta1": xi_diag["eta1"], "eta2": xi_diag["eta2"],
            "positivity": positivity,
            "bond_dims": dict(state.bond_dims),
            "max_discarded_weight": max(history["discarded_weight"], default=0.0),
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


def eps_kind(epsilon: float) -> str:
    """Cache-safe kind label for one epsilon tier, e.g. 0.15 -> 'eps015'."""
    return f"eps{epsilon:.2f}".replace(".", "")


def build_sizing() -> tuple[dict, dict]:
    """Draw every sample at every epsilon and size its schedule from the exact solve.

    Output: (by_eps, sizing) -- the draws keyed by epsilon, and
        sizing[(epsilon, sample)] carrying 'c2', 'q_eff', 'factor', 'units',
        'target_units' and the exact_* diagnostics.
    """
    by_eps = {eps: draw_at(eps) for eps in EPSILONS}

    # The whole R ~ epsilon^2 reading assumes the rescaling is norm-only.
    ref = max(EPSILONS)
    for eps in EPSILONS:
        for s in ALL_SAMPLES:
            ratio = by_eps[eps][s]["c2"] / by_eps[ref][s]["c2"]
            assert abs(ratio - (eps / ref) ** 2) < 1e-9, (
                f"sample {s}: |c|^2 ratio {ratio} != {(eps / ref) ** 2} at "
                f"epsilon={eps}; the rescaling is not norm-only, so R ~ epsilon^2 "
                "does not follow"
            )

    sizing = {}
    for eps in EPSILONS:
        for s in ALL_SAMPLES:
            ref_exact = exact_reference(by_eps[eps][s]["h"])
            q_eff = effective_creation_rate(ref_exact["R"])
            factor = schedule_factor(q_eff)
            sizing[(eps, s)] = {
                "c2": by_eps[eps][s]["c2"], "q_eff": q_eff, "factor": factor,
                "units": factor * BASE_UNITS,
                "target_units": TARGET_GAP_TIMES / q_eff if q_eff else float("inf"),
                **{f"exact_{k}": v for k, v in ref_exact.items()},
            }
    return by_eps, sizing


def build_jobs() -> tuple[list[dict], dict]:
    """Enumerate the grid plus its controls, longest job first.

    Output: (jobs, sizing); see build_sizing() for the sizing table, which is
        pickled with the results -- the schedule argument is part of the result,
        not scaffolding around it.
    """
    by_eps, sizing = build_sizing()
    ref_eps = max(EPSILONS)

    jobs = []
    for eps in EPSILONS:
        for s in ALL_SAMPLES:
            z = sizing[(eps, s)]
            jobs.append({
                "kind": eps_kind(eps), "label": f"sample{s}", "sample": s,
                "init": "zero", "epsilon": eps, "h": by_eps[eps][s]["h"],
                "c2": z["c2"], "q_eff": z["q_eff"], "factor": z["factor"],
            })

    # h = 0 control: the vacuum is dark, so this is the false-positive floor.
    jobs.append({"kind": "control", "label": "h0", "sample": None, "init": "zero",
                 "epsilon": 0.0, "h": None, "c2": 0.0, "q_eff": 0.0, "factor": 1})

    # 3x-length control on the FASTEST factor-1 sample at the reference epsilon:
    # a factor-1 job has the least margin over its target, so it is the one that
    # actually tests TARGET_GAP_TIMES.
    factor1 = [s for s in ALL_SAMPLES if sizing[(ref_eps, s)]["factor"] == 1]
    if factor1:
        s = max(factor1, key=lambda k: sizing[(ref_eps, k)]["q_eff"])
        jobs.append({"kind": "longctrl", "label": f"sample{s}", "sample": s,
                     "init": "zero", "epsilon": ref_eps, "h": by_eps[ref_eps][s]["h"],
                     "c2": sizing[(ref_eps, s)]["c2"],
                     "q_eff": sizing[(ref_eps, s)]["q_eff"],
                     "factor": LONG_CONTROL_MULTIPLE})

    # |neel> cross-check on the samples bracketing the ensemble in |c|.
    by_c = sorted(ALL_SAMPLES, key=lambda k: sizing[(ref_eps, k)]["c2"])
    for s in sorted({by_c[0], by_c[len(by_c) // 2], by_c[-1]}):
        jobs.append({"kind": "neel", "label": f"sample{s}", "sample": s, "init": "neel",
                     "epsilon": ref_eps, "h": by_eps[ref_eps][s]["h"],
                     "c2": sizing[(ref_eps, s)]["c2"],
                     "q_eff": sizing[(ref_eps, s)]["q_eff"],
                     "factor": sizing[(ref_eps, s)]["factor"]})

    # Descending epsilon, longest-first inside each tier. Pure -factor ordering
    # would keep the pool's tail shortest, but it runs epsilon=0.05 FIRST (its
    # factors are the largest, up to 165) and so banks the cheap, highest-signal
    # tiers last. Ordering by epsilon instead means a run that dies partway still
    # leaves whole tiers cached and reportable. The h=0 control rides with the
    # reference tier: it is the false-positive floor, so it wants to be early.
    ref_for_sort = {"control": ref_eps}
    jobs.sort(key=lambda j: (-ref_for_sort.get(j["kind"], j["epsilon"]),
                             -j["factor"]))
    return jobs, sizing


# ---------------------------------------------------------------------------
# Trajectory summaries
# ---------------------------------------------------------------------------
def plateau(traj: list[dict], r: int, frac: float = 0.3) -> float:
    """Median of R(r) over the last `frac` of evolved time.

    Median not mean: individual stages carry occasional large spikes that a
    mean tracks and a median does not.
    """
    t = np.array([p["t"] for p in traj])
    v = np.array([p["R"][r] for p in traj])
    sel = v[t > (1 - frac) * t.max()]
    return float(np.nanmedian(sel)) if sel.size else float("nan")


def median_by_thirds(traj: list[dict], r: int) -> list[float]:
    """Median R(r) over each third of the trajectory -- the drift diagnostic."""
    v = np.array([p["R"][r] for p in traj])
    n = len(v)
    return [float(np.nanmedian(v[lo:hi]))
            for lo, hi in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))]


def drift(traj: list[dict], r: int | None = None) -> float:
    """Relative change in R(r) between the first and last third of the run.

    Input: traj; r, the separation (defaults to R_MAX, the largest measured).
    Output: float, (last third / first third) - 1.
    """
    th = median_by_thirds(traj, R_MAX if r is None else r)
    return (th[2] / th[0] - 1.0) if th[0] else float("nan")


def rolling(t: np.ndarray, v: np.ndarray, window: float) -> np.ndarray:
    """Centred rolling mean of v over a window of width `window` in t."""
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


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(sizing: dict) -> None:
    """Print the sizing table without running anything.

    Costs seconds (ten 64x64 dense sector solves per epsilon) and answers the
    two questions worth answering before hours of evolution: how long the grid
    is, and whether the rate it is sized from behaves as epsilon^2.
    """
    print(f"\n{'=' * 100}")
    print("PREFLIGHT -- schedule sizing from the exact N=%d sector solve" % EXACT_N)
    print(f"{'=' * 100}")
    print(f"  q_eff = {FINITE_LAW_DENOM:.0f} x R_exact(N={EXACT_N}), "
          f"target time = {TARGET_GAP_TIMES:.0f} / q_eff, "
          f"rounded up to whole {BASE_UNITS:.0f}-unit copies\n")

    header = (f"{'eps':>5} {'s':>2} {'|c|^2':>11} {'R_exact':>11} {'q_eff':>11} "
              f"{'target':>10} {'factor':>7} {'units':>10} {'gap_rate':>9}")
    total = 0.0
    for eps in EPSILONS:
        print(f"\n{header}")
        sub = 0.0
        for s in ALL_SAMPLES:
            z = sizing[(eps, s)]
            sub += z["units"]
            print(f"{eps:>5.2f} {s:>2} {z['c2']:>11.4e} {z['exact_R']:>11.4e} "
                  f"{z['q_eff']:>11.4e} {z['target_units']:>10.0f} {z['factor']:>7} "
                  f"{z['units']:>10.0f} {z['exact_gap_rate']:>9.4f}")
        print(f"{'':>5} {'':>2} {'':>11} {'':>11} {'':>11} {'':>10} {'':>7} "
              f"{sub:>10.0f}  <- epsilon={eps} subtotal")
        total += sub

    print(f"\n  GRID TOTAL: {total:.0f} evolved time units across "
          f"{len(EPSILONS) * len(ALL_SAMPLES)} production jobs.")

    # The two things the preflight exists to check.
    print(f"\n{'-' * 100}\nSanity checks\n{'-' * 100}")
    ref = max(EPSILONS)
    print(f"{'eps':>5} {'mean R_exact/|c|^2':>20} {'spread':>9} "
          f"{'q_eff/q_eff(ref)':>17} {'(eps/ref)^2':>12} {'gap_rate':>9}")
    for eps in EPSILONS:
        ratios = np.array([sizing[(eps, s)]["exact_R"] / sizing[(eps, s)]["c2"]
                           for s in ALL_SAMPLES])
        qr = np.mean([sizing[(eps, s)]["q_eff"] / sizing[(ref, s)]["q_eff"]
                      for s in ALL_SAMPLES])
        gr = np.mean([sizing[(eps, s)]["exact_gap_rate"] for s in ALL_SAMPLES])
        print(f"{eps:>5.2f} {ratios.mean():>20.6f} {ratios.std() / ratios.mean():>8.2%} "
              f"{qr:>17.5f} {(eps / ref) ** 2:>12.5f} {gr:>9.4f}")
    print("\n  q_eff must track (eps/ref)^2 -- that is what makes the schedule scale\n"
          "  as 1/epsilon^2. gap_rate must NOT: it is the baseline diffusive mode of a\n"
          "  4-site chain and is epsilon-independent, which is exactly why the schedule\n"
          "  is sized from the correlator instead (see the module docstring).",
          flush=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def report_controls(results: dict, sizing: dict) -> None:
    """The h=0 floor, the 3x-length control, |neel>, positivity, truncation.

    Printed first and deliberately so: none of the physics below is readable
    until these four say the runs mean something.
    """
    print(f"\n{'=' * 100}\nCONTROLS -- read these before any number below\n{'=' * 100}")

    ctrl = results.get(("control", None))
    if ctrl is not None:
        floor = abs(dict(ctrl["final_profile"]).get(R_MAX, float("nan")))
        print(f"\n  h = 0 from |0...0>:  |R({R_MAX})| = {floor:.3e}   "
              f"(exact answer is 0 -- this is the false-positive floor)")

    ref_eps = max(EPSILONS)
    for (kind, s), out in sorted(results.items(), key=lambda kv: str(kv[0])):
        if kind != "longctrl":
            continue
        short = results.get((eps_kind(ref_eps), s))
        if short is None:
            continue
        a, b = plateau(short["trajectory"], R_MAX), plateau(out["trajectory"], R_MAX)
        print(f"\n  {LONG_CONTROL_MULTIPLE}x-length control, sample {s} at "
              f"epsilon={ref_eps}:")
        print(f"    {short['evolved_time']:>8.0f} units -> R({R_MAX}) = {a:.6e}")
        print(f"    {out['evolved_time']:>8.0f} units -> R({R_MAX}) = {b:.6e}"
              f"    ({b / a - 1:+.2%})")
        print(f"    A few tenths of a percent means TARGET_GAP_TIMES = "
              f"{TARGET_GAP_TIMES:.0f} is enough and every factor-1 job is sound;\n"
              f"    a large move means q_eff underestimates the relaxation time and "
              f"the grid must be re-sized.")

    neel = [(s, out) for (kind, s), out in results.items() if kind == "neel"]
    if neel:
        print(f"\n  |neel> vs |0...0> at epsilon={ref_eps} "
              f"(consistency only -- see CLAUDE.md Trap 5):")
        print(f"    {'s':>3} {'|0...0>':>13} {'|neel>':>13} {'spread':>8}")
        for s, out in sorted(neel):
            z = results.get((eps_kind(ref_eps), s))
            if z is None:
                continue
            a, b = plateau(z["trajectory"], R_MAX), plateau(out["trajectory"], R_MAX)
            print(f"    {s:>3} {a:>13.4e} {b:>13.4e} "
                  f"{abs(a - b) / max(abs(a + b) / 2, 1e-300):>7.2%}")

    print("\n  Positivity of the true %d-site window RDM, and truncation:" % POSITIVITY_WINDOW)
    print(f"    {'eps':>5} {'worst min_eig':>14} {'max purity':>11} {'peak bond':>10}")
    for eps in EPSILONS:
        rows = [results[(eps_kind(eps), s)] for s in ALL_SAMPLES
                if (eps_kind(eps), s) in results]
        if not rows:
            continue
        pos = [r["positivity"] for r in rows if r.get("positivity")]
        worst = min((p["min_eig"] for p in pos), default=float("nan"))
        pur = max((p["purity"] for p in pos), default=float("nan"))
        bond = max(max(r["bond_dims"].values()) for r in rows)
        flag = "  <-- AT CAP" if bond >= CHI else ""
        print(f"    {eps:>5.2f} {worst:>14.3e} {pur:>11.4f} {bond:>7d}/{CHI}{flag}")
    print("    purity > 1 or a clearly negative min_eig means the converged state is "
          "not a\n    density matrix (CLAUDE.md Trap 3 -- R was unaffected there, "
          "but check, don't assume).", flush=True)


def report_flatness(results: dict) -> None:
    """R(r)/R(1) per sample: the long-range-order signature, no fitting."""
    print(f"\n{'=' * 100}\nFLATNESS -- R(r) not decaying with r IS the SWSSB signal"
          f"\n{'=' * 100}")
    print(f"\n{'eps':>5} {'s':>2} {f'R({REFERENCE_R[0]})':>12} {f'R({R_MID})':>12} "
          f"{f'R({R_MAX})':>12} {'ratio':>12} {'drift(3rds)':>12} {'bond':>9}")
    for eps in EPSILONS:
        for s in ALL_SAMPLES:
            out = results.get((eps_kind(eps), s))
            if out is None:
                continue
            prof = dict(out["final_profile"])
            r1, r20, r100 = prof.get(REFERENCE_R[0]), prof.get(R_MID), prof.get(R_MAX)
            bond = max(out["bond_dims"].values())
            print(f"{eps:>5.2f} {s:>2} {r1:>12.4e} {r20:>12.4e} {r100:>12.4e} "
                  f"{(r100 / r1 if r1 else float('nan')):>12.5f} "
                  f"{drift(out['trajectory']):>+11.1%} {bond:>6d}/{CHI}")
    print("\n  A ratio near 1 over r = 1..100 is long-range order with no finite-size\n"
          "  caveat. A run still drifting by tens of percent has not relaxed, and its\n"
          "  flatness says nothing yet.", flush=True)


def report_epsilon_scaling(results: dict, sizing: dict) -> dict:
    """THE DELIVERABLE: the exponent in R ~ epsilon^p, fitted free.

    Two readings, both reported because they fail differently. The pooled fit
    over every (epsilon, sample) point measures the exponent directly. The
    R/|c|^2 table measures the same thing normalized by the pair-creation
    amplitude, where an exact epsilon^2 law shows up as a CONSTANT column --
    and any residual epsilon dependence there is the leading correction, which
    the first study found to be O(epsilon^2) with coefficient -0.036.

    Output: dict with the fitted exponent, its standard error, and the
        per-epsilon R/|c|^2 means.
    """
    print(f"\n{'=' * 100}\nEPSILON SCALING -- R ~ epsilon^p, exponent fitted FREE"
          f"\n{'=' * 100}")

    print(f"\n{'eps':>5} {'n':>3} {'mean R/|c|^2':>14} {'spread':>8} "
          f"{'mean R':>13} {'vs eps^2 from ref':>18}")
    ref = max(EPSILONS)
    per_eps, logs_e, logs_R = {}, [], []
    ref_mean = None
    for eps in EPSILONS:
        vals, norm = [], []
        for s in ALL_SAMPLES:
            out = results.get((eps_kind(eps), s))
            if out is None:
                continue
            R = plateau(out["trajectory"], R_MAX)
            if not np.isfinite(R) or R <= 0:
                continue
            vals.append(R)
            norm.append(R / sizing[(eps, s)]["c2"])
            logs_e.append(np.log(eps))
            logs_R.append(np.log(R))
        if not vals:
            continue
        a, n_ = np.array(norm), np.array(vals)
        per_eps[eps] = {"n": len(vals), "R_over_c2": float(a.mean()),
                        "spread": float(a.std() / a.mean()), "R": float(n_.mean())}
        if eps == ref:
            ref_mean = n_.mean()
        pred = (eps / ref) ** 2 * ref_mean if ref_mean else float("nan")
        print(f"{eps:>5.2f} {len(vals):>3} {a.mean():>14.6f} "
              f"{a.std() / a.mean():>7.2%} {n_.mean():>13.4e} "
              f"{(n_.mean() / pred if pred else float('nan')):>17.5f}x")

    fit = {"per_eps": per_eps}
    if len(logs_e) >= 3:
        x, y = np.array(logs_e), np.array(logs_R)
        p, b = np.polyfit(x, y, 1)
        resid = y - (p * x + b)
        dof = max(len(x) - 2, 1)
        se = float(np.sqrt((resid @ resid / dof) / np.sum((x - x.mean()) ** 2)))
        fit.update({"exponent": float(p), "exponent_stderr": se})
        print(f"\n  FREE-EXPONENT FIT over all {len(x)} points:  "
              f"R ~ epsilon^({p:.5f} +- {se:.5f})")
        print(f"  Against the adiabatic-elimination prediction p = 2: "
              f"{(p - 2) / se:+.1f} standard errors.")

    if len(per_eps) >= 3:
        # R/|c|^2 = A (1 + k eps^2): the same correction the first study found.
        e = np.array(sorted(per_eps))
        v = np.array([per_eps[x]["R_over_c2"] for x in e])
        k, a0 = np.polyfit(e**2, v, 1)
        fit.update({"amplitude_intercept": float(a0),
                    "amplitude_eps2_coeff": float(k / a0) if a0 else float("nan")})
        print(f"\n  R/|c|^2 = {a0:.6f} (1 {k / a0:+.4f} eps^2)  "
              f"-> {a0:.6f} as epsilon -> 0")
        print(f"  Exact N=4 law for comparison: R_exact/|c|^2 -> "
              f"{np.mean([sizing[(min(EPSILONS), s)]['exact_R'] / sizing[(min(EPSILONS), s)]['c2'] for s in ALL_SAMPLES]):.6f}"
              f"  (ratio infinite/finite = "
              f"{a0 / np.mean([sizing[(min(EPSILONS), s)]['exact_R'] / sizing[(min(EPSILONS), s)]['c2'] for s in ALL_SAMPLES]):.4f})")
        print("  The first study's dissipative law has infinite/finite = 2 exactly "
              "(q/4 vs q/8).", flush=True)
    return fit


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_flatness(results: dict) -> str:
    """R(r)/R(1) per sample, one panel per epsilon. Flat lines = SWSSB."""
    plt = ex._mpl()
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(4.4 * len(EPSILONS), 4.2),
                             squeeze=False, sharey=True)
    cmap = plt.get_cmap("viridis")
    for col, eps in enumerate(EPSILONS):
        ax = axes[0][col]
        for s in ALL_SAMPLES:
            out = results.get((eps_kind(eps), s))
            if out is None:
                continue
            prof = out["final_profile"]
            r = np.array([p[0] for p in prof], dtype=float)
            v = np.array([p[1] for p in prof], dtype=float)
            if v[0]:
                ax.plot(r, v / v[0], "-", lw=1.4, color=cmap(s / 9), label=f"s{s}")
        ax.axhline(1.0, color="black", ls="--", lw=1.2)
        ax.set_xlabel("separation r")
        ax.set_title(rf"$\epsilon = {eps}$")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r"$R(r)\ /\ R(1)$")
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Hamiltonian perturbation: long-range order in the thermodynamic limit")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_ham_flatness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_epsilon_scaling(results: dict, sizing: dict, fit: dict) -> str:
    """R vs epsilon with the free-exponent fit, and R/|c|^2 vs epsilon."""
    plt = ex._mpl()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")

    for s in ALL_SAMPLES:
        xs, ys = [], []
        for eps in EPSILONS:
            out = results.get((eps_kind(eps), s))
            if out is None:
                continue
            R = plateau(out["trajectory"], R_MAX)
            if np.isfinite(R) and R > 0:
                xs.append(eps)
                ys.append(R)
        if xs:
            ax0.plot(xs, ys, "o-", ms=5, lw=1.0, color=cmap(s / 9), alpha=0.85,
                     label=f"sample {s}")
    if "exponent" in fit:
        e = np.array(sorted(EPSILONS), dtype=float)
        anchor = fit["per_eps"].get(max(EPSILONS))
        if anchor:
            ref = max(EPSILONS)
            ax0.plot(e, anchor["R"] * (e / ref) ** fit["exponent"], "k--", lw=2.0,
                     label=rf"fit $\epsilon^{{{fit['exponent']:.3f}}}$")
            ax0.plot(e, anchor["R"] * (e / ref) ** 2, ":", color="crimson", lw=2.0,
                     label=r"$\epsilon^2$")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"perturbation strength $\epsilon = \|h\|$")
    ax0.set_ylabel(rf"$R(r={R_MAX})$")
    ax0.set_title("Correlator vs perturbation strength")
    ax0.grid(True, alpha=0.3, which="both")
    ax0.legend(fontsize=7, ncol=2)

    for eps in EPSILONS:
        xs, ys = [], []
        for s in ALL_SAMPLES:
            out = results.get((eps_kind(eps), s))
            if out is None:
                continue
            R = plateau(out["trajectory"], R_MAX)
            if np.isfinite(R) and R > 0:
                xs.append(eps)
                ys.append(R / sizing[(eps, s)]["c2"])
        ax1.plot(xs, ys, "o", ms=6, alpha=0.75, color="tab:blue")
    if "amplitude_intercept" in fit:
        a0, k = fit["amplitude_intercept"], fit["amplitude_eps2_coeff"]
        e = np.linspace(0, max(EPSILONS) * 1.05, 100)
        ax1.plot(e, a0 * (1 + k * e**2), "-", color="crimson", lw=2.0,
                 label=rf"$A(1 {k:+.3f}\,\epsilon^2)$, $A={a0:.5f}$")
        ax1.axhline(a0, ls="--", color="black", lw=1.2,
                    label=rf"$\epsilon\to 0$: {a0:.5f}")
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel(r"$R\ /\ |\langle 11|h|00\rangle|^2$")
    ax1.set_title(r"Amplitude law (flat $\Rightarrow$ exactly $\epsilon^2$)")
    ax1.grid(True, alpha=0.3)
    if ax1.get_legend_handles_labels()[1]:  # the fit needs >= 3 epsilons
        ax1.legend(fontsize=8)

    fig.suptitle("Hamiltonian perturbation of the dark steady state, iTEBD")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_ham_R_vs_epsilon.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_trajectories(results: dict) -> str:
    """Every sample x every epsilon, raw and rolling -- for eyeballing relaxation."""
    plt = ex._mpl()
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(4.4 * len(EPSILONS), 4.2),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for col, eps in enumerate(EPSILONS):
        ax = axes[0][col]
        for s in ALL_SAMPLES:
            out = results.get((eps_kind(eps), s))
            if out is None:
                continue
            traj = out["trajectory"]
            t = np.array([p["t"] for p in traj])
            v = np.array([p["R"][R_MAX] for p in traj])
            ax.plot(t, v, "-", lw=0.4, alpha=0.20, color=cmap(s / 9))
            ax.plot(t, rolling(t, v, max(t.max() * ROLLING_WINDOW_FRAC, 1.0)),
                    "-", lw=1.4, color=cmap(s / 9), label=f"s{s}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("evolved time")
        ax.set_title(rf"$\epsilon = {eps}$")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.set_ylabel(rf"$R(r={R_MAX})$")
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Relaxation of the Renyi-2 correlator (flat tail = converged)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_ham_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    jobs, sizing = build_jobs()

    if "--preflight" in sys.argv:
        preflight(sizing)
        return

    # --eps 0.2,0.15 runs only those tiers (controls always ride along, and the
    # per-job cache makes a later superset run reuse everything already done).
    # Sizing is always computed over the FULL grid so the pickled sizing table
    # and the preflight table stay identical across staged runs.
    only = [a for a in sys.argv if a.startswith("--eps")]
    if only:
        arg = only[0].split("=", 1)[1] if "=" in only[0] else sys.argv[sys.argv.index(only[0]) + 1]
        keep = {eps_kind(float(x)) for x in arg.split(",")}
        jobs = [j for j in jobs if j["kind"] in keep or j["sample"] is None
                or j["kind"] in ("longctrl", "neel")]
        print(f"--eps {arg}: {len(jobs)} of 45 jobs selected "
              f"(tiers {sorted(keep)} + controls)", flush=True)

    preflight(sizing)
    total = sum(j["factor"] * BASE_UNITS for j in jobs)
    print(f"\n{len(jobs)} iTEBD jobs on {ex.N_WORKERS} workers, chi={CHI}, "
          f"{total:.0f} evolved time units in total.\n", flush=True)

    results: dict = {}
    failures = []
    t0 = time.perf_counter()
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            key = (out["kind"], out["sample"])
            tag = f"  {out['kind']:>9} {out['label']:>9} {out['init']:>4}"
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue
            results[key] = out
            timing = "cached" if out["cached"] else f"{out['seconds'] / 60:.1f}min"
            last = out["trajectory"][-1]
            print(f"{tag}  t={out['evolved_time']:>8.0f}  "
                  f"R({R_MAX})={last['R'][R_MAX]:.4e}  "
                  f"drift={drift(out['trajectory']):>+7.1%}  "
                  f"bond={last['bond']:3d}/{CHI}  "
                  f"[{timing}, elapsed {(time.perf_counter() - t0) / 60:.1f}min]",
                  flush=True)

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "imps_ham_eps_grid.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "config": {"perturbation": PERTURBATION, "epsilons": EPSILONS,
                       "samples": ALL_SAMPLES, "chi": CHI, "seed": HAM_BASE_SEED,
                       "target_gap_times": TARGET_GAP_TIMES, "r_max": R_MAX,
                       "base_shape": BASE_SHAPE, "steps_per_dt": STEPS_PER_DT},
            "sizing": sizing, "results": results,
            "failures": [{k: v for k, v in f_.items() if k != "h"} for f_ in failures],
        }, f)
    print(f"\n{len(results)}/{len(jobs)} jobs succeeded in "
          f"{(time.perf_counter() - t0) / 60:.1f} min.  Saved -> {path}", flush=True)

    fit = {}
    for name, fn in (("controls", lambda: report_controls(results, sizing)),
                     ("flatness", lambda: report_flatness(results))):
        try:
            fn()
        except Exception:
            print(f"!! {name} report failed (data is saved):\n"
                  f"{traceback.format_exc()}", flush=True)
    try:
        fit = report_epsilon_scaling(results, sizing)
    except Exception:
        print(f"!! scaling report failed (data is saved):\n"
              f"{traceback.format_exc()}", flush=True)

    for name, fn in (("flatness", lambda: plot_flatness(results)),
                     ("scaling", lambda: plot_epsilon_scaling(results, sizing, fit)),
                     ("trajectories", lambda: plot_trajectories(results))):
        try:
            print(f"Saved {name:<13s} -> {fn()}", flush=True)
        except Exception:
            print(f"!! {name} plot failed (data is saved):\n"
                  f"{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
