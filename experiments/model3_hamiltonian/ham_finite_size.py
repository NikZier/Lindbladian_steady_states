"""HAMILTONIAN perturbations of the dark steady state at FINITE size: exact
sector solves at N in {4, 6} and TEBD at N in {4, 6, 8, 12}, for epsilon in
{0.20, 0.15, 0.10, 0.05} with ten random draws each.

Why finite size, after imps_ham_eps_grid.py
-------------------------------------------
The iTEBD version of this study is not affordable. Measured 2026-09-23 on the
12-core desktop: the Hamiltonian flow saturates chi=128 (the dissipative
driftinf runs settle at bond 32-50), which puts every two-site SVD at 512x512
and drops throughput to 1.3 steps/s. Combined with a schedule that must scale
as 1/epsilon^2 -- the infinite chain relaxes at the perturbation-induced rate
~epsilon^2 -- the grid costs 6347 CPU-hours, 62 days of wall clock on the
epsilon=0.05 tier alone. See imps_ham_eps_grid.py for the sizing argument that
makes that unavoidable *in the thermodynamic limit*.

Finite N escapes it for one reason, already recorded in exact.sector_steady_state
and re-measured here: the finite-chain relaxation rate is the BASELINE diffusive
mode, and it is epsilon-INDEPENDENT (gap_rate = 3.0633/3.0576/3.0562 at N=4 for
epsilon = 0.20/0.10/0.05; 1.2806/1.2762/1.2750 at N=6). What made that number
useless for sizing an infinite-system schedule is exactly what makes a finite
grid cheap: ONE fixed schedule serves every epsilon. The standard finite
schedule is 1500 steps against the iTEBD shape's 41000 per factor-unit.

The exponent is settled by exact diagonalisation, not by TEBD
------------------------------------------------------------
R/|c|^2 from the dense (+,+) sector solve, measured 2026-09-23:

    N=4:  0.023084  0.023101  0.023105     epsilon = 0.20, 0.10, 0.05
    N=6:  0.021576  0.021596  0.021601

Flat to ~0.1% across a 4x span in epsilon at both sizes. Since |c|^2 ~ epsilon^2
by construction (draw_at rescales a fixed direction), that IS R ~ epsilon^2, with
no schedule, no truncation and no fit. The adiabatic-elimination prediction in
imps_ham_eps_grid.py's docstring is therefore confirmed exactly, and the free
exponent fit that grid was built to produce is not needed to establish it.

What TEBD is for here is the OTHER question: R/|c|^2 falls 0.023084 -> 0.021576
between N=4 and N=6, and whether that flow converges to a nonzero constant
(long-range order survives) or decays is what the size scan measures.

Certification, and the one caveat
---------------------------------
  * **TEBD vs exact at N=4 and N=6**, every sample, not a spot check. Measured
    0.01% (N=4) and 0.02% (N=6) at chi=128 on sample 0. This certifies the
    schedule AND the truncation against ground truth -- something the infinite
    pipeline structurally cannot do, and the reason Trap 5 does not apply here.
  * **chi=32 is NOT enough.** At N=6 it pins the bond at the cap and lands
    0.23% off exact, against 0.02% at chi=128. The dissipative study's CHI_MAX
    of 32 does not transfer: the Hamiltonian part is norm-preserving rather
    than contractive, so the state stays more entangled at every size.
  * **N=12 IS TRUNCATION-CORRUPTED and must not be quoted** (measured
    2026-09-24, the full grid). Max discarded weight is 6.8e-05 at eps=0.05 and
    1.5e-04 at eps=0.20, against 1.3e-08 at N=8 -- five orders of magnitude.
    The consequence is visible directly in the observable: R/|c|^2 has an
    ensemble spread of 38% at N=12 (0.0221 to 0.0307 at eps=0.05) against
    0.0014% at N=8, and the stage correlators never settle (sample 6 rises 37%
    on its FINAL stage). --convergence sweeps chi at N=12 to find what it needs.

    ⚠️ The trap that hid this, worth not repeating: 'bond_at_cap' reads the
    FINAL bond dimension, which at eps=0.05 N=12 is only 40-64 of 128 -- so the
    run looks comfortably unconstrained. The damage happens in the entangled
    TRANSIENT, and max_discarded_weight over the whole run is the only field
    that reports it. This is CLAUDE.md's "a small bond dimension can mean the
    opposite of well converged", in a new disguise.
  * **h = 0 control** at every size: the vacuum is dark, so R must return ~0.
  * **|neel> cross-check** at the largest size. Read as consistency only --
    Trap 5 is precisely that two starts agreeing is not evidence of relaxation.

Schedule margin, honestly stated
--------------------------------
gap(6)/gap(4) = 0.418 against diffusive (4/6)^2 = 0.444, so the gap closes as
~1/N^2 and gap(20) ~ 0.12. The standard 55.5-unit schedule is then only ~6.8
relaxation times at N=20 -- thin, and inherited from the dissipative study
rather than introduced here. At N <= 12 it is >= 19 relaxation times and the
exact comparison confirms it directly, so the main grid is safe; the N=20
convergence run reports its own stage drift and residual so the margin is
visible rather than assumed.

Cache safety
------------
Namespaced 'hamfin_...', and run_config carries the perturbation type, epsilon,
sample, N, init and chi -- so a Hamiltonian run can never alias onto a
dissipative one from renyi2_swssb, nor one chi onto another. All draws happen
in the PARENT and the matrices are passed by value (Windows spawn re-imports).

Outputs (experiments/model3_hamiltonian/results/):
    ham_finite_size.pkl             every run, plus the exact reference table
    ham_finite_size_R_vs_N.png      R/|c|^2 vs N, one line per epsilon
    ham_finite_size_profiles.png    R(i, r) profiles at the largest size
    ham_finite_size_convergence.pkl  the --convergence chi sweep

Usage:
    uv run python experiments/model3_hamiltonian/ham_finite_size.py                # the grid
    uv run python experiments/model3_hamiltonian/ham_finite_size.py --convergence  # chi at N=12
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  (puts the repo root and every study folder on sys.path)

import numpy as np

import imps_ham_eps_grid as ham
import renyi2_swssb as ex
from lindblad_mps import exact, models, observables, residual, tebd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERTURBATION = "hamiltonian"
EPSILONS = ham.EPSILONS                 # [0.20, 0.15, 0.10, 0.05]
ALL_SAMPLES = ham.ALL_SAMPLES           # range(10)
BASE_SEED = ham.HAM_BASE_SEED           # same ensemble as the iTEBD study

# N <= 6 is dense; both are also run through TEBD so every sample is certified
# against ground truth rather than one spot check.
EXACT_SIZES = [4, 6]
TEBD_SIZES = [4, 6, 8, 12]

CHI = 128                # chi=32 lands 0.23% off exact at N=6; see docstring
CUTOFF = ex.CUTOFF
DT_SCHEDULE = ex.DT_SCHEDULE
STEPS_PER_DT = ex.STEPS_PER_DT
RECANON_EVERY = ex.RECANON_EVERY

# The truncation question, deliberately a separate run (--convergence).
#
# Retargeted 2026-09-24 from N=20 to N=12. The main grid showed N=12 is already
# truncation-corrupted: max discarded weight 6.8e-05 (eps=0.05) to 1.5e-04
# (eps=0.20) against 1.3e-08 at N=8, five orders of magnitude, and the stage
# correlators are still moving violently at the end of the schedule (sample 6,
# eps=0.05: 0.02194 0.01272 0.02167 0.02233 0.03065 in R/|c|^2). The ensemble
# spread of R/|c|^2 is 38% at N=12 against 0.0014% at N=8. So N=12, not N=20,
# is where the method first breaks, and it is the size whose repair would
# extend the clean N-series from three points to four.
#
# NOTE the trap this exposed: 'bond_at_cap' reads the FINAL state, which is
# only 40-64 here, and that made N=12 look safe. The damage is done in the
# transient, which max_discarded_weight is what actually reports.
# Retargeted again 2026-09-25 to N in {14, 16}: chi=256 repaired N=12 (full row
# below), so the live question is how far up in N it keeps working. chi=320 is
# included deliberately -- "is chi=256 sufficient" cannot be answered by a sweep
# whose largest leg IS 256, which is the mistake the dissipative study's chi
# sweep made (CLAUDE.md Trap 1: "exactly one point that is not cap-bound, so it
# brackets convergence from below without ever demonstrating a plateau").
#
# ⚠️ N=14 is 'zero'-start only -- see the neel parity note below.
CONV_NS = [14, 16]
CONV_CHIS = [192, 256, 320]
# Two probes: the worst OBSERVABLE deviation, and the worst TRUNCATION
# pressure. They are different samples, and either alone would mislead.
CONV_PROBES = [(0.05, 6),    # R/|c|^2 = 0.030653, +44% over the N=8 reference
               (0.20, 9)]    # residual 4.10e-04, bond pinned at the cap

# --highchi: the full ensemble at one (N, chi), to replace a row the main grid
# got wrong. The N=12 sweep (2026-09-24) showed chi=128 overstates R/|c|^2 by
# 46% at eps=0.05 and that chi=256 restores the eps-independence the exact
# solves prove must hold (probes agree to 0.9% at chi=256, 42% apart at 128).
#
# Advanced to N=16 on 2026-09-26, after the N=12 row was completed at chi=256
# (40/40, eps-spread 0.41%) and the N in {14,16} sweep certified chi=256 there:
# |chi=256 - chi=320| is 0.74% (eps=0.20) and 0.12% (eps=0.05) at N=16, both
# well inside the 1.47% ensemble spread the N=12 row already carries. The one
# leg that failed to certify, N=14 eps=0.05, failed because its chi=320
# REFERENCE never relaxed (drift 5.93%, stage tail still climbing), not because
# chi=256 was short -- see report_convergence's table.
HIGHCHI_N = 16
HIGHCHI_CHI = 256

# ⚠️ |neel> is NOT in the (+,+) sector for N = 14: the state |0101...> has
# N/2 ones, odd when N/2 is odd, so it sits in the opposite parity sector and
# is not comparable. Only 'zero' is used for the convergence probes, which is
# why N=14 is safe there -- do not add a neel job at an N in {6, 10, 14, 18}.

NEEL_SIZE = max(TEBD_SIZES)
NEEL_EPSILON = max(EPSILONS)

KIND_PREFIX = "hamfin"
N_WORKERS = ex.N_WORKERS


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------
def run_config(job: dict) -> dict:
    """Everything that changes the numbers, for cache invalidation.

    job_key() alone is not enough: it names the run, not the settings behind
    it. A schedule or chi change must invalidate, silently reusing a stale
    pickle is exactly how a grid ends up reporting a schedule it never ran.
    """
    return {
        "perturbation": PERTURBATION,
        "epsilon": job["epsilon"],
        "sample": job["sample"],
        "N": job["N"],
        "init": job["init"],
        "chi_max": job["chi_max"],
        "dt_schedule": list(DT_SCHEDULE),
        "steps_per_dt": STEPS_PER_DT,
        "cutoff": CUTOFF,
        "recanon_every": RECANON_EVERY,
        "seed": BASE_SEED,
    }


def job_key(job: dict) -> str:
    """Cache filename stem, namespaced away from every other study here."""
    eps = f"{job['epsilon']:.2f}".replace(".", "")
    s = "ctrl" if job["sample"] is None else f"s{job['sample']}"
    return (f"{KIND_PREFIX}_{job['kind']}_eps{eps}_{s}_"
            f"N{job['N']}_{job['init']}_chi{job['chi_max']}")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def steady_state_correlator(h, N: int, init_name: str, chi_max: int) -> dict:
    """TEBD steady state at size N under H = sum_a h_{a,a+1}, then R(i, j).

    Mirrors renyi2_swssb.run_steady_state_correlator, which takes no H2 terms
    (that study is purely dissipative). Input h may be None for the h = 0
    control, which must return R ~ 0 from the dark vacuum.

    Input: h, the (4,4) Hermitian bond operator or None; N; init_name
        ('zero'/'neel'); chi_max.
    Output: dict with 'correlator', 'profile', 'stage_correlators',
        'stage_drift', 'final_bond_dims', 'max_discarded_weight',
        'min_profile', 'residual'.
    """
    i, j = ex.correlator_sites(N)
    H2_terms = [] if h is None else [(h, 1.0)]
    L2_terms = ex.build_L2_terms(None)
    stage_correlators = []

    def record_stage(stage: int, dt: float, st) -> None:
        stage_correlators.append(
            (dt, observables.renyi2_correlator_mps(st, models.X, i, j)))

    state, history = tebd.find_steady_state(
        H2_terms=H2_terms, H1_terms=[], L2_terms=L2_terms, L1_terms=[],
        N=N, dt_schedule=DT_SCHEDULE, steps_per_dt=STEPS_PER_DT,
        chi_max=chi_max, cutoff=CUTOFF, recanonicalize_every=RECANON_EVERY,
        initial_state=ex.build_initial_state(init_name, N),
        stage_callback=record_stage)

    R = observables.renyi2_correlator_mps(state, models.X, i, j)
    profile = [(r, observables.renyi2_correlator_mps(state, models.X, i, r))
               for r in range(i + 1, N)]

    # Scaled by the larger magnitude so a correlator collapsing to zero (the
    # h=0 control) reads as small drift rather than dividing by ~0.
    if len(stage_correlators) >= 2:
        last, prev = stage_correlators[-1][1], stage_correlators[-2][1]
        scale = max(abs(last), abs(prev))
        stage_drift = abs(last - prev) / scale if scale > 0 else 0.0
    else:
        stage_drift = float("nan")

    # Absolute convergence test: zero at a steady state of any sector, and
    # unlike stage_drift it measures the STATE rather than the trajectory.
    # Never allowed to fail a run (same policy as renyi2_swssb).
    try:
        res = residual.steady_state_residual(state, H2_terms, [], L2_terms, [])
    except Exception:
        res = {}

    return {
        "correlator": R,
        "profile": profile,
        "stage_correlators": stage_correlators,
        "stage_drift": stage_drift,
        "final_bond_dims": list(state.bond_dims),
        "max_bond": max(state.bond_dims),
        "bond_at_cap": max(state.bond_dims) >= chi_max,
        "max_discarded_weight": max(history["discarded_weight"], default=0.0),
        "min_profile": min([R] + [v for _, v in profile]),
        "residual": res,
    }


def run_job(job: dict) -> dict:
    """Worker entry point: one TEBD run, cached, exceptions returned not raised.

    Module-level so it survives pickling under Windows spawn. Reads job['h']
    only, never module state, so a worker's fresh import cannot contaminate it.
    One failed job must not abort the pool.
    """
    path = os.path.join(ex.CACHE_DIR, job_key(job) + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                out = pickle.load(f)
            if out.get("run_config") == run_config(job):
                out["cached"] = True
                return out
        except Exception:  # corrupt/partial cache -- recompute
            pass

    t0 = time.perf_counter()
    out = {k: v for k, v in job.items() if k != "h"}
    out["cached"] = False
    out["run_config"] = run_config(job)
    try:
        out["result"] = steady_state_correlator(
            job["h"], job["N"], job["init"], job["chi_max"])
    except Exception:
        out["error"] = traceback.format_exc()
    out["seconds"] = time.perf_counter() - t0

    if "error" not in out:
        os.makedirs(ex.CACHE_DIR, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f)
        os.replace(tmp, path)  # atomic: no reader sees a half-written file
    return out


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
def exact_table(by_eps: dict) -> dict:
    """Dense (+,+)-sector steady state at every EXACT_SIZES, epsilon and sample.

    This is the study's ground truth: no schedule, no truncation. It also
    re-measures gap_rate, whose epsilon-independence is the premise that makes
    one fixed schedule legitimate across the whole grid.

    Output: {(N, epsilon, sample): {'R', 'R_over_c2', 'gap_rate', 'purity'}}
    """
    table = {}
    for N in EXACT_SIZES:
        i, j = ex.correlator_sites(N)
        for eps in EPSILONS:
            for s in ALL_SAMPLES:
                d = by_eps[eps][s]
                out = exact.sector_steady_state(
                    [(d["h"], 1.0)], [], ex.build_L2_terms(None), [], N)
                rho = out["rho"]
                R = float(observables.renyi2_correlator_dense(
                    rho, models.X, i, j, N))
                table[(N, eps, s)] = {
                    "R": R,
                    "R_over_c2": R / d["c2"],
                    "gap_rate": out["gap_rate"],
                    "purity": float(np.trace(rho @ rho).real),
                }
    return table


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------
def estimated_cost(job: dict) -> float:
    """Relative cost, to queue long jobs first.

    One SVD per bond per step, O(chi^3) at fixed physical dimension, N-1 bonds
    -- the same model renyi2_swssb uses.
    """
    return (job["N"] - 1) * job["chi_max"] ** 3


def build_jobs(by_eps: dict) -> list[dict]:
    """The grid plus its controls, longest job first."""
    jobs = []
    for N in TEBD_SIZES:
        for eps in EPSILONS:
            for s in ALL_SAMPLES:
                jobs.append({
                    "kind": "grid", "epsilon": eps, "sample": s, "N": N,
                    "init": "zero", "chi_max": CHI,
                    "h": by_eps[eps][s]["h"], "c2": by_eps[eps][s]["c2"],
                })

    # h = 0 at every size: the vacuum is dark, so this is the false-positive
    # floor of the whole pipeline.
    for N in TEBD_SIZES:
        jobs.append({"kind": "control", "epsilon": 0.0, "sample": None, "N": N,
                     "init": "zero", "chi_max": CHI, "h": None, "c2": 0.0})

    # |neel> cross-check on the samples bracketing the ensemble in |c|^2.
    by_c = sorted(ALL_SAMPLES, key=lambda k: by_eps[NEEL_EPSILON][k]["c2"])
    for s in sorted({by_c[0], by_c[len(by_c) // 2], by_c[-1]}):
        jobs.append({"kind": "neel", "epsilon": NEEL_EPSILON, "sample": s,
                     "N": NEEL_SIZE, "init": "neel", "chi_max": CHI,
                     "h": by_eps[NEEL_EPSILON][s]["h"],
                     "c2": by_eps[NEEL_EPSILON][s]["c2"]})

    jobs.sort(key=lambda j: -estimated_cost(j))
    return jobs


def build_convergence_jobs(by_eps: dict) -> list[dict]:
    """The chi sweep at CONV_N -- the question the main grid deliberately defers.

    Note the chi=CHI leg duplicates a grid run but under kind='conv', so it
    gets its own cache entry rather than aliasing onto the grid's. That costs
    one re-run and buys a sweep whose baseline was measured under the same
    conditions as its other legs.

    Input: by_eps, the draws keyed by epsilon.
    Output: list of job dicts, most expensive first.
    """
    jobs = []
    for N in CONV_NS:
        for eps, s in CONV_PROBES:
            d = by_eps[eps][s]
            for chi in CONV_CHIS:
                jobs.append({"kind": "conv", "epsilon": eps, "sample": s,
                             "N": N, "init": "zero", "chi_max": chi,
                             "h": d["h"], "c2": d["c2"]})
    jobs.sort(key=lambda j: -estimated_cost(j))
    return jobs


def d_exact(by_eps: dict) -> dict:
    """Exact table, loaded from the main pickle if present, else recomputed.

    The 80 dense solves cost ~253 s; a --highchi run only needs them for the
    report's N<=6 rows, so reuse them when ham_finite_size.pkl already has them.
    """
    path = os.path.join(layout.MODEL3_RESULTS, "ham_finite_size.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)["exact"]
        except Exception:
            pass
    return exact_table(by_eps)


def build_highchi_jobs(by_eps: dict) -> list[dict]:
    """The full ensemble at (HIGHCHI_N, HIGHCHI_CHI), to replace a bad grid row.

    Kind stays 'grid' so these land in the same result keys the reports read,
    but the cache key carries chi, so they cannot collide with the chi=128 runs
    they supersede -- both remain on disk.

    Input: by_eps, the draws keyed by epsilon.
    Output: list of job dicts (40 = 4 epsilons x 10 samples).
    """
    return [{"kind": "grid", "epsilon": eps, "sample": s, "N": HIGHCHI_N,
             "init": "zero", "chi_max": HIGHCHI_CHI,
             "h": by_eps[eps][s]["h"], "c2": by_eps[eps][s]["c2"]}
            for eps in EPSILONS for s in ALL_SAMPLES]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def report_exact(table: dict) -> None:
    """R/|c|^2 and gap_rate from the dense solves -- the exponent, and the
    premise that one schedule serves every epsilon."""
    print("\n" + "=" * 92)
    print("EXACT sector solve -- R/|c|^2 (the epsilon exponent) and gap_rate")
    print("=" * 92)
    print(f"{'N':>3} {'eps':>6} {'mean R/|c|^2':>14} {'spread':>9} "
          f"{'mean gap_rate':>14} {'spread':>9}")
    for N in EXACT_SIZES:
        for eps in EPSILONS:
            v = np.array([table[(N, eps, s)]["R_over_c2"] for s in ALL_SAMPLES])
            g = np.array([table[(N, eps, s)]["gap_rate"] for s in ALL_SAMPLES])
            print(f"{N:>3} {eps:>6.2f} {v.mean():>14.6f} "
                  f"{np.ptp(v) / v.mean():>8.2%} {g.mean():>14.4f} "
                  f"{np.ptp(g) / g.mean():>8.2%}")
    print("\n  R/|c|^2 flat in epsilon at fixed N  <=>  R ~ epsilon^2, since")
    print("  |c|^2 ~ epsilon^2 by construction. No fit, no schedule, no truncation.")
    print("  gap_rate flat in epsilon is what licenses ONE schedule for all epsilon.")


def report_validation(results: dict, table: dict) -> None:
    """TEBD against exact at N in EXACT_SIZES -- every sample, not a spot check."""
    print("\n" + "=" * 92)
    print("VALIDATION -- TEBD vs exact (certifies schedule AND truncation)")
    print("=" * 92)
    print(f"{'N':>3} {'eps':>6} {'max |rel err|':>14} {'mean |rel err|':>15} "
          f"{'max bond':>9} {'at cap':>7}")
    for N in EXACT_SIZES:
        for eps in EPSILONS:
            errs, bonds, capped = [], [], 0
            for s in ALL_SAMPLES:
                key = ("grid", eps, s, N, "zero", CHI)
                if key not in results:
                    continue
                r = results[key]["result"]
                Rx = table[(N, eps, s)]["R"]
                errs.append(abs(r["correlator"] - Rx) / abs(Rx))
                bonds.append(r["max_bond"])
                capped += bool(r["bond_at_cap"])
            if errs:
                print(f"{N:>3} {eps:>6.2f} {max(errs):>13.3%} "
                      f"{np.mean(errs):>14.3%} {max(bonds):>9} "
                      f"{capped:>4}/{len(errs)}")


def report_size_scan(results: dict, table: dict) -> None:
    """R/|c|^2 vs N -- the long-range-order question the exact solve cannot reach."""
    print("\n" + "=" * 92)
    print("SIZE SCAN -- R/|c|^2 vs N  (exact where available, else TEBD)")
    print("=" * 92)
    header = f"{'eps':>6}" + "".join(f"{('N=%d' % N):>13}" for N in TEBD_SIZES)
    print(header)
    for eps in EPSILONS:
        row = f"{eps:>6.2f}"
        for N in TEBD_SIZES:
            vals = []
            for s in ALL_SAMPLES:
                key = ("grid", eps, s, N, "zero", CHI)
                if key in results:
                    c2 = results[key]["c2"]
                    vals.append(results[key]["result"]["correlator"] / c2)
            row += f"{np.mean(vals):>13.6f}" if vals else f"{'--':>13}"
        print(row)
    print("\n  Flat across a row  <=>  R ~ epsilon^2 holds at that size too.")
    print("  The trend ALONG a row is the long-range-order question.")

    capped = [(k, v) for k, v in results.items()
              if v.get("result", {}).get("bond_at_cap")]
    if capped:
        sizes = sorted({k[3] for k, _ in capped})
        print(f"\n  !! {len(capped)} runs hit the chi={CHI} cap, at N in {sizes}.")
        print(f"     Their truncation error is NOT bounded by the N<={max(EXACT_SIZES)}")
        print("     agreement above. Run --convergence before trusting them.")


def report_residuals(results: dict, table: dict) -> None:
    """Residual, read against the dt^2 Trotter floor AND the slow-mode gap.

    CLAUDE.md Trap 2: r ~ gap * delta, so a small residual is compatible with a
    state far from the fixed point when the gap is small -- that is how sample 8
    of the drift model sat on its apparent floor while being 19x off in R.
    Dividing by the gap is the one line that separates the two readings.

    The gap here is the exact sector gap_rate, measured at N in EXACT_SIZES and
    extrapolated as ~1/N^2 above that (gap(6)/gap(4) = 0.418 vs diffusive
    0.444, measured 2026-09-23). Marked '~' where extrapolated, since the
    extrapolation is the assumption and not a measurement.

    Input: results, the run dict; table, the exact reference.
    Output: None (prints).
    """
    dt2_floor = 0.0202 * DT_SCHEDULE[-1] ** 2   # C measured at N=4, CLAUDE.md
    g4 = np.mean([table[(EXACT_SIZES[0], eps, s)]["gap_rate"]
                  for eps in EPSILONS for s in ALL_SAMPLES])
    print("\n" + "=" * 92)
    print("RESIDUALS -- vs the dt^2 floor and vs the slow-mode gap (Trap 2)")
    print("=" * 92)
    print(f"  dt^2 Trotter floor = 0.0202 * {DT_SCHEDULE[-1]}^2 = {dt2_floor:.2e}")
    print(f"\n{'N':>3} {'gap':>10} {'mean resid':>12} {'max resid':>12} "
          f"{'max r/floor':>12} {'max r/gap':>11}")
    for N in TEBD_SIZES:
        gap = (np.mean([table[(N, eps, s)]["gap_rate"]
                        for eps in EPSILONS for s in ALL_SAMPLES])
               if N in EXACT_SIZES else g4 * (EXACT_SIZES[0] / N) ** 2)
        rs = [results[k]["result"]["residual"].get("residual_per_bond")
              for k in results if k[3] == N and k[0] == "grid"]
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
        mark = "" if N in EXACT_SIZES else "~"
        print(f"{N:>3} {mark + f'{gap:.4f}':>10} {np.mean(rs):>12.2e} "
              f"{max(rs):>12.2e} {max(rs) / dt2_floor:>12.1f} "
              f"{max(rs) / gap:>11.2e}")
    print("\n  r/gap is the distance to the fixed point in norm. R is QUADRATIC")
    print("  in rho, so a few percent there is a large error in R.")
    print("  At N <= %d the exact comparison above supersedes all of this."
          % max(EXACT_SIZES))


def report_highchi(results: dict, table: dict) -> None:
    """The --highchi row, against the chi=CHI row it supersedes.

    The decisive column is 'spread': R/|c|^2 must be eps-INDEPENDENT, which the
    exact solves prove at N<=6 to 0.14%. At chi=128 the N=12 row spread 38%;
    recovering a small spread is what says the truncation error is gone, and it
    is a stronger test than any single number moving.

    Input: results, the high-chi runs; table, the exact reference.
    Output: None (prints).
    """
    N, chi = HIGHCHI_N, HIGHCHI_CHI
    print("\n" + "=" * 92)
    print(f"HIGH-CHI ROW: N={N} at chi={chi}")
    print("=" * 92)
    print(f"{'eps':>6} {'mean R/|c|^2':>14} {'spread':>9} {'max drift':>10} "
          f"{'max bond':>9} {'at cap':>7}")
    allv = []
    for eps in EPSILONS:
        v, dr, bo, cap = [], [], [], 0
        for s in ALL_SAMPLES:
            k = ("grid", eps, s, N, "zero", chi)
            if k not in results:
                continue
            r = results[k]["result"]
            v.append(r["correlator"] / results[k]["c2"])
            dr.append(abs(r["stage_drift"]))
            bo.append(r["max_bond"])
            cap += bool(r["bond_at_cap"])
        if v:
            allv += v
            print(f"{eps:>6.2f} {np.mean(v):>14.6f} {np.ptp(v)/np.mean(v):>8.2%} "
                  f"{max(dr):>9.2%} {max(bo):>9} {cap:>4}/{len(v)}")
    if allv:
        print(f"\n  pooled over all epsilon: {np.mean(allv):.6f} "
              f"+/- {np.std(allv):.6f}   (spread {np.ptp(allv)/np.mean(allv):.2%})")
        # Only N=12 has a chi=CHI row to be compared against; at any other N
        # the main grid never ran, so quoting that baseline would be false.
        if N in TEBD_SIZES:
            prior = [results[k]["result"]["correlator"] / results[k]["c2"]
                     for k in results if k[3] == N and k[5] == CHI]
            if prior:
                print(f"  chi={CHI} at this N gave {np.mean(prior):.6f} "
                      f"(spread {np.ptp(prior)/np.mean(prior):.1%})")
            else:
                print(f"  (chi={CHI} row for N={N} is in ham_finite_size.pkl: "
                      f"0.022868, 38% spread -- the corrupted one)")
        else:
            print(f"  no chi={CHI} row exists at N={N}; the main grid stopped "
                  f"at N={max(TEBD_SIZES)}.")
        for n in EXACT_SIZES:
            ev = [table[(n, e, s)]["R_over_c2"] for e in EPSILONS for s in ALL_SAMPLES]
            print(f"  exact N={n}: {np.mean(ev):.6f}")


def report_controls(results: dict) -> None:
    """h=0 floor and the |neel> cross-check."""
    print("\n" + "=" * 92)
    print("CONTROLS")
    print("=" * 92)
    for key in sorted([k for k in results if k[0] == "control"], key=lambda k: k[3]):
        r = results[key]["result"]
        print(f"  h=0    N={key[3]:>3}  R={r['correlator']:>12.4e}  "
              f"bond={r['max_bond']:>3}  (dark vacuum: must be ~0)")
    for key in sorted([k for k in results if k[0] == "neel"], key=lambda k: k[2]):
        r = results[key]["result"]
        zero_key = ("grid", key[1], key[2], key[3], "zero", CHI)
        if zero_key in results:
            Rz = results[zero_key]["result"]["correlator"]
            rel = abs(r["correlator"] - Rz) / abs(Rz) if Rz else float("nan")
            print(f"  neel   N={key[3]:>3} sample{key[2]}  "
                  f"R={r['correlator']:>12.4e}  vs zero-start {Rz:>12.4e}  "
                  f"({rel:>6.2%})")
    print("\n  Two starts agreeing is consistency, NOT evidence of relaxation")
    print("  (CLAUDE.md Trap 5). The exact comparison is the real certification.")


def report_convergence(results: dict) -> None:
    """The chi sweep at CONV_N, one block per probe.

    'discarded' is the column that matters, not 'max bond': the N=12 damage is
    done in the transient, and the final bond hides it (see CONV_N's comment).
    N=8's value, 1.3e-08, is the reference for what an uncorrupted run looks
    like at this schedule.
    """
    print("\n" + "=" * 92)
    print(f"CHI CONVERGENCE at N in {CONV_NS}")
    print("=" * 92)
    for N in CONV_NS:
      for eps, s in CONV_PROBES:
        print(f"\n  probe: N={N}, epsilon={eps:.2f}, sample{s}")
        print(f"  {'chi':>5} {'R':>13} {'R/|c|^2':>10} {'vs prev':>9} "
              f"{'bond':>6} {'cap':>6} {'drift':>8} {'discarded':>11} "
              f"{'stages (R/|c|^2)':>16}")
        prev = None
        for chi in CONV_CHIS:
            key = ("conv", eps, s, N, "zero", chi)
            if key not in results:
                continue
            r = results[key]["result"]
            c2 = results[key]["c2"]
            R = r["correlator"]
            rel = f"{abs(R - prev) / abs(prev):>8.2%}" if prev else f"{'--':>9}"
            stages = " ".join(f"{v / c2:.4f}"
                              for _, v in r["stage_correlators"][-3:])
            print(f"  {chi:>5} {R:>13.5e} {R / c2:>10.6f} {rel} "
                  f"{r['max_bond']:>6} {str(r['bond_at_cap']):>6} "
                  f"{r['stage_drift']:>7.2%} {r['max_discarded_weight']:>11.2e} "
                  f"  {stages}")
            prev = R

    print("\n  'vs prev' is the truncation error estimate: it must SHRINK.")
    print("  The last three stage values must settle -- at chi=128 sample 6 rose")
    print("  37% on its final stage, which is the failure this sweep tests.")
    # Plain ASCII, deliberately: a non-cp1252 character in a print() raises
    # UnicodeEncodeError on the Windows console and killed this report's run
    # AFTER it had emitted the whole table but BEFORE the pickle was written
    # (2026-09-25). Emoji are fine in docstrings and comments, never in output.
    print("\n  !! 'discarded' is NOT the indicator to read here. Measured")
    print("  2026-09-24: at chi=256 it is still 2.9e-05 (eps=0.05) and 9.7e-05")
    print("  (eps=0.20), barely below chi=128 and four orders above the N=8")
    print("  reference 1.3e-08 -- yet R has converged to 0.22-4% and the stages")
    print("  have settled. It is a max over the whole run, so it is pinned by the")
    print("  entangled TRANSIENT and says nothing about the accuracy of the FINAL")
    print("  state. It is the right alarm for 'is truncation active anywhere',")
    print("  and the wrong gauge for 'is this number converged'. Use drift, the")
    print("  stage tail and 'vs prev' for that.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_size_scan(results: dict, table: dict) -> str:
    """R/|c|^2 vs N, one line per epsilon, with the exact points marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for eps in EPSILONS:
        xs, ys, es = [], [], []
        for N in TEBD_SIZES:
            vals = [results[("grid", eps, s, N, "zero", CHI)]["result"]["correlator"]
                    / results[("grid", eps, s, N, "zero", CHI)]["c2"]
                    for s in ALL_SAMPLES
                    if ("grid", eps, s, N, "zero", CHI) in results]
            if vals:
                xs.append(N); ys.append(np.mean(vals)); es.append(np.std(vals))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                        label=f"$\\epsilon={eps:.2f}$ (TEBD)")
        ex_x = [N for N in EXACT_SIZES]
        ex_y = [np.mean([table[(N, eps, s)]["R_over_c2"] for s in ALL_SAMPLES])
                for N in ex_x]
        ax.plot(ex_x, ex_y, marker="x", ls="--", ms=9, mew=2, color="k",
                alpha=0.35, label="exact" if eps == EPSILONS[0] else None)

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$R\,/\,|c|^2$")
    ax.set_title(r"Hamiltonian perturbation: $R/|c|^2$ vs size"
                 f"  ($\\chi={CHI}$)")
    ax.set_xticks(TEBD_SIZES)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(layout.MODEL3_RESULTS, "ham_finite_size_R_vs_N.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profiles(results: dict) -> str:
    """R(i, r) at the largest TEBD size, one line per epsilon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = max(TEBD_SIZES)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for eps in EPSILONS:
        for s in ALL_SAMPLES:
            key = ("grid", eps, s, N, "zero", CHI)
            if key not in results:
                continue
            prof = results[key]["result"]["profile"]
            c2 = results[key]["c2"]
            ax.plot([r for r, _ in prof], [v / c2 for _, v in prof],
                    alpha=0.5, lw=1,
                    color=f"C{EPSILONS.index(eps)}",
                    label=f"$\\epsilon={eps:.2f}$" if s == 0 else None)
    ax.set_xlabel("$r$")
    ax.set_ylabel(r"$R(i, r)\,/\,|c|^2$")
    ax.set_title(f"Correlator profiles at $N={N}$, $\\chi={CHI}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(layout.MODEL3_RESULTS, "ham_finite_size_profiles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def execute(jobs: list[dict], label: str) -> dict:
    """Farm jobs to the pool, printing one line per completion."""
    # chi comes from the jobs, not from CHI: --highchi and --convergence both
    # run at other bond dimensions, and a banner that always said CHI would
    # mislabel exactly the runs whose chi is the point of the run.
    chis = sorted({j["chi_max"] for j in jobs})
    print(f"\n{len(jobs)} {label} jobs on {min(N_WORKERS, len(jobs))} workers, "
          f"chi={chis[0] if len(chis) == 1 else chis}.\n", flush=True)
    results, failures = {}, []
    t0 = time.perf_counter()
    with mp.Pool(processes=min(N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_job, jobs):
            key = (out["kind"], out["epsilon"], out["sample"], out["N"],
                   out["init"], out["chi_max"])
            s = "ctrl" if out["sample"] is None else f"s{out['sample']}"
            tag = (f"  {out['kind']:>7} eps={out['epsilon']:.2f} {s:>4} "
                   f"N={out['N']:>3} chi={out['chi_max']:>3}")
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue
            results[key] = out
            r = out["result"]
            timing = "cached" if out["cached"] else f"{out['seconds'] / 60:.1f}min"
            print(f"{tag}  R={r['correlator']:>12.4e}  bond={r['max_bond']:>3}"
                  f"{'*' if r['bond_at_cap'] else ' '}  "
                  f"drift={r['stage_drift']:>7.2%}  "
                  f"[{timing}, elapsed {(time.perf_counter() - t0) / 60:.1f}min]",
                  flush=True)
    print(f"\n{len(results)}/{len(jobs)} succeeded in "
          f"{(time.perf_counter() - t0) / 60:.1f} min.", flush=True)
    if failures:
        print(f"{len(failures)} FAILED.", flush=True)
    return results


def save(payload: dict, name: str) -> str:
    os.makedirs(layout.MODEL3_RESULTS, exist_ok=True)
    path = os.path.join(layout.MODEL3_RESULTS, name)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def main() -> None:
    by_eps, _ = ham.build_sizing()

    if "--highchi" in sys.argv:
        jobs = build_highchi_jobs(by_eps)
        results = execute(jobs, f"N={HIGHCHI_N} chi={HIGHCHI_CHI}")
        report_highchi(results, d_exact(by_eps))
        payload = {"config": {"N": HIGHCHI_N, "chi": HIGHCHI_CHI,
                              "epsilons": EPSILONS, "samples": ALL_SAMPLES,
                              "dt_schedule": list(DT_SCHEDULE),
                              "steps_per_dt": STEPS_PER_DT},
                   "results": results}
        print(f"\nSaved -> {save(payload, 'ham_finite_size_highchi.pkl')}",
              flush=True)
        return

    if "--convergence" in sys.argv:
        results = execute(build_convergence_jobs(by_eps), "convergence")
        report_convergence(results)
        payload = {
            "config": {"Ns": CONV_NS, "chis": CONV_CHIS, "probes": CONV_PROBES},
            "results": results,
        }
        print(f"\nSaved -> {save(payload, 'ham_finite_size_convergence.pkl')}",
              flush=True)
        return

    print("Computing exact reference (dense sector solves) ...", flush=True)
    t0 = time.perf_counter()
    table = exact_table(by_eps)
    print(f"  {len(table)} exact solves in {time.perf_counter() - t0:.1f}s",
          flush=True)
    report_exact(table)

    results = execute(build_jobs(by_eps), "grid")

    for name, fn in (("exact", lambda: None),
                     ("validation", lambda: report_validation(results, table)),
                     ("size scan", lambda: report_size_scan(results, table)),
                     ("residuals", lambda: report_residuals(results, table)),
                     ("controls", lambda: report_controls(results))):
        try:
            fn()
        except Exception:
            print(f"!! {name} report failed (data is saved):\n"
                  f"{traceback.format_exc()}", flush=True)

    path = save({
        "config": {"perturbation": PERTURBATION, "epsilons": EPSILONS,
                   "samples": ALL_SAMPLES, "exact_sizes": EXACT_SIZES,
                   "tebd_sizes": TEBD_SIZES, "chi": CHI, "seed": BASE_SEED,
                   "dt_schedule": list(DT_SCHEDULE),
                   "steps_per_dt": STEPS_PER_DT},
        "exact": table, "results": results,
    }, "ham_finite_size.pkl")
    print(f"\nSaved -> {path}", flush=True)

    for name, fn in (("R vs N", lambda: plot_size_scan(results, table)),
                     ("profiles", lambda: plot_profiles(results))):
        try:
            print(f"Saved {name:<10s} -> {fn()}", flush=True)
        except Exception:
            print(f"!! {name} plot failed (data is saved):\n"
                  f"{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
