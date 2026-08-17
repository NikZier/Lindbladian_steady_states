"""Renyi-2 (SWSSB) correlator of the parity-symmetric dissipative chain under
random parity-commuting perturbations, as a function of system size, from two
strongly-symmetric initial states.

Model (purely dissipative, H = 0), bond jump operators applied uniformly to
every nearest-neighbour bond:

    L   = X_a X_{a+1} (1 - Z_a Z_{a+1})           rate 1
    L'  = X_a X_{a+1} (1 - Z_a)(1 - Z_{a+1})      rate 1
    L'' = random, [L'', Z(x)Z] = 0, ||L''|| = epsilon = 0.2   rate 1

Every jump operator commutes with the strong Z_2 symmetry P = Z_1...Z_N, so
the dynamics preserve the strong-symmetry sector of the initial state. We
start TEBD from two computational-basis (pure, strongly-symmetric) states,
both in the same (+,+) parity sector so their results are directly comparable:

    'zero' : |0...0><0...0|  -- a DARK state of the baseline (both L, L'
             annihilate it), so the baseline correlator is exactly 0 and any
             signal is driven purely by L''.
    'neel' : |0101...><0101...|  -- NOT a dark state (L drives every |01>
             bond), so the baseline alone already flows to a nontrivial
             steady state.

Comparing the two starts probes whether the (+,+)-sector steady state is
unique (both give the same R) or degenerate / symmetry-broken (they differ).

For each initial state, random L'' and size N in {4, 8, 12, 16, 20} we find
the steady state by TEBD and evaluate the Renyi-2 correlator

    R(i, j) = Tr[rho A rho^dag A^dag] / Tr[rho^dag rho],   A = X_i X_j

at i = N//4, j = 3N//4 (separation N/2, away from the open boundaries), plus
the full profile R(i, r) for every r > i at every size.

Convergence
-----------
Bond dimension is checked separately by re-running one L'' sample at
N = CONV_N over CHI_CONVERGENCE, from the 'zero' start. N=20 is where the
truncation is most strained (at chi=32 fourteen of nineteen bonds sit at the
cap), so the check is done there rather than at a smaller size. Note that
run-to-run reproducibility of R is only ~1% -- BLAS summation order alone
shifts it that much through 1500 steps of truncated non-unitary evolution --
so the sweep resolves "is chi=32 adequate at the percent level", no finer.

Execution
---------
The (initial state, L'', N) runs are completely independent, and BLAS is
pinned to one thread inside TEBD (see lindblad_mps.blas), so they are farmed
out to N_WORKERS processes. Jobs are queued longest-first to keep the tail
short.

Outputs (experiments/results/):
    renyi2_swssb.pkl -- config, per-sample L'' descriptions (matrix, Pauli
        coefficients, operator norm, seed) and correlator results for both
        initial states, plus the L''=0 baseline.
    renyi2_swssb_chi_convergence.pkl -- the bond-dimension sweep, with the
        description of the L'' it used.
    renyi2_swssb.png        -- R vs N, one panel row per initial state.
    renyi2_swssb_profile.png -- full R(i, r) profile at N = PROFILE_SIZE.
    renyi2_swssb_chi.png    -- correlator and profile vs bond dimension.
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lindblad_mps import exact, models, observables, residual, tebd, vectorize
from lindblad_mps import mps as mps_module

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
EPSILON = 0.2
SIZES = [4, 8, 12, 16, 20]
N_SAMPLES = 10
BASE_SEED = 20260726
INITIAL_STATES = ["zero", "neel"]  # both must share the parity sector (asserted)

CHI_MAX = 32
CUTOFF = 1e-10
DT_SCHEDULE = [0.1, 0.05, 0.02, 0.01, 0.005]
STEPS_PER_DT = 300
RECANON_EVERY = 10

PROFILE_SIZE = 20  # size whose full R(i, r) profile is plotted

CHI_CONVERGENCE = [8, 16, 32, 48, 64]
CONV_N = 20
CONV_INIT = "zero"
CONV_SAMPLE = 0  # which L'' sample the chi sweep uses

# 12, not 6: measured 2026-07-28 on the 6-core/12-thread desktop, SMT gives
# 1.59x throughput at chi=32 (12 jobs in 183.7 s vs 6 in 145.7 s) and ~1.46x at
# chi=128, where the larger SVDs press harder on cache. Memory is not a
# constraint (~20 MB per worker). Drop back to 6 on a machine without SMT.
N_WORKERS = 12

CONVERGENCE_TOL = 1e-6  # 1 - |<rho(t)|rho(t+dt)>| below this counts as converged

# 'converged' above is a weak test and is kept only for continuity with the
# existing pickles: it measures the change over ONE Trotter step, which shrinks
# with dt whether or not the state is near the fixed point, so a slowly-drifting
# state passes it trivially. Measured case: the L''=0 'neel' control at chi=128
# reports converged=True with 1-overlap = 6.7e-12 at N=20 while sitting at bond
# dimension 56, though its exact steady state is the dark state at bond
# dimension 1 -- the relaxation front had not finished crossing the chain.
# STAGE_DRIFT_TOL tests the honest thing instead: the relative change in the
# correlator between the last two dt stages.
STAGE_DRIFT_TOL = 1e-2

# R = Tr[rho A rho A]/Tr[rho^2] with A = X_i X_j Hermitian and unitary equals
# Tr[(A rho A) rho], a trace of two positive semidefinite matrices, so R >= 0
# for any physical rho. Truncation destroys positivity, and a negative R is a
# direct measure of how much: it is the one error the pipeline can detect
# without knowing the right answer. Recorded, never raised -- runs that are
# merely truncation-limited are still worth keeping (the chi=128 L''=0 control
# reaches -3.8e-06), and run_job would otherwise bank them as failures.
POSITIVITY_TOL = 1e-12  # |negative R| above this is worth flagging

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "_cache")  # one pickle per completed run

_KET0 = np.array([1, 0], dtype=complex)
_KET1 = np.array([0, 1], dtype=complex)


def correlator_sites(N: int) -> tuple[int, int]:
    """Return the (i, j) site pair used for the reported correlator at size N.

    i = N//4, j = 3N//4: separation N/2, kept away from the open boundaries.
    """
    return N // 4, (3 * N) // 4


def basis_bits(name: str, N: int) -> list[int]:
    """Return the 0/1 bit string of the named computational-basis initial state.

    'zero' -> all zeros (dark state of the baseline).
    'neel' -> 0,1,0,1,... (non-dark).
    """
    if name == "zero":
        return [0] * N
    if name == "neel":
        return [i % 2 for i in range(N)]
    raise ValueError(f"unknown initial state '{name}'")


def parity_charge(bits: list[int]) -> int:
    """Strong-symmetry charge P = Z_1...Z_N eigenvalue of |bits>: (-1)^(#ones)."""
    return -1 if (sum(bits) % 2) else 1


def build_initial_state(name: str, N: int) -> mps_module.MPS:
    """Build the bond-dim-1 MPS for the named pure computational-basis density matrix.

    Input: name ('zero' or 'neel'); N, number of sites.
    Output: an MPS representing |bits><bits| in the local-vec convention.
    """
    kets = [_KET0 if b == 0 else _KET1 for b in basis_bits(name, N)]
    return mps_module.MPS.pure_product_state(kets)


def build_L2_terms(
    L_pp: np.ndarray | None, baseline: list[np.ndarray] | None = None
) -> list[tuple[np.ndarray, float]]:
    """Assemble the list of (bond jump operator, rate) terms for the model.

    Input: L_pp, the perturbation operator L'' (4x4), or None for the
        unperturbed baseline; baseline, the list of baseline bond jumps to use,
        defaulting to models.baseline_jump_operators(). A second study
        (renyi2_drift_annihilation.py) passes the classical biased-hopping /
        pair-annihilation jumps here; everything downstream -- the correlator,
        the diagnostics, the cache -- is model-agnostic, so only this one hook
        is needed.
    Output: list of (op, rate) pairs: the baseline jumps plus, if given,
        L'', all at rate 1.0.
    """
    ops = models.baseline_jump_operators() if baseline is None else list(baseline)
    terms = [(op, 1.0) for op in ops]
    if L_pp is not None:
        terms.append((L_pp, 1.0))
    return terms


def run_steady_state_correlator(
    L2_terms: list[tuple[np.ndarray, float]],
    N: int,
    init_name: str,
    chi_max: int = CHI_MAX,
) -> dict:
    """Find the TEBD steady state at size N (from init_name) and measure R(i,j).

    Runs tebd.find_steady_state with the module-level truncation / schedule
    settings from the chosen strongly-symmetric initial state, then measures
    R(i, j) at i = N//4, j = 3N//4 with order parameter O = X, and records the
    full profile R(i, r) for r > i.

    R is also measured at the end of every dt stage, which costs ~50 ms against
    a run of many minutes and is what actually establishes convergence in time
    (see STAGE_DRIFT_TOL). The most negative value over the profile is recorded
    as a positivity check (see POSITIVITY_TOL).

    Input:
        L2_terms: bond jump terms (op, rate) (H and single-site terms empty).
        N: number of sites.
        init_name: 'zero' or 'neel'.
        chi_max: bond-dimension cap (defaults to the study-wide CHI_MAX; the
            convergence sweep overrides it).
    Output: dict with keys 'N', 'init', 'chi_max', 'i', 'j', 'correlator',
        'profile' (list of (r, R(i, r))), 'final_overlap',
        'max_discarded_weight', 'final_bond_dims', 'converged', 'state' (MPS),
        plus 'stage_correlators' (list of (dt, R) per dt stage), 'stage_drift'
        (relative change in R over the last stage), 'time_converged'
        (stage_drift < STAGE_DRIFT_TOL), 'min_profile',
        'positivity_violation' (see POSITIVITY_TOL), and 'residual' /
        'residual_per_bond' -- ||L|rho>||/|||rho>||, the absolute convergence
        test (see lindblad_mps.residual), None if it raised.
    """
    i, j = correlator_sites(N)
    stage_correlators = []

    def record_stage(stage: int, dt: float, st: mps_module.MPS) -> None:
        """Sample R at a stage boundary (see tebd.find_steady_state)."""
        stage_correlators.append((dt, observables.renyi2_correlator_mps(st, models.X, i, j)))

    state, history = tebd.find_steady_state(
        H2_terms=[],
        H1_terms=[],
        L2_terms=L2_terms,
        L1_terms=[],
        N=N,
        dt_schedule=DT_SCHEDULE,
        steps_per_dt=STEPS_PER_DT,
        chi_max=chi_max,
        cutoff=CUTOFF,
        recanonicalize_every=RECANON_EVERY,
        initial_state=build_initial_state(init_name, N),
        stage_callback=record_stage,
    )

    R = observables.renyi2_correlator_mps(state, models.X, i, j)
    profile = [
        (r, observables.renyi2_correlator_mps(state, models.X, i, r))
        for r in range(i + 1, N)
    ]

    # Drift over the final stage. Scaled by the larger magnitude so that a
    # correlator collapsing towards zero (the L''=0 case) reads as small drift
    # rather than dividing by a vanishing denominator.
    if len(stage_correlators) >= 2:
        last, prev = stage_correlators[-1][1], stage_correlators[-2][1]
        scale = max(abs(last), abs(prev))
        stage_drift = abs(last - prev) / scale if scale > 0 else 0.0
    else:
        stage_drift = float("nan")

    min_profile = min([R] + [v for _, v in profile])

    # The absolute convergence test: ||L|rho>|| / |||rho>|| is zero at a steady
    # state of any sector and says nothing about dt, the initial state or the
    # schedule -- unlike 'converged' and 'stage_drift', which both measure the
    # trajectory rather than the state. Read it against the dt^2 Trotter floor
    # of the final stage, not against zero (see lindblad_mps.residual). Costs a
    # few seconds at the bond dimensions this study reaches.
    #
    # Diagnostics must never cost a run: caught and recorded as None, the same
    # policy as positivity_violation.
    try:
        residual_diag = residual.steady_state_residual(state, [], [], L2_terms, [])
    except Exception:
        residual_diag = {}

    final_overlap = history["overlap"][-1] if history["overlap"] else float("nan")
    return {
        "N": N,
        "init": init_name,
        "chi_max": chi_max,
        "i": i,
        "j": j,
        "correlator": R,
        "profile": profile,
        "final_overlap": final_overlap,
        "max_discarded_weight": max(history["discarded_weight"], default=0.0),
        "final_bond_dims": list(state.bond_dims),
        "converged": (1.0 - final_overlap) < CONVERGENCE_TOL,
        "stage_correlators": stage_correlators,
        "stage_drift": stage_drift,
        "time_converged": stage_drift < STAGE_DRIFT_TOL,
        "min_profile": min_profile,
        "positivity_violation": max(0.0, -min_profile) > POSITIVITY_TOL,
        "residual": residual_diag.get("residual"),
        "residual_per_bond": residual_diag.get("residual_per_bond"),
        "state": state,
    }


def _strip_state(res: dict) -> dict:
    """Drop the MPS object from a run result (keeps the pickle small)."""
    return {k: v for k, v in res.items() if k != "state"}


def format_diagnostics(res: dict) -> str:
    """One-line convergence/positivity summary of a run result, for progress output.

    Marks a run that is still drifting between its last two dt stages with '!'
    and one whose correlator went negative with 'NEG' (see STAGE_DRIFT_TOL,
    POSITIVITY_TOL). Cache entries written before these diagnostics existed
    lack the keys and fall back to the old 'conv=' field.

    Input: res, a run result dict from run_steady_state_correlator().
    Output: a short string, padded to a fixed width so columns line up.
    """
    if "stage_drift" not in res:
        return f"conv={str(res['converged']):5s}(old)  "
    drift = res["stage_drift"]
    # Absent on every entry cached before the residual diagnostic existed, and
    # re-running will not add it: the cache key and run_config are both
    # unchanged by it, so a cached run is returned as it stands. Delete the
    # cache file of any specific run you want the residual for.
    resid = res.get("residual")
    resid_field = f"res={resid:7.1e} " if resid is not None else " " * 12
    return (f"drift={drift:7.1e}{' ' if res['time_converged'] else '!'}"
            f"{'NEG ' if res['positivity_violation'] else '    '}"
            f"{resid_field}")


# ---------------------------------------------------------------------------
# Parallel job plumbing
# ---------------------------------------------------------------------------
def job_key(job: dict) -> str:
    """Filename-safe identifier for one job, used as its cache key."""
    return (f"{job['kind']}_{job['label']}_{job['init']}"
            f"_N{job['N']}_chi{job['chi_max']}")


def run_config(job: dict | None = None) -> dict:
    """The settings that change a run's result but do NOT appear in its cache key.

    The key covers (kind, label, init, N, chi_max) only, so lengthening the dt
    schedule -- exactly the fix if a run turns out not to be converged in time
    -- would otherwise be silently defeated by the cache handing back the old,
    shorter run under the same name. run_job compares this against the value
    stored in a cache file and recomputes on a mismatch.

    A job carrying a non-default baseline (job['model'], see build_L2_terms)
    adds it to the config, so two models can never alias onto one cache entry.
    The key is added only when present, which is what keeps every entry cached
    before the second model existed matching as it stands -- adding it
    unconditionally would invalidate the whole committed cache.

    Input: job, the job dict about to run (or None for the default model).
    Output: dict of the schedule/truncation settings a result depends on.
    """
    config = {
        "dt_schedule": list(DT_SCHEDULE),
        "steps_per_dt": STEPS_PER_DT,
        "recanonicalize_every": RECANON_EVERY,
        "cutoff": CUTOFF,
    }
    if job is not None and job.get("model") is not None:
        config["model"] = job["model"]
    return config


# The settings every cache entry written before run_config() existed was
# produced with. Pinned as literals, not read from the constants above: the
# whole point is to detect a later edit to those constants, so deriving this
# from them would make every legacy entry match whatever the constants happen
# to say and silently defeat the check.
LEGACY_RUN_CONFIG = {
    "dt_schedule": [0.1, 0.05, 0.02, 0.01, 0.005],
    "steps_per_dt": 300,
    "recanonicalize_every": 10,
    "cutoff": 1e-10,
}


def cache_is_current(out: dict, job: dict | None = None) -> bool:
    """Whether a loaded cache entry was produced with the settings in force now.

    Input: out, a dict unpickled from a cache file; job, the job it would be
        reused for (its model, if any, is part of the comparison).
    Output: True if it may be reused, False if it must be recomputed.
    """
    return out.get("run_config", LEGACY_RUN_CONFIG) == run_config(job)


def run_job(job: dict) -> dict:
    """Execute one steady-state run described by a job dict (worker entry point).

    Must be a module-level function so that it survives pickling to the
    worker processes (Windows uses spawn, not fork).

    Every completed job is written to its own cache file before returning, and
    a cached job is returned without recomputing. A full sweep is ~45 minutes,
    so a crash (or a deliberate re-run after changing only part of the study)
    must not throw away the runs that already succeeded.

    Exceptions are caught and returned rather than raised: pool.imap_unordered
    re-raises in the parent and would abort every other worker, losing the
    whole sweep because one bond SVD failed to converge.

    Input: job dict with keys 'kind', 'label', 'L_pp', 'N', 'init', 'chi_max',
        and optionally 'baseline' / 'model' to run a model other than the
        default one (see build_L2_terms and run_config).
    Output: the job dict augmented with 'result' (state-stripped) and
        'seconds', or with 'error' (a traceback string) if the run raised.
    """
    path = os.path.join(CACHE_DIR, job_key(job) + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                out = pickle.load(f)
            if cache_is_current(out, job):
                out["cached"] = True
                return out
        except Exception:  # corrupt/partial cache file -- just recompute
            pass

    t0 = time.perf_counter()
    out = dict(job)
    out["cached"] = False
    out["run_config"] = run_config(job)
    try:
        res = run_steady_state_correlator(
            build_L2_terms(job["L_pp"], job.get("baseline")),
            job["N"], job["init"], job["chi_max"],
        )
        out["result"] = _strip_state(res)
    except Exception:
        out["error"] = traceback.format_exc()
    out["seconds"] = time.perf_counter() - t0

    if "error" not in out:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written file
    return out


def estimated_cost(job: dict) -> float:
    """Rough relative cost of a job, used to queue the long ones first.

    TEBD cost is dominated by one SVD per bond per step, which is O(chi^3)
    at fixed physical dimension, and there are N-1 bonds.
    """
    return (job["N"] - 1) * job["chi_max"] ** 3


def build_jobs(samples: list[dict]) -> list[dict]:
    """Enumerate every steady-state run the study needs, longest job first.

    Input: samples, the list of per-sample dicts (each with 'seed', 'L_pp',
        'description') produced by draw_samples().
    Output: list of job dicts ready for run_job(), sorted by descending
        estimated cost so the 6-worker pool has a short tail.
    """
    jobs = []
    for init in INITIAL_STATES:
        for N in SIZES:
            jobs.append({"kind": "main", "label": "baseline", "sample": None,
                         "L_pp": None, "N": N, "init": init, "chi_max": CHI_MAX})
            for s, sample in enumerate(samples):
                jobs.append({"kind": "main", "label": f"sample{s}", "sample": s,
                             "L_pp": sample["L_pp"], "N": N, "init": init,
                             "chi_max": CHI_MAX})

    conv_Lpp = samples[CONV_SAMPLE]["L_pp"]
    for chi in CHI_CONVERGENCE:
        if chi == CHI_MAX:
            continue  # already covered by the main grid; copied across afterwards
        jobs.append({"kind": "conv", "label": f"chi{chi}", "sample": CONV_SAMPLE,
                     "L_pp": conv_Lpp, "N": CONV_N, "init": CONV_INIT,
                     "chi_max": chi})

    jobs.sort(key=estimated_cost, reverse=True)
    return jobs


def draw_samples() -> list[dict]:
    """Draw the N_SAMPLES random parity-commuting L'' operators.

    Each is generated from its own seeded Generator so a sample can be
    reproduced (or extended to new sizes) independently of the others.

    Output: list of dicts with 'seed', 'L_pp' and 'description' (see
        models.describe_operator).
    """
    samples = []
    for s in range(N_SAMPLES):
        seed = BASE_SEED + 1 + s
        rng = np.random.default_rng(seed)
        L_pp = models.random_zz_commuting_operator(EPSILON, rng)
        assert models.commutes_with_zz(L_pp), "sampled L'' must commute with ZZ"
        samples.append({
            "seed": seed,
            "L_pp": L_pp,
            "description": models.describe_operator(L_pp, EPSILON, seed=seed),
        })
    return samples


# ---------------------------------------------------------------------------
# Validation against dense exact diagonalization
# ---------------------------------------------------------------------------
def validate_against_exact(
    L2_terms: list[tuple[np.ndarray, float]], init_name: str, N: int = 4
) -> dict:
    """Cross-check the TEBD pipeline against dense references at small N.

    Checks that do not assume a unique steady state:
      1. steady-state residual ||generator @ vec(rho)|| / ||vec(rho)||,
      2. strong-symmetry preservation ||P rho - q rho|| and ||rho P - q rho||
         (q = parity charge of init_name), confirming the run stayed in sector,
      3. MPS/dense correlator agreement.

    Input: L2_terms; init_name; N (default 4).
    Output: dict with 'residual', 'sym_breaking', 'correlator_abs_diff'.
    """
    result = run_steady_state_correlator(L2_terms, N, init_name)
    state = result["state"]
    rho = state.to_dense()

    jump_ops = exact.build_jump_operators(
        [(op, 1.0) for op, _ in L2_terms], L1_terms=[], N=N
    )
    generator = vectorize.liouvillian_generator([], jump_ops, d=2**N)
    v = vectorize.vec(rho)
    residual = float(np.linalg.norm(generator @ v) / max(np.linalg.norm(v), 1e-300))

    P = np.array([[1]], dtype=complex)
    for _ in range(N):
        P = np.kron(P, models.Z)
    q = parity_charge(basis_bits(init_name, N))
    sym = float(np.linalg.norm(P @ rho - q * rho) + np.linalg.norm(rho @ P - q * rho))
    sym /= max(np.linalg.norm(rho), 1e-300)

    i, j = result["i"], result["j"]
    R_dense = observables.renyi2_correlator_dense(rho, models.X, i, j, N)
    diff = abs(result["correlator"] - R_dense)
    return {"residual": residual, "sym_breaking": sym, "correlator_abs_diff": diff}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_experiment() -> tuple[dict, dict]:
    """Run the full study on a pool of N_WORKERS processes.

    Output: (results, convergence) -- two pickle-ready dicts (no live MPS
        objects anywhere).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Guard: all initial states must share the same parity sector at every size.
    for N in SIZES + [CONV_N]:
        charges = {name: parity_charge(basis_bits(name, N)) for name in INITIAL_STATES}
        assert len(set(charges.values())) == 1, (
            f"initial states span different parity sectors at N={N}: {charges}"
        )

    print("Validating TEBD pipeline against dense exact at N=4 ...", flush=True)
    validation = {}
    for init in INITIAL_STATES:
        chk = validate_against_exact(build_L2_terms(None), init, N=4)
        validation[init] = chk
        print(
            f"  [{init}] residual={chk['residual']:.2e}  "
            f"sym_breaking={chk['sym_breaking']:.2e}  "
            f"MPS/dense diff={chk['correlator_abs_diff']:.2e}",
            flush=True,
        )

    samples = draw_samples()
    for s, sample in enumerate(samples):
        print(f"  L'' sample {s}: seed={sample['seed']}  "
              f"||L''||={sample['description']['operator_norm']:.4f}", flush=True)

    jobs = build_jobs(samples)
    print(f"\n{len(jobs)} runs queued on {N_WORKERS} workers "
          f"(sizes {SIZES}, {N_SAMPLES} samples + baseline, "
          f"chi sweep {CHI_CONVERGENCE} at N={CONV_N}) ...\n", flush=True)

    config = {
        "epsilon": EPSILON,
        "sizes": SIZES,
        "n_samples": N_SAMPLES,
        "base_seed": BASE_SEED,
        "initial_states": INITIAL_STATES,
        "chi_max": CHI_MAX,
        "cutoff": CUTOFF,
        "dt_schedule": DT_SCHEDULE,
        "steps_per_dt": STEPS_PER_DT,
        "recanonicalize_every": RECANON_EVERY,
        "site_rule": "i = N//4, j = 3N//4",
        "profile_size": PROFILE_SIZE,
        "order_parameter": "X",
        "model": "L=XX(1-ZZ), L'=XX(1-Za)(1-Zb), L''=random parity-commuting, rates=1",
        "neel_note": "|0101...>, same (+,+) parity sector as |0...0> for all sizes used",
    }
    results = {
        "config": config,
        "validation_N4": validation,
        "baseline": {init: {} for init in INITIAL_STATES},
        "samples": [{"description": s["description"],
                     "results": {init: {} for init in INITIAL_STATES}}
                    for s in samples],
    }
    convergence = {
        "config": dict(config, chi_convergence=CHI_CONVERGENCE, conv_N=CONV_N,
                       conv_init=CONV_INIT, conv_sample=CONV_SAMPLE),
        "description": samples[CONV_SAMPLE]["description"],
        "results": {},
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    t_start = time.perf_counter()
    done = 0
    failures = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for out in pool.imap_unordered(run_job, jobs):
            done += 1
            tag = (f"  [{done:3d}/{len(jobs)}] {out['label']:>9s} {out['init']:>5s} "
                   f"N={out['N']:2d} chi={out['chi_max']:2d}")
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            res = out["result"]
            if out["kind"] == "main":
                if out["sample"] is None:
                    results["baseline"][out["init"]][out["N"]] = res
                else:
                    results["samples"][out["sample"]]["results"][out["init"]][out["N"]] = res
            else:
                convergence["results"][out["chi_max"]] = res
            timing = "cached" if out["cached"] else f"{out['seconds']:.0f}s"
            print(
                f"{tag}  R({res['i']},{res['j']})={res['correlator']:.6e}  "
                f"{format_diagnostics(res)} maxdisc={res['max_discarded_weight']:.1e}  "
                f"[{timing}, elapsed {(time.perf_counter()-t_start)/60:.1f}min]",
                flush=True,
            )

    # The chi = CHI_MAX point of the sweep is the corresponding main-grid run.
    conv_main = results["samples"][CONV_SAMPLE]["results"][CONV_INIT].get(CONV_N)
    if conv_main is not None:
        convergence["results"][CHI_MAX] = conv_main

    results["failures"] = [{k: v for k, v in f.items() if k != "L_pp"} for f in failures]
    convergence["failures"] = results["failures"]

    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t_start)/60:.1f} min.", flush=True)
    if failures:
        print(f"{len(failures)} FAILED (results for these are absent from the "
              f"pickles; re-running reuses the cache and retries only these):",
              flush=True)
        for f in failures:
            print(f"  {job_key(f)}", flush=True)
    return results, convergence


def save_results(results: dict, convergence: dict) -> tuple[str, str]:
    """Pickle the main results and the bond-dimension sweep.

    Output: (main_path, convergence_path).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main_path = os.path.join(RESULTS_DIR, "renyi2_swssb.pkl")
    conv_path = os.path.join(RESULTS_DIR, "renyi2_swssb_chi_convergence.pkl")
    with open(main_path, "wb") as f:
        pickle.dump(results, f)
    with open(conv_path, "wb") as f:
        pickle.dump(convergence, f)
    return main_path, conv_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _mpl():
    """Import pyplot with a headless backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_results(results: dict) -> str:
    """Plot R vs N: one row per initial state, columns linear and semilog."""
    plt = _mpl()
    sizes = results["config"]["sizes"]
    inits = results["config"]["initial_states"]
    n_samp = len(results["samples"])
    cmap = plt.get_cmap("viridis")

    def series(store):
        """(sizes, R) for the sizes actually present -- a failed run is skipped."""
        present = [N for N in sizes if N in store]
        return present, [store[N]["correlator"] for N in present]

    fig, axes = plt.subplots(len(inits), 2, figsize=(12, 4.5 * len(inits)), squeeze=False)
    for row, init in enumerate(inits):
        ax_lin, ax_log = axes[row]
        bx, by = series(results["baseline"][init])
        for ax in (ax_lin, ax_log):
            ax.plot(bx, by, "k--o", lw=2.5, zorder=5, label="baseline (L''=0)")
        for s, sample in enumerate(results["samples"]):
            sx, sy = series(sample["results"][init])
            color = cmap(s / max(n_samp - 1, 1))
            for ax in (ax_lin, ax_log):
                ax.plot(sx, sy, "-o", color=color, alpha=0.85, label=f"sample {s}")
        for ax in (ax_lin, ax_log):
            ax.set_xlabel("system size N")
            ax.set_ylabel(r"$R(N/4,\ 3N/4)$")
            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.3)
        ax_lin.set_title(f"init = |{init}>  (linear)")
        ax_log.set_title(f"init = |{init}>  (semilog)")
        ax_log.set_yscale("log")
        ax_lin.legend(fontsize=8, ncol=2)

    fig.suptitle(
        r"SWSSB Renyi-2 correlator vs system size, random parity-commuting "
        rf"$L''$ ($\epsilon={results['config']['epsilon']}$)"
    )
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "renyi2_swssb.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profile(results: dict) -> str:
    """Plot the full R(i, r) profile at N = PROFILE_SIZE, one panel per initial state."""
    plt = _mpl()
    N = results["config"]["profile_size"]
    inits = results["config"]["initial_states"]
    n_samp = len(results["samples"])
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, len(inits), figsize=(6.5 * len(inits), 5), squeeze=False)
    for col, init in enumerate(inits):
        ax = axes[0][col]
        if N not in results["baseline"][init]:
            continue
        base = results["baseline"][init][N]["profile"]
        ax.plot([r for r, _ in base], [v for _, v in base],
                "k--o", lw=2.5, zorder=5, label="baseline (L''=0)")
        for s, sample in enumerate(results["samples"]):
            if N not in sample["results"][init]:
                continue
            prof = sample["results"][init][N]["profile"]
            ax.plot([r for r, _ in prof], [v for _, v in prof], "-o",
                    color=cmap(s / max(n_samp - 1, 1)), alpha=0.85, label=f"sample {s}")
        i = results["baseline"][init][N]["i"]
        ax.set_xlabel(f"site r   (reference site i = {i})")
        ax.set_ylabel(r"$R(i,\ r)$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"N = {N},  init = |{init}>")
        if col == 0:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle(rf"Full Renyi-2 profile at $N={N}$ ($\epsilon={results['config']['epsilon']}$)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "renyi2_swssb_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_convergence(convergence: dict) -> str:
    """Plot the bond-dimension sweep: R vs chi, and the profile at each chi."""
    plt = _mpl()
    cfg = convergence["config"]
    chis = sorted(convergence["results"])
    R = [convergence["results"][c]["correlator"] for c in chis]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    ax0.plot(chis, R, "-o", color="crimson")
    ax0.axhline(R[-1], ls=":", color="grey", label=rf"$\chi={chis[-1]}$ value")
    ax0.set_xlabel(r"bond dimension $\chi$")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax0.set_xticks(chis)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=9)
    ax0.set_title(f"convergence at N={cfg['conv_N']}, init=|{cfg['conv_init']}>")

    cmap = plt.get_cmap("plasma")
    for k, c in enumerate(chis):
        prof = convergence["results"][c]["profile"]
        ax1.plot([r for r, _ in prof], [v for _, v in prof], "-o",
                 color=cmap(k / max(len(chis) - 1, 1)), label=rf"$\chi={c}$")
    ax1.set_xlabel("site r")
    ax1.set_ylabel(r"$R(i,\ r)$")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title("profile vs bond dimension")

    fig.suptitle(rf"Bond-dimension convergence, sample {cfg['conv_sample']} "
                 rf"($\epsilon={cfg['epsilon']}$)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "renyi2_swssb_chi.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    results, convergence = run_experiment()

    # Pickle before plotting: the data is the deliverable, a plotting failure
    # must never cost the simulation output.
    pkl, conv_pkl = save_results(results, convergence)
    print(f"\nSaved results     -> {pkl}", flush=True)
    print(f"Saved convergence -> {conv_pkl}", flush=True)

    for name, fn, arg in (
        ("plot", plot_results, results),
        ("profile", plot_profile, results),
        ("chi plot", plot_convergence, convergence),
    ):
        try:
            print(f"Saved {name:<13s} -> {fn(arg)}", flush=True)
        except Exception:
            print(f"!! {name} failed (data is still saved):\n{traceback.format_exc()}",
                  flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
