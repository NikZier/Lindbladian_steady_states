"""Renyi-2 (SWSSB) correlator of the parity-symmetric dissipative chain under
random parity-commuting perturbations, as a function of system size, from two
strongly-symmetric initial states.

Model (purely dissipative, H = 0), bond jump operators applied uniformly to
every nearest-neighbour bond:

    L   = X_a X_{a+1} (1 - Z_a Z_{a+1})           rate 1
    L'  = X_a X_{a+1} (1 - Z_a)(1 - Z_{a+1})      rate 1
    L'' = random, [L'', Z(x)Z] = 0, ||L''|| = epsilon = 0.1   rate 1

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

For each initial state, random L'' and size N in {4, 8, 16} we find the steady
state by imaginary-time TEBD and evaluate the Renyi-2 correlator

    R(i, j) = Tr[rho A rho^dag A^dag] / Tr[rho^dag rho],   A = X_i X_j

at i = N//4, j = 3N//4 (separation N/2, away from the open boundaries).

Outputs (experiments/results/):
    renyi2_swssb.pkl -- config, per-sample L'' descriptions (matrix, Pauli
        coefficients, operator norm, seed) and correlator results for both
        initial states, plus the L''=0 baseline. The Pauli coefficients let
        each L'' be reconstructed exactly and extended to larger N later.
    renyi2_swssb.png -- R vs N, one panel row per initial state, one line per
        L'' sample, baseline overlaid.
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lindblad_mps import exact, models, observables, tebd, vectorize
from lindblad_mps import mps as mps_module

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
EPSILON = 0.1
SIZES = [4, 8, 16]
N_SAMPLES = 8
BASE_SEED = 20260726
INITIAL_STATES = ["zero", "neel"]  # both must share the parity sector (asserted)

CHI_MAX = 32
CUTOFF = 1e-10
DT_SCHEDULE = [0.1, 0.05, 0.02, 0.01, 0.005]
STEPS_PER_DT = 300
RECANON_EVERY = 10

CONVERGENCE_TOL = 1e-6  # 1 - |<rho(t)|rho(t+dt)>| below this counts as converged

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

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


def build_L2_terms(L_pp: np.ndarray | None) -> list[tuple[np.ndarray, float]]:
    """Assemble the list of (bond jump operator, rate) terms for the model.

    Input: L_pp, the perturbation operator L'' (4x4), or None for the
        unperturbed baseline.
    Output: list of (op, rate) pairs: the two baseline jumps plus, if given,
        L'', all at rate 1.0.
    """
    L, L_prime = models.baseline_jump_operators()
    terms = [(L, 1.0), (L_prime, 1.0)]
    if L_pp is not None:
        terms.append((L_pp, 1.0))
    return terms


def run_steady_state_correlator(
    L2_terms: list[tuple[np.ndarray, float]], N: int, init_name: str
) -> dict:
    """Find the TEBD steady state at size N (from init_name) and measure R(i,j).

    Runs tebd.find_steady_state with the module-level truncation / schedule
    settings from the chosen strongly-symmetric initial state, then measures
    R(i, j) at i = N//4, j = 3N//4 with order parameter O = X, and records the
    full profile R(i, r) for r > i.

    Input:
        L2_terms: bond jump terms (op, rate) (H and single-site terms empty).
        N: number of sites.
        init_name: 'zero' or 'neel'.
    Output: dict with keys 'N', 'init', 'i', 'j', 'correlator', 'profile'
        (list of (r, R(i, r))), 'final_overlap', 'max_discarded_weight',
        'converged', 'state' (the MPS).
    """
    state, history = tebd.find_steady_state(
        H2_terms=[],
        H1_terms=[],
        L2_terms=L2_terms,
        L1_terms=[],
        N=N,
        dt_schedule=DT_SCHEDULE,
        steps_per_dt=STEPS_PER_DT,
        chi_max=CHI_MAX,
        cutoff=CUTOFF,
        recanonicalize_every=RECANON_EVERY,
        initial_state=build_initial_state(init_name, N),
    )

    i, j = correlator_sites(N)
    R = observables.renyi2_correlator_mps(state, models.X, i, j)
    profile = [
        (r, observables.renyi2_correlator_mps(state, models.X, i, r))
        for r in range(i + 1, N)
    ]

    final_overlap = history["overlap"][-1] if history["overlap"] else float("nan")
    return {
        "N": N,
        "init": init_name,
        "i": i,
        "j": j,
        "correlator": R,
        "profile": profile,
        "final_overlap": final_overlap,
        "max_discarded_weight": max(history["discarded_weight"], default=0.0),
        "converged": (1.0 - final_overlap) < CONVERGENCE_TOL,
        "state": state,
    }


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


def _strip_state(res: dict) -> dict:
    """Drop the MPS object from a run result (keeps the pickle small)."""
    return {k: v for k, v in res.items() if k != "state"}


def run_experiment() -> dict:
    """Run the full study: baseline + N_SAMPLES random L'' across all sizes,
    for every initial state in INITIAL_STATES.

    Output: a results dict ready to pickle (no live MPS objects).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Guard: all initial states must share the same parity sector at every size.
    for N in SIZES:
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

    results = {
        "config": {
            "epsilon": EPSILON,
            "sizes": SIZES,
            "n_samples": N_SAMPLES,
            "base_seed": BASE_SEED,
            "initial_states": INITIAL_STATES,
            "chi_max": CHI_MAX,
            "cutoff": CUTOFF,
            "dt_schedule": DT_SCHEDULE,
            "steps_per_dt": STEPS_PER_DT,
            "site_rule": "i = N//4, j = 3N//4",
            "order_parameter": "X",
            "model": "L=XX(1-ZZ), L'=XX(1-Za)(1-Zb), L''=random parity-commuting, rates=1",
            "neel_note": "|0101...>, same (+,+) parity sector as |0...0> for N in {4,8,16}",
        },
        "validation_N4": validation,
        "baseline": {init: {} for init in INITIAL_STATES},
        "samples": [],
    }

    # Baseline (no L'') for each initial state.
    print("Baseline (L''=0):", flush=True)
    for init in INITIAL_STATES:
        for N in SIZES:
            res = run_steady_state_correlator(build_L2_terms(None), N, init)
            results["baseline"][init][N] = _strip_state(res)
            print(
                f"  [{init}] N={N:2d}  R({res['i']},{res['j']})={res['correlator']:.6e}  "
                f"conv={res['converged']}  maxdisc={res['max_discarded_weight']:.1e}",
                flush=True,
            )

    # Random L'' samples.
    for s in range(N_SAMPLES):
        seed = BASE_SEED + 1 + s
        rng = np.random.default_rng(seed)
        L_pp = models.random_zz_commuting_operator(EPSILON, rng)
        description = models.describe_operator(L_pp, EPSILON, seed=seed)
        assert models.commutes_with_zz(L_pp), "sampled L'' must commute with ZZ"

        print(
            f"Sample {s} (seed={seed}, ||L''||={description['operator_norm']:.3f}):",
            flush=True,
        )
        sample_results = {init: {} for init in INITIAL_STATES}
        for init in INITIAL_STATES:
            for N in SIZES:
                res = run_steady_state_correlator(build_L2_terms(L_pp), N, init)
                sample_results[init][N] = _strip_state(res)
                print(
                    f"  [{init}] N={N:2d}  R({res['i']},{res['j']})={res['correlator']:.6e}"
                    f"  conv={res['converged']}  maxdisc={res['max_discarded_weight']:.1e}",
                    flush=True,
                )
        results["samples"].append({"description": description, "results": sample_results})

    return results


def save_results(results: dict) -> str:
    """Pickle the results dict to experiments/results/renyi2_swssb.pkl."""
    path = os.path.join(RESULTS_DIR, "renyi2_swssb.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def plot_results(results: dict) -> str:
    """Plot R vs N: one row per initial state, columns linear and semilog."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = results["config"]["sizes"]
    inits = results["config"]["initial_states"]
    n_samp = len(results["samples"])
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(len(inits), 2, figsize=(12, 4.5 * len(inits)), squeeze=False)

    for row, init in enumerate(inits):
        ax_lin, ax_log = axes[row]
        baseline_R = [results["baseline"][init][N]["correlator"] for N in sizes]
        for ax in (ax_lin, ax_log):
            ax.plot(sizes, baseline_R, "k--o", lw=2.5, zorder=5, label="baseline (L''=0)")

        for s, sample in enumerate(results["samples"]):
            R_vals = [sample["results"][init][N]["correlator"] for N in sizes]
            color = cmap(s / max(n_samp - 1, 1))
            for ax in (ax_lin, ax_log):
                ax.plot(sizes, R_vals, "-o", color=color, alpha=0.85, label=f"sample {s}")

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


def main() -> None:
    results = run_experiment()
    pkl = save_results(results)
    png = plot_results(results)
    print(f"\nSaved results -> {pkl}", flush=True)
    print(f"Saved plot    -> {png}", flush=True)


if __name__ == "__main__":
    main()
