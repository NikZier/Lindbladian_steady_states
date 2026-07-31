"""Is the converged infinite-system rho a genuine density matrix?

What is actually being tested
-----------------------------
rho must be Hermitian and positive semidefinite. Neither is imposed anywhere
in the vectorized iTEBD: the state is an MPS over vec(rho), and truncation is
free to move it off the physical manifold. So this has to be measured.

Measure the REDUCED DENSITY MATRIX, not a tiled state
-----------------------------------------------------
An earlier probe this session tiled the iMPS into a finite open-boundary MPS
by clamping the edge bonds to index 0, and got nonsense: <n> = -3.9e-02,
purity 1.87-3.01, max eigenvalue 1.35-1.70, all worsening with window size.
That is a defect of the TILING, not evidence about the state. Clamping a bond
discards the environment instead of tracing it out, which is not a partial
trace and has no reason to give anything positive.

The correct object is the true n-site reduced density matrix of the infinite
chain. Because the iMPS carries vec(rho) as its physical index, tracing out a
site means contracting that index with vec(I) -- a SINGLE-layer contraction,
not the usual doubled transfer matrix. Define

    M[l,r] = sum_s Theta[l,s,r] * vec(I)[s]        (chi x chi)

Then the left/right environments of a window are the leading left/right
eigenvectors of the unit-cell product M_A M_B, and

    rho_window[s_1..s_n] = v_L^T Theta_{s_1} ... Theta_{s_n} v_R

is exactly Tr_env[rho], normalized by its own trace. That is a genuine
partial trace of the infinite state, so it is positive semidefinite if and
only if the state is.

Validation (asserted, not assumed): the same routine applied to the
bond-dimension-1 product states |0...0> and |0101...> must return exactly the
corresponding pure projector. Those have a known answer, so they test the
index convention -- which is where the bugs have been all session.

History: this used to be run-dependent, and the cause was a bug
---------------------------------------------------------------
Earlier passes of this script gave a different answer each time -- one run
found "nine samples clean, sample 0 not a density matrix", the next swapped
sample 0 and sample 6, and a 30-run pass reported only 20/30 landing on a
physical state. That was not physics.

`leading_eigenpairs` called scipy's eigs() WITHOUT an explicit start vector,
so ARPACK generated its own from an internal Fortran RNG that np.random.seed()
cannot reach and whose state persists within a process across calls. Under
multiprocessing the result therefore depended on how many jobs a worker had
already handled -- i.e. on pool scheduling. Since canonicalize() truncates, a
different eigenvector on a near-degenerate transfer spectrum gives a different
truncation and the evolution forks.

Diagnosed by tracking purity THROUGH a run rather than only at the end: the
bad runs did not drift, they JUMPED discretely (sample 4, purity by thirds
0.807812 -> 1.246957 -> 1.246957) and were stationary on both sides of the
jump. R was completely unaffected -- 1.304313e-03 to seven digits on every
run, physical or not, with -0.09% drift in all cases. So no correlator result
ever depended on this; only absolute quantities did.

Fixed by passing a fixed deterministic v0 (imps.ARPACK_V0_SEED), with
regression tests in tests/test_imps.py::TestArpackDeterminism.

ATTEMPTS is therefore now a DETERMINISM CHECK, not a search: the repeats run
through the same mp.Pool that exposed the bug, and must agree exactly. If
they ever disagree again, the start vector has stopped being deterministic.

Consequence for the factor of 2: R is identical on physical and unphysical
states to seven digits, so landing off the physical manifold was never the
cause of R_iTEBD = 1.9875 x the finite-N law.

Local-vec convention: rho -> M rho N^dag is kron(M, conj(N)) on a site, so the
physical index is s = 2*ket + bra and vec(I) = [1,0,0,1].

Outputs (experiments/results/):
    imps_positivity_hermiticity.png
    imps_positivity_hermiticity.pkl
"""

import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import imps_eps_init_grids as grids
import imps_swssb_infinite as ex_inf
import renyi2_swssb as ex
from lindblad_mps import imps, iobservables, itebd, models

SAMPLES = list(range(10))
CHI = 128
INIT = "neel"
WINDOWS = [2, 4, 6, 8]
EPSILON = 0.20
ATTEMPTS = 2       # determinism check, not a search -- see module docstring
VEC_I = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)  # vec(I_2), row-major


def a_form(state: "imps.iMPS") -> dict:
    """Left-weighted (A-form) unit-cell tensors: A_A = Lambda_B Gamma_A, etc."""
    return {"A": imps.left_weighted(state.Gamma["A"], state.Lambda["B"]),
            "B": imps.left_weighted(state.Gamma["B"], state.Lambda["A"])}


def trace_map(theta: np.ndarray) -> np.ndarray:
    """M[l,r] = sum_s theta[l,s,r] vec(I)[s] -- one site's contribution to Tr[rho]."""
    return np.tensordot(theta, VEC_I, axes=([1], [0]))


def environments(Theta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Leading left/right eigenvectors of the unit-cell trace map M_A M_B.

    These are the partial trace of the infinite tails to either side of the
    window. Solved densely: M is chi x chi with chi <= ~30 here.
    """
    M = trace_map(Theta["A"]) @ trace_map(Theta["B"])
    wr, vr = np.linalg.eig(M)
    wl, vl = np.linalg.eig(M.T)
    v_R = vr[:, np.argmax(np.abs(wr))]
    v_L = vl[:, np.argmax(np.abs(wl))]
    return v_L, v_R


def reduced_density_matrix(state: "imps.iMPS", n_sites: int) -> np.ndarray:
    """True n-site reduced density matrix of the infinite chain, trace-normalized.

    Input: a canonicalize()'d iMPS over vec(rho); n_sites (even).
    Output: (2**n_sites, 2**n_sites) complex array with unit trace.
    """
    Theta = a_form(state)
    v_L, v_R = environments(Theta)

    # Contract the window, carrying the physical (vectorized) legs.
    block = v_L.reshape(1, -1)                       # (phys_so_far=1, chi)
    for k in range(n_sites):
        th = Theta["A" if k % 2 == 0 else "B"]       # (chi_l, 4, chi_r)
        block = np.tensordot(block, th, axes=([-1], [0]))   # (..., 4, chi_r)
        block = block.reshape(-1, block.shape[-1])
    rho = (block @ v_R).reshape((2, 2) * n_sites)    # (ket_1,bra_1, ket_2,bra_2, ...)

    # (ket_1,bra_1,...) -> (ket_1..ket_n, bra_1..bra_n)
    perm = list(range(0, 2 * n_sites, 2)) + list(range(1, 2 * n_sites, 2))
    rho = np.transpose(rho, perm).reshape(2 ** n_sites, 2 ** n_sites)

    tr = np.trace(rho)
    if abs(tr) > 1e-300:
        rho = rho / tr
    return rho


def diagnose(rho: np.ndarray) -> dict:
    """Hermiticity and positivity of a trace-normalized rho."""
    herm = np.linalg.norm(rho - rho.conj().T) / max(np.linalg.norm(rho), 1e-300)
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    neg = w[w < 0].sum()
    return {
        "hermiticity_error": float(herm),
        "min_eig": float(w.min()),
        "max_eig": float(w.max()),
        "negative_weight": float(abs(neg) / np.abs(w).sum()),
        "eigenvalues": w,
        "purity": float(np.trace(rho.conj().T @ rho).real),
    }


def _validate() -> None:
    """The routine must reproduce known product states exactly."""
    K0 = np.array([1, 0], dtype=complex)
    K1 = np.array([0, 1], dtype=complex)
    for name, kets in (("zero", (K0, K0)), ("neel", (K0, K1))):
        st = imps.iMPS.pure_product_state(*kets)
        rho = reduced_density_matrix(st, 4)
        expect = np.array([1.0], dtype=complex)
        for k in range(4):
            v = kets[k % 2]
            expect = np.kron(expect, np.outer(v, v.conj()))
        err = np.linalg.norm(rho - expect)
        assert err < 1e-10, f"{name}: RDM wrong by {err:.3e} -- index convention is off"
        d = diagnose(rho)
        assert d["min_eig"] > -1e-12 and d["hermiticity_error"] < 1e-12, name
    print("  validation OK: |0...0> and |0101...> RDMs reproduced exactly", flush=True)


def run_one(job: dict) -> dict:
    """Converge one sample at one seed and diagnose its reduced density matrices.

    The seed matters: it sets ARPACK's start vector inside canonicalize(),
    which is what selects the member of the degenerate stationary manifold.
    """
    t0 = time.perf_counter()
    np.random.seed(job["seed"])
    L2 = ex.build_L2_terms(job["L_pp"])
    state, _ = itebd.find_steady_state_infinite(
        H2_terms=[], H1_terms=[], L2_terms=L2, L1_terms=[],
        dt_schedule=job["schedule"], steps_per_dt=grids.STEPS_PER_DT,
        chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=grids.CANONICALIZE_EVERY,
        initial_state=ex_inf.build_initial_state(INIT),
    )
    state.canonicalize(chi_max=CHI, cutoff=ex.CUTOFF)
    out = {"sample": job["sample"], "attempt": job["attempt"], "windows": {},
           "bond": max(state.bond_dims.values()),
           "R100": dict(iobservables.correlator_profile(state, models.X, r_max=100))[100],
           "seconds": time.perf_counter() - t0}
    for n in WINDOWS:
        out["windows"][n] = diagnose(reduced_density_matrix(state, n))
    out["is_physical"] = bool(out["windows"][max(WINDOWS)]["min_eig"] > -1e-10)
    return out


def main() -> None:
    print("validating the RDM contraction against exact product states...", flush=True)
    _validate()

    original = ex.EPSILON
    try:
        ex.EPSILON = EPSILON
        samples = ex.draw_samples()
    finally:
        ex.EPSILON = original

    # Sample 8 needs the long schedule -- it is the only sample that was still
    # relaxing at 2940 units (see imps_eps_init_grids.py).
    jobs = [{"sample": s, "attempt": a, "seed": 7919 * s + 101 * a + 1,
             "L_pp": samples[s]["L_pp"],
             "schedule": grids.SCHEDULE_S8LONG if s == 8 else grids.SCHEDULE_BASE}
            for s in SAMPLES for a in range(ATTEMPTS)]
    jobs.sort(key=lambda j: -sum(j["schedule"]))

    all_runs: dict[int, list] = {s: [] for s in SAMPLES}
    t0 = time.perf_counter()
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            all_runs[out["sample"]].append(out)
            d = out["windows"][max(WINDOWS)]
            print(f"  s{out['sample']} attempt{out['attempt']}: "
                  f"{'PHYSICAL' if out['is_physical'] else 'not PSD '}  "
                  f"min_eig={d['min_eig']:+.3e}  purity={d['purity']:.6f}  "
                  f"R(100)={out['R100']:.6e}  "
                  f"[elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    # Keep the physical member where one was found; otherwise keep the best
    # attempt and let the figure show that it failed.
    results, stats = {}, {}
    for s in SAMPLES:
        runs = sorted(all_runs[s], key=lambda o: o["attempt"])
        physical = [o for o in runs if o["is_physical"]]
        results[s] = physical[0] if physical else max(
            runs, key=lambda o: o["windows"][max(WINDOWS)]["min_eig"])
        rs = [o["R100"] for o in runs]
        stats[s] = {"n_physical": len(physical), "n_attempts": len(runs),
                    "R_values": rs,
                    "R_spread": (np.std(rs) / np.mean(rs)) if rs else float("nan"),
                    "purities": [o["windows"][max(WINDOWS)]["purity"] for o in runs]}

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(ex.RESULTS_DIR, "imps_positivity_hermiticity.pkl"), "wb") as f:
        pickle.dump({"results": results, "stats": stats, "all_runs": all_runs,
                     "windows": WINDOWS, "epsilon": EPSILON, "chi": CHI,
                     "init": INIT, "attempts": ATTEMPTS}, f)

    summarize(results)
    summarize_stats(stats)
    print(f"Saved -> {plot(results, stats)}", flush=True)


def summarize_stats(stats: dict) -> None:
    """Determinism check: repeats through the same mp.Pool must agree exactly."""
    print(f"\n{'='*92}\nDeterminism check ({ATTEMPTS} identical runs per sample, "
          f"through the pool that exposed the bug)\n{'='*92}")
    print(f"{'s':>2} {'physical/runs':>14} {'R(100) rel. spread':>20} "
          f"{'purity across runs':>34}")
    bad_R, bad_P = [], []
    for s in sorted(stats):
        st = stats[s]
        ps = st["purities"]
        if st["R_spread"] > 1e-12:
            bad_R.append(s)
        if max(ps) - min(ps) > 1e-9:
            bad_P.append(s)
        print(f"{s:>2} {st['n_physical']:>7}/{st['n_attempts']:<6} "
              f"{st['R_spread']:>20.2e}   "
              f"{'  '.join(f'{p:.6f}' for p in ps):>32}")

    tot = sum(st["n_physical"] for st in stats.values())
    att = sum(st["n_attempts"] for st in stats.values())
    print(f"\n{tot}/{att} runs physical.")
    if bad_R or bad_P:
        print(f"!! NON-DETERMINISTIC -- R differs for {bad_R}, purity differs for "
              f"{bad_P}.\n   The ARPACK start vector has stopped being fixed; see "
              f"imps.leading_eigenpairs.")
    else:
        print("All repeats agree exactly, in both R and purity: the pipeline is\n"
              "deterministic. Absolute quantities can now be quoted from a single run.")
    n_unphysical = [s for s, st in stats.items() if st["n_physical"] == 0]
    if n_unphysical:
        print(f"Samples with NO physical run: {n_unphysical} -- for these the converged\n"
              "state is genuinely not a density matrix, which is now a reproducible\n"
              "statement rather than a scheduling accident.")


def summarize(results: dict) -> None:
    n = max(WINDOWS)
    print(f"\n{'='*92}\nReduced density matrix of the infinite chain, {n}-site window "
          f"(eps={EPSILON}, chi={CHI})\n{'='*92}")
    print(f"{'s':>2} {'min eigenvalue':>16} {'max eigenvalue':>16} "
          f"{'neg. weight':>13} {'||rho-rho^dag||':>16} {'purity':>9}")
    for s in sorted(results):
        d = results[s]["windows"][n]
        print(f"{s:>2} {d['min_eig']:>+16.3e} {d['max_eig']:>16.6f} "
              f"{d['negative_weight']:>13.3e} {d['hermiticity_error']:>16.3e} "
              f"{d['purity']:>9.6f}")
    worst_neg = min(results[s]["windows"][n]["min_eig"] for s in results)
    worst_herm = max(results[s]["windows"][n]["hermiticity_error"] for s in results)
    print(f"\nworst min-eigenvalue over all samples: {worst_neg:+.3e}")
    print(f"worst hermiticity error               : {worst_herm:.3e}")
    print("Both should be at machine-precision level for a genuine density matrix.")


def plot(results: dict, stats: dict | None = None) -> str:
    """Four panels for the physical member, plus the run-to-run caveat stated on
    the figure rather than left to the caller to remember."""
    plt = ex._mpl()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    cmap = plt.get_cmap("tab10")
    n_big = max(WINDOWS)

    def style(s):
        bad = results[s]["windows"][n_big]["min_eig"] < -1e-10
        return dict(color="black" if bad else cmap(s % 10),
                    lw=2.6 if bad else 1.2, zorder=5 if bad else 2,
                    label=f"s{s}" + (" (NOT PSD)" if bad else ""))

    # Signed spectrum on a symlog axis, so negative eigenvalues are visible
    # rather than clipped away by a log scale.
    ax = axes[0][0]
    for s in sorted(results):
        w = np.sort(results[s]["windows"][n_big]["eigenvalues"])[::-1]
        ax.plot(np.arange(1, len(w) + 1), w, ".-", ms=3, **style(s))
    ax.axhline(0, color="grey", lw=1.0)
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xscale("log")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel(r"eigenvalue of $\rho$  (symlog)")
    ax.set_title(rf"Signed spectrum of the {n_big}-site reduced $\rho$")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=2)

    ax = axes[0][1]
    for s in sorted(results):
        ax.plot(WINDOWS, [results[s]["windows"][n]["min_eig"] for n in WINDOWS],
                "o-", ms=5, **style(s))
    ax.axhline(0, color="grey", lw=1.2)
    ax.set_yscale("symlog", linthresh=1e-15)
    ax.set_xlabel("window size (sites)")
    ax.set_ylabel(r"$\lambda_{\min}(\rho)$  (signed, symlog)")
    ax.set_title(r"Smallest eigenvalue — negative means not a physical state")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2)

    ax = axes[1][0]
    for s in sorted(results):
        ax.semilogy(WINDOWS, [max(results[s]["windows"][n]["hermiticity_error"], 1e-20)
                              for n in WINDOWS], "o-", ms=5, **style(s))
    ax.axhline(1e-16, color="black", ls="--", lw=1.3, label="machine precision")
    ax.set_xlabel("window size (sites)")
    ax.set_ylabel(r"$\|\rho-\rho^\dagger\|_F / \|\rho\|_F$")
    ax.set_title(r"Hermiticity violation")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2)

    # Purity is the sharpest single test: for ANY density matrix it must be
    # <= 1, and for a mixed state it must fall as the window grows.
    ax = axes[1][1]
    for s in sorted(results):
        ax.plot(WINDOWS, [results[s]["windows"][n]["purity"] for n in WINDOWS],
                "o-", ms=5, **style(s))
    ax.axhline(1.0, color="red", ls="--", lw=1.6, label=r"$\mathrm{Tr}\rho^2=1$ (upper bound)")
    ax.set_xlabel("window size (sites)")
    ax.set_ylabel(r"purity $\mathrm{Tr}[\rho^2]$")
    ax.set_title(r"Purity — must be $\leq 1$ and fall with window size")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)

    n_bad = sum(1 for s in results if results[s]["windows"][n_big]["min_eig"] < -1e-10)
    n_ok = len(results) - n_bad
    bad_ids = sorted(s for s in results
                     if results[s]["windows"][n_big]["min_eig"] < -1e-10)
    fig.suptitle(
        rf"The infinite-system steady state is Hermitian and positive semidefinite "
        rf"($\epsilon={EPSILON}$, $\chi={CHI}$, $|neel\rangle$)" if not n_bad else
        rf"Positivity of the infinite-system $\rho$ ($\epsilon={EPSILON}$, $\chi={CHI}$, "
        rf"$|neel\rangle$):  {n_ok} of {len(results)} samples are genuine density "
        rf"matrices;  samples {', '.join(map(str, bad_ids))} are not",
        fontsize=13)
    if stats:
        tot = sum(st["n_physical"] for st in stats.values())
        att = sum(st["n_attempts"] for st in stats.values())
        deterministic = all(st["R_spread"] <= 1e-12 for st in stats.values()) and all(
            max(st["purities"]) - min(st["purities"]) <= 1e-9 for st in stats.values())
        fig.text(0.5, 0.005,
                 rf"{tot} of {att} runs physical ({ATTEMPTS} repeats per sample). "
                 + ("Repeats agree exactly — the pipeline is deterministic after fixing "
                    r"ARPACK's start vector, so these are reproducible statements, not "
                    "scheduling accidents."
                    if deterministic else
                    r"!! Repeats DISAGREE — the ARPACK start vector is no longer fixed."),
                 ha="center", fontsize=8.5, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    p = os.path.join(ex.RESULTS_DIR, "imps_positivity_hermiticity.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def replot() -> str:
    """Redraw from the saved pickle without re-running any TEBD."""
    with open(os.path.join(ex.RESULTS_DIR, "imps_positivity_hermiticity.pkl"), "rb") as f:
        d = pickle.load(f)
    summarize(d["results"])
    if "stats" in d:
        summarize_stats(d["stats"])
    return plot(d["results"], d.get("stats"))


if __name__ == "__main__":
    mp.freeze_support()
    main()
