"""SWSSB Renyi-2 correlator of the *driven* classical circuit: biased hopping
plus pair annihilation, perturbed by random parity-commuting L''.

The model
---------
A purely classical two-site circuit, translated to Lindblad form by
models.classical_drift_annihilation_jump_operators (see its docstring for why
each classical transition becomes its own jump operator):

    |10> -> p |01> + (1-p) |10>      L_R = sqrt(p)     |01><10|
    |01> -> p |01> + (1-p) |10>      L_L = sqrt(1-p)   |10><01|
    |00> -> |00>                     (inert)
    |11> -> |00>                     L_A =             |00><11|

    L''  = random, [L'', Z(x)Z] = 0, ||L''|| = epsilon = 0.2

all at rate 1. For p > 1/2 charge drifts right, so unlike the diffusive SWSSB
baseline this chain is *driven*: it has no left-right symmetry.

What is shared with the earlier study, and what is not
-----------------------------------------------------
Shared, deliberately, so the two are directly comparable: the strong Z_2
symmetry P = Z_1...Z_N (particle-number parity -- annihilation removes
particles two at a time), the dark vacuum |0...0>, the diagnostic
R(i,j) = Tr[rho A rho^dag A^dag]/Tr[rho^dag rho] with A = X_i X_j, the two
initial states, and the *identical ten L'' samples* (ex.draw_samples(), same
seeds), so a sample-by-sample comparison against the earlier model is a
comparison of models rather than of draws.

Not shared: the baseline. The earlier baseline L = XX(1 - ZZ) is
2(|01><10| + |10><01|) -- also classical on the diagonal, but bundled into one
jump operator, which adds interference terms on coherences. So this model is
NOT that one at p = 1/2, and its rates are ~16x smaller (hop 0.8/0.2 and
annihilation 1, against hop 4 and annihilation 16), which makes epsilon = 0.2 a
relatively stronger perturbation here. R is correspondingly ~100x larger. The
normalization-free comparison is R/q with q = |<11|L''|00>|^2, the
pair-creation matrix element that drove the entire signal in the earlier model.

Convergence
-----------
Calibrated at N=4 against the exact steady state of the (+,+) parity sector,
which this script recomputes and reports (`exact_N4`): N=4 at chi=128 is
truncation-free (the middle-bond bound is 4^2 = 16), so the comparison isolates
Trotter error plus unfinished relaxation. The study-wide schedule already
reproduces it to six digits from both starts, so it is reused unchanged. The
chi sweep at N=SWEEP_N addresses Trap 1 of CLAUDE.md -- a chi-limited run
fabricates decay -- and the zero/neel spread addresses Trap 2.

Outputs (experiments/results/):
    renyi2_drift_annihilation.pkl   grid, control, chi sweep, exact N=4
    renyi2_drift_annihilation.png   R vs N per sample + size-to-size ratio
    renyi2_drift_annihilation_profile.png   R(i, r) at N = PROFILE_SIZE
    renyi2_drift_annihilation_chi.png       the bond-dimension sweep
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lindblad_mps import exact, models, observables, vectorize
import renyi2_swssb as ex

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
P_RIGHT = 0.8            # p > 1/2: charge drifts right
HOP_RATE = 1.0
ANNIHILATION_RATE = 1.0

CHI = 128
SIZES = [4, 8, 12, 16, 20]
INITS = ["zero", "neel"]
PROFILE_SIZE = 20

# Trap 1 of CLAUDE.md: a chi-limited run fabricates a convincing exponential
# decay in both separation and system size. The sweep is run at the largest
# size, where truncation is most strained, on the sample with the smallest
# pair-creation amplitude (slowest to relax, so the least forgiving).
CHI_SWEEP = [32, 64]     # CHI itself comes from the main grid
SWEEP_N = 20
SWEEP_SAMPLE = 8
SWEEP_INIT = "zero"

KIND = "drift"           # cache-key namespace, keeps this study off the old one

# Recorded in run_config so a cache entry can never alias onto the other
# model's run under the same (kind, label, init, N, chi) key.
MODEL = {
    "name": "classical_drift_annihilation",
    "p": P_RIGHT,
    "hop_rate": HOP_RATE,
    "annihilation_rate": ANNIHILATION_RATE,
}


def baseline_ops() -> list[np.ndarray]:
    """The three bond jump operators of the unperturbed classical circuit."""
    return models.classical_drift_annihilation_jump_operators(
        P_RIGHT, HOP_RATE, ANNIHILATION_RATE
    )


def creation_amplitude(L_pp: np.ndarray) -> float:
    """q = |<11|L''|00>|^2, the pair-creation rate of the perturbation.

    In the earlier model this single number fixed the correlator exactly
    (R = q/8 at finite N). It is the natural normalization here too, because
    it is what destabilizes the dark vacuum, and it is independent of the
    baseline rate convention.
    """
    return float(abs(L_pp[0b11, 0b00]) ** 2)


# ---------------------------------------------------------------------------
# Exact reference at N = 4
# ---------------------------------------------------------------------------
def sector_steady_state(
    L2_terms: list[tuple[np.ndarray, float]], N: int, charge: int = +1
) -> tuple[np.ndarray, float]:
    """Exact steady state inside one strong-symmetry sector, by dense null space.

    exact.steady_state cannot be used here: the strong Z_2 symmetry makes the
    zero eigenvalue of the full Liouvillian degenerate (one steady state per
    sector), and it refuses on exactly that condition. Restricting to the
    (charge, charge) block -- vec indices |a><b| with both parities equal to
    `charge` -- makes the null vector unique again, and it is the sector the
    TEBD runs actually live in.

    Input:
        L2_terms: bond jump terms (op, rate), as passed to tebd.
        N: number of sites (dense: keep to N <= 6).
        charge: the parity eigenvalue of the sector (+1 for both starts here).
    Output: (rho, gap) -- the normalized sector steady state, and the smallest
        nonzero singular value of the restricted generator (a nonzero gap is
        what makes the state unique).
    """
    dim = 2**N
    jump_ops = exact.build_jump_operators([(op, 1.0) for op, _ in L2_terms], [], N)
    gen = vectorize.liouvillian_generator([], jump_ops, d=dim)

    bits = np.array([[(k >> b) & 1 for b in range(N)] for k in range(dim)])
    keep = np.flatnonzero(np.where(bits.sum(axis=1) % 2, -1, 1) == charge)
    idx = (keep[:, None] * dim + keep[None, :]).reshape(-1)  # vec(|a><b|) = dim*a+b

    # A strong symmetry means the generator does not connect sectors at all.
    # Assert it rather than assume it: it is the premise of the whole study.
    outside = np.setdiff1d(np.arange(dim * dim), idx)
    assert np.abs(gen[np.ix_(outside, idx)]).max() < 1e-12, "sector leakage"

    _, S, Vh = np.linalg.svd(gen[np.ix_(idx, idx)])
    v = np.zeros(dim * dim, dtype=complex)
    v[idx] = Vh[-1].conj()
    rho = vectorize.unvec(v, dim)
    rho = (rho + rho.conj().T) / 2
    return rho / np.trace(rho), float(S[-2])


def exact_reference(samples: list[dict], N: int = 4) -> dict:
    """Exact R at size N for every sample, plus the L''=0 control.

    Output: dict with 'N', 'i', 'j', 'R' (list over samples), 'q' (list),
        'gap' (list), 'baseline_R' and 'baseline_purity' (1.0 iff the
        unperturbed steady state is the pure dark vacuum, as it must be).
    """
    i, j = ex.correlator_sites(N)
    base = baseline_ops()
    out = {"N": N, "i": i, "j": j, "R": [], "q": [], "gap": []}
    for samp in samples:
        terms = ex.build_L2_terms(samp["L_pp"], base)
        rho, gap = sector_steady_state(terms, N)
        out["R"].append(observables.renyi2_correlator_dense(rho, models.X, i, j, N))
        out["q"].append(creation_amplitude(samp["L_pp"]))
        out["gap"].append(gap)

    rho0, _ = sector_steady_state(ex.build_L2_terms(None, base), N)
    out["baseline_R"] = observables.renyi2_correlator_dense(rho0, models.X, i, j, N)
    out["baseline_purity"] = float(np.trace(rho0 @ rho0).real)
    return out


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def build_jobs(samples: list[dict]) -> list[dict]:
    """Enumerate every run: the grid, the L''=0 control, and the chi sweep.

    Input: samples, from ex.draw_samples() (the same ten L'' as the earlier
        study, by construction).
    Output: list of job dicts for ex.run_job(), longest first.
    """
    base = baseline_ops()
    common = {"baseline": base, "model": MODEL, "chi_max": CHI}
    jobs = [
        {"kind": KIND, "label": f"sample{s}", "sample": s, "L_pp": samp["L_pp"],
         "N": N, "init": init, **common}
        for s, samp in enumerate(samples)
        for N in SIZES
        for init in INITS
    ]

    # L''=0 control. Only 'neel' is informative: 'zero' IS the dark state, so
    # it sits at exactly R = 0 and measures no truncation floor at all.
    jobs += [
        {"kind": f"{KIND}ctrl", "label": "baseline", "sample": None, "L_pp": None,
         "N": N, "init": "neel", **common}
        for N in SIZES
    ]

    jobs += [
        {"kind": f"{KIND}sweep", "label": f"sample{SWEEP_SAMPLE}",
         "sample": SWEEP_SAMPLE, "L_pp": samples[SWEEP_SAMPLE]["L_pp"],
         "N": SWEEP_N, "init": SWEEP_INIT, "baseline": base, "model": MODEL,
         "chi_max": chi}
        for chi in CHI_SWEEP
    ]

    jobs.sort(key=ex.estimated_cost, reverse=True)
    return jobs


def run_grid() -> dict:
    """Run every job on ex.N_WORKERS processes and assemble the result dict."""
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()

    # Both starts must sit in the same parity sector at every size, or the two
    # are not measuring the same steady state (Neel has an even number of ones
    # for every N in SIZES).
    for N in SIZES:
        charges = {n: ex.parity_charge(ex.basis_bits(n, N)) for n in INITS}
        assert len(set(charges.values())) == 1, f"sectors differ at N={N}: {charges}"

    print("Exact (+,+)-sector steady state at N=4 (truncation-free reference) ...",
          flush=True)
    ref = exact_reference(samples)
    print(f"  L''=0 control: R={ref['baseline_R']:.2e}, "
          f"purity={ref['baseline_purity']:.6f} (1.0 = pure dark vacuum)",
          flush=True)
    ratios = np.array(ref["R"]) / np.array(ref["q"])
    print(f"  R/q over 10 samples: mean {ratios.mean():.4f}, "
          f"spread {(ratios.max()-ratios.min())/ratios.mean()*100:.1f}%, "
          f"sector gap {min(ref['gap']):.2f}\n", flush=True)

    grid = {
        "config": {
            "model": MODEL, "epsilon": ex.EPSILON, "chi_max": CHI, "sizes": SIZES,
            "inits": INITS, "n_samples": len(samples), "base_seed": ex.BASE_SEED,
            "cutoff": ex.CUTOFF, "dt_schedule": ex.DT_SCHEDULE,
            "steps_per_dt": ex.STEPS_PER_DT,
            "recanonicalize_every": ex.RECANON_EVERY,
            "site_rule": "i = N//4, j = 3N//4", "order_parameter": "X",
            "profile_size": PROFILE_SIZE,
            "chi_sweep": CHI_SWEEP, "sweep_N": SWEEP_N,
            "sweep_sample": SWEEP_SAMPLE, "sweep_init": SWEEP_INIT,
        },
        "descriptions": [s["description"] for s in samples],
        "q": [creation_amplitude(s["L_pp"]) for s in samples],
        "exact_N4": ref,
        "results": {init: {N: [None] * len(samples) for N in SIZES} for init in INITS},
        "control_Lpp0": {},
        "chi_sweep": {},
    }

    jobs = build_jobs(samples)
    print(f"{len(jobs)} runs queued on {ex.N_WORKERS} workers: "
          f"{len(samples)} samples x N in {SIZES} x {INITS} at chi={CHI}, "
          f"plus {len(SIZES)} controls and {len(CHI_SWEEP)} chi-sweep points.",
          flush=True)
    print("Watch for '!' (still drifting), 'NEG' (positivity lost) and a bond "
          "dimension at the cap.\n", flush=True)

    t0 = time.perf_counter()
    done, failures = 0, []
    with mp.Pool(processes=ex.N_WORKERS) as pool:
        for out in pool.imap_unordered(ex.run_job, jobs):
            done += 1
            tag = (f"  [{done:3d}/{len(jobs)}] {out['label']:>9s} {out['init']:>4s} "
                   f"N={out['N']:2d} chi={out['chi_max']:3d}")
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            res = out["result"]
            if out["kind"] == KIND:
                grid["results"][out["init"]][out["N"]][out["sample"]] = res
            elif out["kind"] == f"{KIND}ctrl":
                grid["control_Lpp0"][out["N"]] = res
            else:
                grid["chi_sweep"][out["chi_max"]] = res

            maxbond = max(res["final_bond_dims"])
            timing = "cached" if out["cached"] else f"{out['seconds']:.0f}s"
            print(f"{tag}  R={res['correlator']:.6e}  {ex.format_diagnostics(res)}"
                  f"bond={maxbond:3d}/{out['chi_max']}"
                  f"{'!' if maxbond >= out['chi_max'] else ' '} "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]",
                  flush=True)

    # The chi = CHI point of the sweep is the corresponding main-grid run.
    main = grid["results"][SWEEP_INIT][SWEEP_N][SWEEP_SAMPLE]
    if main is not None:
        grid["chi_sweep"][CHI] = main

    grid["failures"] = [{k: v for k, v in f.items() if k not in ("L_pp", "baseline")}
                        for f in failures]
    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t0)/60:.1f} min.", flush=True)
    for f in failures:
        print(f"  FAILED: {ex.job_key(f)}", flush=True)
    return grid


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(grid: dict) -> None:
    """Print R vs N per sample, the R/q law, and every convergence flag."""
    n = grid["config"]["n_samples"]
    q = grid["q"]

    for init in INITS:
        print(f"\n=== init |{init}>,  chi={CHI},  p={P_RIGHT} ===")
        print(f"{'sample':>6} {'q':>11} " + " ".join(f"{'R(N=%d)' % N:>12}" for N in SIZES)
              + f" {'R20/R4':>8}  flags")
        for s in range(n):
            row = [grid["results"][init][N][s] for N in SIZES]
            if any(r is None for r in row):
                print(f"{s:>6}  (incomplete)")
                continue
            Rs = [r["correlator"] for r in row]
            flags = []
            for N, r in zip(SIZES, row):
                if not r.get("time_converged", True):
                    flags.append(f"drift@{N}")
                if r.get("positivity_violation"):
                    flags.append(f"neg@{N}")
                if max(r["final_bond_dims"]) >= CHI:
                    flags.append(f"CHI-LIMITED@{N}")
            print(f"{s:>6} {q[s]:>11.4e} " + " ".join(f"{R:12.4e}" for R in Rs)
                  + f" {Rs[-1]/Rs[0]:>8.3f}  {' '.join(flags) if flags else 'ok'}")

        print(f"\n  R/q by size (the earlier model gave a size-independent 1/8):")
        print(f"  {'N':>4} {'mean R/q':>10} {'spread':>8}")
        for N in SIZES:
            vals = [grid["results"][init][N][s]["correlator"] / q[s]
                    for s in range(n) if grid["results"][init][N][s] is not None]
            if not vals:
                continue
            v = np.array(vals)
            print(f"  {N:>4} {v.mean():>10.5f} "
                  f"{(v.max()-v.min())/v.mean()*100:>7.2f}%")

    # Uniqueness / relaxation check: a unique sector steady state forces both
    # starts to the same R, so the spread IS the error bar (Trap 2).
    print(f"\n  zero-vs-neel spread (unconverged runs straddle the true value):")
    print(f"  {'N':>4} {'worst':>8} {'median':>8}")
    for N in SIZES:
        d = []
        for s in range(n):
            a = grid["results"]["zero"][N][s]
            b = grid["results"]["neel"][N][s]
            if a is None or b is None:
                continue
            x, y = a["correlator"], b["correlator"]
            d.append(abs(x - y) / max(abs(x), abs(y)))
        if d:
            print(f"  {N:>4} {max(d)*100:>7.2f}% {np.median(d)*100:>7.2f}%")

    # Exact N=4 cross-check.
    ref = grid["exact_N4"]
    print(f"\n  N=4 against the exact sector steady state:")
    for init in INITS:
        errs = [abs(grid["results"][init][4][s]["correlator"] - ref["R"][s])
                / abs(ref["R"][s])
                for s in range(n) if grid["results"][init][4][s] is not None]
        if errs:
            print(f"    {init:>5}: worst relative error {max(errs)*100:.4f}%")

    ctrl = grid["control_Lpp0"]
    if ctrl:
        floor = max(abs(min(v for _, v in r["profile"])) for r in ctrl.values())
        print(f"\n  L''=0 control floor (worst profile point over N): {floor:.2e}")

    if len(grid["chi_sweep"]) > 1:
        print(f"\n  chi sweep, sample {SWEEP_SAMPLE}, N={SWEEP_N}, "
              f"init |{SWEEP_INIT}> (Trap 1):")
        print(f"  {'chi':>5} {'R':>13} {'max bond':>9} {'limited by':>11}")
        for c in sorted(grid["chi_sweep"]):
            r = grid["chi_sweep"][c]
            mb = max(r["final_bond_dims"])
            print(f"  {c:>5} {r['correlator']:>13.6e} {mb:>9d} "
                  f"{'chi' if mb >= c else 'cutoff':>11}")

    print("\n  R flat in N (ratio ~ 1) and flat in separation = SWSSB.", flush=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_grid(grid: dict) -> str:
    """R vs N per sample (one column per initial state) and the size ratio."""
    plt = ex._mpl()
    n = grid["config"]["n_samples"]
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, len(INITS), figsize=(6.5 * len(INITS), 9),
                             squeeze=False)
    for col, init in enumerate(INITS):
        ax0, ax1 = axes[0][col], axes[1][col]
        for s in range(n):
            pts = [(N, grid["results"][init][N][s]["correlator"]) for N in SIZES
                   if grid["results"][init][N][s] is not None]
            if not pts:
                continue
            color = cmap(s / max(n - 1, 1))
            ax0.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color,
                     lw=2.0, ms=6, label=f"sample {s}")
            ax1.plot([p[0] for p in pts[1:]],
                     [pts[k][1] / pts[k - 1][1] for k in range(1, len(pts))],
                     "-o", color=color, lw=2.0, ms=6)

        ctrl = grid["control_Lpp0"]
        if ctrl:
            floor = max(abs(min(v for _, v in r["profile"])) for r in ctrl.values())
            ax0.axhline(floor, ls="--", color="crimson", lw=1.5,
                        label=r"$L''=0$ control floor")

        ax0.set_yscale("log")
        ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
        ax0.set_title(rf"init $|{init}\rangle$")
        ax1.axhline(1.0, ls=":", color="grey", lw=1.5)
        ax1.set_ylabel(r"$R(N)/R(N-4)$")
        ax1.set_title("size-to-size ratio (1.0 = plateau = SWSSB)")
        for ax, ticks in ((ax0, SIZES), (ax1, SIZES[1:])):
            ax.set_xlabel("system size N")
            ax.set_xticks(ticks)
            ax.grid(True, alpha=0.3)
        if col == 0:
            ax0.legend(fontsize=8, ncol=2)

    fig.suptitle(rf"Driven classical circuit ($p={P_RIGHT}$): SWSSB correlator "
                 rf"over {n} random $L''$ ($\epsilon={grid['config']['epsilon']}$, "
                 rf"$\chi={CHI}$)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_drift_annihilation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_profile(grid: dict) -> str:
    """Full R(i, r) profile at N = PROFILE_SIZE, one panel per initial state."""
    plt = ex._mpl()
    N = PROFILE_SIZE
    n = grid["config"]["n_samples"]
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, len(INITS), figsize=(6.5 * len(INITS), 5),
                             squeeze=False)
    for col, init in enumerate(INITS):
        ax = axes[0][col]
        for s in range(n):
            res = grid["results"][init][N][s]
            if res is None:
                continue
            prof = res["profile"]
            ax.plot([r for r, _ in prof], [v for _, v in prof], "-o",
                    color=cmap(s / max(n - 1, 1)), alpha=0.85, label=f"sample {s}")
        ctrl = grid["control_Lpp0"].get(N)
        if ctrl:
            ax.plot([r for r, _ in ctrl["profile"]],
                    [abs(v) for _, v in ctrl["profile"]],
                    "k--o", lw=2.0, zorder=5, label=r"$|L''=0|$ control")
        ax.set_xlabel(f"site r   (reference site i = {N // 4})")
        ax.set_ylabel(r"$R(i,\ r)$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(rf"$N={N}$, init $|{init}\rangle$")
        if col == 0:
            ax.legend(fontsize=8, ncol=2)

    # The measured profiles decay exponentially (xi ~ 4-9 sites) and turn up
    # over the last two sites against the open boundary -- exclude those when
    # fitting xi, or the upturn inverts the slope.
    fig.suptitle(rf"Renyi-2 profile at $N={N}$, driven circuit ($p={P_RIGHT}$, "
                 rf"$\chi={CHI}$) -- exponential decay, finite $\xi$")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_drift_annihilation_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_chi_sweep(grid: dict) -> str:
    """R and the profile vs bond dimension at N = SWEEP_N."""
    plt = ex._mpl()
    chis = sorted(grid["chi_sweep"])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    ax0.plot(chis, [grid["chi_sweep"][c]["correlator"] for c in chis], "-o",
             color="crimson")
    ax0.axhline(grid["chi_sweep"][chis[-1]]["correlator"], ls=":", color="grey",
                label=rf"$\chi={chis[-1]}$ value")
    ax0.set_xlabel(r"bond dimension $\chi$")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax0.set_xticks(chis)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=9)
    ax0.set_title(f"sample {SWEEP_SAMPLE}, N={SWEEP_N}, init |{SWEEP_INIT}>")

    cmap = plt.get_cmap("plasma")
    for k, c in enumerate(chis):
        prof = grid["chi_sweep"][c]["profile"]
        ax1.plot([r for r, _ in prof], [v for _, v in prof], "-o",
                 color=cmap(k / max(len(chis) - 1, 1)), label=rf"$\chi={c}$")
    ax1.set_xlabel("site r")
    ax1.set_ylabel(r"$R(i,\ r)$")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title("profile vs bond dimension")

    fig.suptitle(rf"Bond-dimension convergence, driven circuit ($p={P_RIGHT}$, "
                 rf"$\epsilon={grid['config']['epsilon']}$)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_drift_annihilation_chi.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_grid()

    # Pickle before plotting: the data is the deliverable.
    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "renyi2_drift_annihilation.pkl")
    with open(path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\nSaved grid -> {path}", flush=True)

    for name, fn in (("summary", summarize), ("plot", plot_grid),
                     ("profile", plot_profile), ("chi plot", plot_chi_sweep)):
        try:
            out = fn(grid)
            if out:
                print(f"Saved {name:<9s} -> {out}", flush=True)
        except Exception:
            print(f"!! {name} failed (data is still saved):\n"
                  f"{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
