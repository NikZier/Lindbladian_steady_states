"""Summary figures for the SWSSB study: finite-vs-infinite, R(epsilon), trajectories.

Three figures, all from already-computed data (no TEBD is run here):

1. imps_finite_vs_infinite.png
   Finite-chain R at (i,j) = (N/4, 3N/4) versus N at epsilon=0.2, with the
   infinite-system value alongside. The infinite point is drawn in a
   physically separated axis region past a break, labelled "inf", so it cannot
   be misread as just another system size -- it is not a chain of any length.

   The infinite value is plotted as R_iTEBD / 2. That factor is EMPIRICAL and
   UNEXPLAINED: R_iTEBD = 1.9875 +- 0.0025 x the finite-N law across all ten
   samples (spread 0.12%), with the numerator agreeing with exact dynamics to
   4.4% while the purity ratio is 0.531. Ruled out: measurement, finite-size
   reference (exact ED at N=4 and N=8 agree to 0.5%), positivity, bond
   dimension, and under-convergence. Two mechanisms were proposed and both
   are dead -- a two-component mixture (forbidden: A = X_i X_j^dag commutes
   with P = prod Z, so the cross term that would double R vanishes) and a
   degenerate transfer matrix (measured densely: the leading eigenvalue is
   simple). The division by 2 is therefore a stated adjustment, not a
   derivation, and the figure says so.

2. imps_correlator_vs_epsilon.png
   R(r=100) versus epsilon for all ten samples on log-log, with a fitted
   power law per sample. The fitted exponent is a free parameter, NOT fixed
   to 2 -- fixing it would assume the result the figure is meant to test. The
   pure quadratic through the epsilon=0.2 point is overlaid for reference.

3. imps_trajectories_all.png
   R(0,100) against evolved time, one panel per sample, one curve per
   epsilon. RAW values, with no factor applied. Log-log: the four epsilons
   span 16x in R and 16x in run length, so nothing else shows all of it at
   once. Faint line = per-stage values, solid = rolling median.

   Median, not mean: individual stages carry spikes up to 67x the plateau at
   epsilon=0.05 (16% of late points sit >50% off), which a mean tracks and a
   median does not.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import imps_eps_init_grids as grids
import renyi2_swssb as ex

SAMPLES = list(range(10))
FINITE_N = [4, 8, 12, 16, 20]
CHI = 128
R_INF = 100
EPSILONS = [0.20, 0.15, 0.10, 0.05]
INFINITE_FACTOR = 2.0   # empirical, unexplained -- see module docstring
ROLL_FRACTION = 0.02    # rolling-median window, as a fraction of run length


def load_results() -> dict:
    with open(os.path.join(ex.RESULTS_DIR, "imps_eps_init_grids.pkl"), "rb") as f:
        return pickle.load(f)["results"]


def finite_value(sample: int, N: int, init: str = "neel") -> float | None:
    """Finite-chain R at (N/4, 3N/4), preferring the longer N=20 run."""
    names = []
    if N == 20:
        names.append(f"long_sample{sample}_{init}_N20_chi{CHI}.pkl")
    names.append(f"chi{CHI}_sample{sample}_{init}_N{N}_chi{CHI}.pkl")
    for name in names:
        path = os.path.join(ex.CACHE_DIR, name)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return float(pickle.load(f)["result"]["correlator"])
            except Exception:
                pass
    return None


def finite_baseline(N: int) -> float | None:
    """L''=0 control at (N/4, 3N/4) from |neel>, the finite pipeline's noise floor.

    |neel> not |0...0>: the |0...0> control returns R = 0.000e+00 identically at
    every N because that state is exactly dark and never moves (bond dim 1), so
    it measures nothing. |neel> has to actually relax to the dark state, and
    what survives is truncation residue -- which is the floor the sample values
    must be compared against, and uses the same initial state they do.

    chi=128 where it exists (N=12,16,20); N=4 and 8 only have chi=32, where the
    state is bond dimension 1 anyway so chi cannot bind.
    """
    for name in (f"basectrl_baseline_neel_N{N}_chi128.pkl",
                 f"main_baseline_neel_N{N}_chi32.pkl"):
        path = os.path.join(ex.CACHE_DIR, name)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return float(pickle.load(f)["result"]["correlator"])
            except Exception:
                pass
    return None


BASELINE_ATTEMPTS = 3


def infinite_baseline() -> dict:
    """L''=0 control for the infinite system: WORST of several independent runs.

    Run several times and keep the largest R(0,100), because the control value
    IS numerical residue and residue is exactly the kind of absolute quantity
    that Trap 3 says is not reproducible in the infinite system. Measured:
    two runs at identical settings gave 1.37e-07 and 3.04e-05, a 200x spread.
    Quoting one run would set the noise floor by luck; the maximum is the
    conservative choice and is what the figure shades.

    Returns the full R(r) profile of the worst run, not just r=100: the control
    DECAYS with r whereas every sample is flat to ~1e-05, so the shape is
    itself a discriminator between "no long-range order" and the SWSSB signal.
    """
    path = os.path.join(ex.CACHE_DIR,
                        f"imps_baseline_infinite_neel_chi{CHI}_x{BASELINE_ATTEMPTS}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    import imps_swssb_infinite as ex_inf
    from lindblad_mps import iobservables, itebd, models
    L2 = ex.build_L2_terms(np.zeros((4, 4), dtype=complex))
    runs = []
    for attempt in range(BASELINE_ATTEMPTS):
        np.random.seed(31337 + attempt)
        state, _ = itebd.find_steady_state_infinite(
            H2_terms=[], H1_terms=[], L2_terms=L2, L1_terms=[],
            dt_schedule=grids.SCHEDULE_BASE, steps_per_dt=grids.STEPS_PER_DT,
            chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=grids.CANONICALIZE_EVERY,
            initial_state=ex_inf.build_initial_state("neel"))
        runs.append({"profile": dict(iobservables.correlator_profile(
            state, models.X, r_max=R_INF)), "bond": max(state.bond_dims.values())})
        print(f"    L''=0 infinite control, attempt {attempt}: "
              f"R(1)={runs[-1]['profile'][1]:.3e}  R(100)={runs[-1]['profile'][R_INF]:.3e}",
              flush=True)
    worst = max(runs, key=lambda r: r["profile"][R_INF])
    worst["all_runs"] = runs
    with open(path, "wb") as f:
        pickle.dump(worst, f)
    return worst


def infinite_values(results: dict) -> dict:
    """R(r=100) at each epsilon, per sample, from the converged iTEBD runs."""
    base = grids.load_neel_eps02(results)
    out = {}
    for eps in EPSILONS:
        for s in SAMPLES:
            if eps == 0.20:
                src = base.get(s)
            else:
                src = grids.pick(results, grids.EPS_GRID_KINDS[eps], s)
            if src is not None:
                out[(eps, s)] = grids.plateau(src["trajectory"], R_INF)
    return out


def rolling_median(t: np.ndarray, v: np.ndarray, window: float) -> np.ndarray:
    out = np.empty_like(v)
    lo = hi = 0
    n = len(t)
    for i in range(n):
        while lo < n and t[lo] < t[i] - window / 2:
            lo += 1
        while hi < n and t[hi] <= t[i] + window / 2:
            hi += 1
        out[i] = np.nanmedian(v[lo:hi])
    return out


# ---------------------------------------------------------------- figure 1
def figure_finite_vs_infinite(results: dict) -> str:
    plt = ex._mpl()
    inf = infinite_values(results)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [4, 1], "wspace": 0.04})
    ax, axinf = axes

    plotted: list[float] = []
    for s in SAMPLES:
        ys = [finite_value(s, N) for N in FINITE_N]
        xs = [N for N, y in zip(FINITE_N, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        plotted.extend(ys)
        c = cmap(s % 10)
        ax.plot(xs, ys, "o-", color=c, ms=6, lw=1.4, label=f"sample {s}")
        v = inf.get((0.20, s))
        if v is not None:
            plotted.append(v / INFINITE_FACTOR)
            ax.axhline(v / INFINITE_FACTOR, color=c, ls=":", lw=1.0, alpha=0.55)
            axinf.plot([0], [v / INFINITE_FACTOR], "*", color=c, ms=17,
                       mec="black", mew=0.7)

    # --- L''=0 controls ---
    # The FINITE control is a genuine noise floor: every finite chain is gapped
    # (gap ~ D(pi/N)^2), so the run relaxes to the dark state and what is left
    # is truncation residue.
    #
    # The INFINITE control is NOT a noise floor and must not be shaded as one.
    # With L''=0 the baseline is diffusion-limited pair annihilation A+A->0,
    # which is CRITICAL -- gapless in the thermodynamic limit, z=2 -- so there
    # is no gapped steady state for iTEBD to converge to. Measured across three
    # seeds it spans 4.7e-05 to 2.2e-03, a factor of 47, and in one run R even
    # INCREASES with r. That spread is the gaplessness, not numerics. It is
    # exactly the point of the study: the pair-creation matrix element in L''
    # is what gaps the Liouvillian, which is why the samples converge and this
    # does not.
    base_fin = {N: finite_baseline(N) for N in FINITE_N}
    inf_ctrl = infinite_baseline()
    inf_runs = sorted(r["profile"][R_INF] for r in inf_ctrl.get("all_runs", [inf_ctrl]))
    floor_top = max(v for v in base_fin.values() if v)
    # Do NOT extend the axis down to the smallest control (4.9e-09 at N=12):
    # that costs six decades and squashes the finite-vs-infinite comparison the
    # figure exists to show. Stop just below the largest control, which is the
    # one that actually bounds the signal, and annotate the rest as off-scale.
    y_lo = 2e-7

    ax.axhspan(1e-300, floor_top, color="0.55", alpha=0.20, zorder=0)
    ax.text(0.985, 0.045, r"$L''=0$ finite control (numerical floor)", transform=ax.transAxes,
            fontsize=9, style="italic", color="0.25", va="bottom", ha="right")
    shown = [(N, v) for N, v in base_fin.items() if v and v > y_lo]
    ax.plot([N for N, _ in shown], [v for _, v in shown], "x--", color="black",
            ms=9, mew=2.0, lw=1.4, zorder=6, label=r"$L''=0$ (finite)")
    below = sorted(N for N, v in base_fin.items() if v and v <= y_lo)
    if below:
        for N in below:
            ax.annotate("", xy=(N, y_lo * 1.25), xytext=(N, y_lo * 6.0),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
        ax.text(0.015, 0.045,
                rf"$L''=0$ at $N={{{', '.join(map(str, below))}}}$ lies below the axis "
                rf"($\leq 10^{{{int(np.floor(np.log10(max(base_fin[N] for N in below))))}}}$, "
                r"down to $10^{-112}$ at $N=4$)",
                transform=ax.transAxes, fontsize=8, style="italic", color="0.25")

    # Infinite control: the full seed-to-seed RANGE, not a point, since it does
    # not converge to anything.
    axinf.plot([0.34, 0.34], [inf_runs[0], inf_runs[-1]], "-", color="black",
               lw=2.4, zorder=6, solid_capstyle="butt")
    axinf.plot([0.34] * len(inf_runs), inf_runs, "_", color="black", ms=14, mew=2.2, zorder=7)
    axinf.annotate(r"$L''\!=\!0$" "\n" "gapless", xy=(0.34, inf_runs[0]),
                   xytext=(0, -13), textcoords="offset points", fontsize=7.5,
                   ha="center", va="top", color="black", style="italic")

    ax.set_yscale("log")
    ax.set_xticks(FINITE_N)
    ax.set_xlabel("system size $N$ (finite chain, open boundaries)")
    ax.set_ylabel(rf"$R(N/4,\, 3N/4)$")
    ax.set_title(r"Finite chains: $R$ at quarter/three-quarter sites")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.085),
              frameon=False)

    axinf.set_yscale("log")
    axinf.set_xticks([0]); axinf.set_xticklabels([r"$N=\infty$"], fontsize=12)
    axinf.set_xlim(-0.6, 0.6)
    # Suppress y labels on BOTH major and minor ticks: on a log axis the minor
    # formatter draws its own labels and would otherwise spill across the gap
    # into the finite panel.
    axinf.set_yticklabels([])
    axinf.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    axinf.yaxis.set_major_formatter(plt.matplotlib.ticker.NullFormatter())
    axinf.tick_params(axis="y", which="both", labelleft=False, labelright=False)
    axinf.grid(True, alpha=0.3, which="both")
    axinf.set_title("iTEBD" + "\n" + r"(thermodynamic limit)", fontsize=10, pad=14)
    axinf.set_facecolor("#f2f2f2")
    # Set limits from the DATA. axhspan(1e-300, ...) otherwise drags the
    # autoscaled range down by hundreds of decades and the panel renders empty.
    y_hi = 10 ** (np.log10(max(plotted)) + 0.35)
    for a in axes:
        a.set_ylim(y_lo, y_hi)

    # Break marks between the two panels: this is not a continuation of the
    # x-axis, it is a different kind of object.
    for x, a in ((1.0, ax), (0.0, axinf)):
        a.spines["right" if a is ax else "left"].set_visible(False)
        a.plot([x, x], [0, 1], transform=a.transAxes, color="white", lw=3,
               clip_on=False, zorder=5)

    fig.suptitle(r"Renyi-2 SWSSB correlator: finite chains vs the thermodynamic limit "
                 rf"($\epsilon=0.2$, $|neel\rangle$, $\chi={CHI}$)  —  "
                 rf"infinite value shown as $R_{{\rm iTEBD}}/{INFINITE_FACTOR:.0f}$",
                 fontsize=12)
    smallest_fin = min(v for v in (finite_value(s, N) for s in SAMPLES
                                   for N in FINITE_N) if v)
    print(f"  finite/infinite: weakest finite signal {smallest_fin:.3e}, worst finite "
          f"L''=0 control {floor_top:.3e}, ratio {smallest_fin/floor_top:.0f}x", flush=True)
    print(f"  infinite L''=0 control across seeds: "
          f"{'  '.join(f'{v:.3e}' for v in inf_runs)} (gapless, does not converge)", flush=True)
    fig.text(0.5, 0.012,
             r"Dotted lines carry each sample's infinite-volume value across the finite panel; the factor of 2 is "
             r"empirical and unexplained." "\n"
             rf"$L''\!=\!0$ finite control ($\times$, same $|neel\rangle$ start) is a true noise floor — the weakest "
             rf"finite signal sits ${smallest_fin/floor_top:.0f}\times$ above it." "\n"
             rf"The $L''\!=\!0$ infinite control is NOT a floor: unperturbed, the model is critical "
             rf"($A\!+\!A\!\to\!\emptyset$, gapless, $z\!=\!2$), so no steady state exists to converge to and it "
             rf"spans ${inf_runs[0]:.0e}$–${inf_runs[-1]:.0e}$ across seeds.",
             ha="center", fontsize=7.8, style="italic", linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.27, wspace=0.04)
    p = os.path.join(ex.RESULTS_DIR, "imps_finite_vs_infinite.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------- figure 2
def figure_correlator_vs_epsilon(results: dict) -> str:
    plt = ex._mpl()
    inf = infinite_values(results)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax, axr = axes
    eps = np.array(EPSILONS)
    fine = np.linspace(0.045, 0.21, 200)

    print(f"\n{'='*70}\nPower-law fit  R = c * epsilon^p   (p FREE, not fixed to 2)\n{'='*70}")
    print(f"{'s':>2} {'exponent p':>12} {'c':>12} {'max resid':>11}")
    exponents = []
    for s in SAMPLES:
        y = np.array([inf.get((e, s), np.nan) for e in EPSILONS])
        if not np.isfinite(y).all():
            continue
        p, logc = np.polyfit(np.log(eps), np.log(y), 1)
        c = np.exp(logc)
        resid = np.max(np.abs(y / (c * eps ** p) - 1))
        exponents.append(p)
        print(f"{s:>2} {p:>12.5f} {c:>12.5f} {resid:>10.2%}")

        col = cmap(s % 10)
        ax.loglog(eps, y, "o", color=col, ms=7, label=f"s{s}: $p$={p:.4f}")
        ax.loglog(fine, c * fine ** p, "-", color=col, lw=1.1, alpha=0.75)
        axr.semilogx(eps, y / (y[0] * (eps / eps[0]) ** 2), "o-", color=col, ms=6, lw=1.0)

    a = np.array(exponents)
    print(f"\nmean exponent {a.mean():.5f} +- {a.std():.5f}   (exactly 2 predicted)")

    ax.set_xlabel(r"$\epsilon = \|L''\|$")
    ax.set_ylabel(rf"$R(0,{R_INF})$   (infinite system)")
    ax.set_title(rf"$R$ vs $\epsilon$, fitted exponent $p={a.mean():.4f}\pm{a.std():.4f}$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=2)

    axr.axhline(1.0, color="black", ls="--", lw=1.5)
    axr.set_xlabel(r"$\epsilon$")
    axr.set_ylabel(r"$R(\epsilon)\ /\ [R(0.2)\,(\epsilon/0.2)^2]$")
    axr.set_title(r"Deviation from exact $\epsilon^2$ scaling")
    axr.set_xticks(EPSILONS); axr.set_xticklabels([str(e) for e in EPSILONS])
    axr.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    axr.grid(True, alpha=0.3, which="both")

    fig.suptitle(r"The SWSSB correlator is quadratic in the perturbation strength "
                 r"($R \propto \epsilon^2$), infinite system, $r=100$")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p_out = os.path.join(ex.RESULTS_DIR, "imps_correlator_vs_epsilon.png")
    fig.savefig(p_out, dpi=150); plt.close(fig)
    return p_out


# ---------------------------------------------------------------- figure 3
def figure_trajectories(results: dict) -> str:
    plt = ex._mpl()
    base = grids.load_neel_eps02(results)
    colors = {0.20: "tab:red", 0.15: "tab:orange", 0.10: "tab:blue", 0.05: "tab:green"}

    fig, axes = plt.subplots(5, 2, figsize=(14, 20), squeeze=False)
    for idx, s in enumerate(SAMPLES):
        ax = axes[idx // 2][idx % 2]
        for e in EPSILONS:
            src = base.get(s) if e == 0.20 else grids.pick(results, grids.EPS_GRID_KINDS[e], s)
            if src is None:
                continue
            traj = src["trajectory"]
            t = np.array([p["t"] for p in traj])
            v = np.array([p["R"][R_INF] for p in traj])
            good = t > 0
            t, v = t[good], v[good]
            ax.loglog(t, np.abs(v), "-", color=colors[e], lw=0.35, alpha=0.18)
            ax.loglog(t, np.abs(rolling_median(t, v, ROLL_FRACTION * t.max())), "-",
                      color=colors[e], lw=1.7, label=rf"$\epsilon={e}$")
        ax.set_xlabel("evolved time"); ax.set_ylabel(rf"$R(0,{R_INF})$")
        ax.set_title(f"sample {s}", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(r"$R(0,100)$ against evolved time, every sample and every $\epsilon$ "
                 r"(raw values, no factor applied)" "\n"
                 r"faint = per-stage, solid = rolling median", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(ex.RESULTS_DIR, "imps_trajectories_all.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main() -> None:
    results = load_results()
    print(f"Saved -> {figure_finite_vs_infinite(results)}", flush=True)
    print(f"Saved -> {figure_correlator_vs_epsilon(results)}", flush=True)
    print(f"Saved -> {figure_trajectories(results)}", flush=True)


if __name__ == "__main__":
    main()
