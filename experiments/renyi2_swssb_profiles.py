"""Reporting for the chi=128 headline run: the full N=20 profiles, and the
bond-dimension convergence picture redrawn now that all 10 samples exist.

Reads existing pickles and re-plots; runs no TEBD.

The N=20 profile R(i, r) at i = N//4 = 5 is the direct form of the SWSSB
question: long-range order means R does not decay as r moves away from i. The
ten samples span 1.9e-04 to 2.8e-03, a factor of 15, so they are shown twice --
absolute on a log axis (where they separate by magnitude) and normalized to
each sample's own value at the largest separation (where flatness is read
against a common scale).

The chi sweep needed redrawing for a reason, not for polish. It was run on
sample 0 alone, and sample 0 turns out to be one of the *easy* samples: its
N=20 state settles at bond dimension 72 of 128. Samples 1 and 9 reach 128 and
124, i.e. the cap is binding and their correlators are lower bounds. So the
sweep's conclusion -- "chi=128 is converged" -- does not generalize across the
ensemble, and the third panel added here says so explicitly.

Outputs (experiments/results/):
    renyi2_swssb_n20_profiles.png -- the ten N=20 profiles, absolute and normalized.
    renyi2_swssb_chi_extended.png -- REPLACES the previous two-panel version with
        a three-panel one; the underlying pickles are not modified.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import renyi2_swssb as ex

PROFILE_N = 20
INITS = ["zero", "neel"]  # a grid pickle that is absent is skipped
SWEEP_PKL = "renyi2_swssb_chi_extended.pkl"

# The L''=0 control floor: worst (most negative) point over the control
# profiles at chi=128. Anything at or below this is indistinguishable from
# the pipeline's own truncation noise.
CONTROL_FLOOR = 3.75e-06

CAPPED_COLOR = "crimson"  # bond dimension at (or near) chi_max -> a lower bound
OK_COLOR = "#4c72b0"

# A run does not have to reach chi_max exactly for the cap to be distorting it:
# the sweep's chi=96 point settles at 95, and sample 9 at N=20 settles at 124 of
# 128, both plainly squeezed. Treat anything within this fraction of the cap as
# cap-bound rather than pretending 124/128 is a converged result.
CAP_BOUND_FRAC = 0.9

# Size to normalize the scaling curves against. Deliberately not the smallest
# size in the grid: at N=4 the correlator sites are i=1, j=3, so both sit on the
# boundary of a four-site chain and R there is dominated by finite-size effects
# rather than by the bulk physics the plateau is about. N=12 is the smallest
# size whose runs are comfortably cutoff-bound.
NORM_N = 12


def is_cap_bound(max_bond: int, chi_max: int) -> bool:
    """Whether truncation is still shaping this run's correlator."""
    return max_bond >= CAP_BOUND_FRAC * chi_max


def load(name: str) -> dict:
    """Unpickle a results file by basename from experiments/results/."""
    with open(os.path.join(ex.RESULTS_DIR, name), "rb") as f:
        return pickle.load(f)


def fig_path(base: str, init: str) -> str:
    """Output path for a figure, suffixed by initial state ('' for zero)."""
    tail = "" if init == "zero" else f"_{init}"
    return os.path.join(ex.RESULTS_DIR, f"{base}{tail}.png")


def init_label(grid: dict) -> str:
    """LaTeX label for the grid's initial state, for figure titles."""
    return {"zero": r"$|0\ldots0\rangle$",
            "neel": r"$|0101\ldots\rangle$"}.get(grid["config"]["init"],
                                                 grid["config"]["init"])


def print_profile_table(grid: dict) -> None:
    """Print R(i, r) at N=PROFILE_N for every sample, one row per sample."""
    res = grid["results"][PROFILE_N]
    i = res[0]["i"]
    radii = [r for r, _ in res[0]["profile"]]

    print(f"\n=== Full Renyi-2 profile at N={PROFILE_N}, "
          f"chi={grid['config']['chi_max']}, "
          f"init=|{grid['config']['init']}>,  reference site i={i} ===")
    print(f"(SWSSB = flat in r. Control floor {CONTROL_FLOOR:.2e}.)\n")
    print(f"{'r':>7} " + " ".join(f"{r:>9d}" for r in radii)
          + f" {'flat?':>8} {'bond':>9}")
    for s, r_ in enumerate(res):
        vals = [v for _, v in r_["profile"]]
        # Flatness over the outer half of the profile, away from the short-range
        # transient near the reference site.
        outer = vals[len(vals) // 2:]
        spread = (max(outer) - min(outer)) / max(abs(v) for v in outer)
        maxbond = max(r_["final_bond_dims"])
        cap = "!" if maxbond >= grid["config"]["chi_max"] else " "
        print(f"sample{s:<1d} " + " ".join(f"{v:9.2e}" for v in vals)
              + f" {spread:7.1%} {maxbond:6d}/{grid['config']['chi_max']}{cap}")
    print("\n  'flat?' = spread of R over the outer half of the profile "
          "(small = long-range order).")
    print("  '!' = bond dimension pinned at chi_max, so that R is a lower bound.")


def _label_curve_ends(ax, anchors: list[tuple], side: str, min_sep_pts: float = 9.5) -> None:
    """Direct-label curves at one end, nudged apart so the text cannot overlap.

    With ten series, identity must not rest on telling ten colors apart, so
    every curve is labelled -- but several samples sit close together and the
    labels collide if placed naively. Offsets are solved in display space,
    which is why the figure must already be laid out when this is called.

    Input:
        ax: the axes to annotate.
        anchors: list of (x, y, text, color), one per curve.
        side: 'right' or 'left', which end of the curve the label sits at.
        min_sep_pts: minimum vertical gap between two labels, in points.
    Output: None; annotations are added to ax.
    """
    dpi = ax.figure.dpi
    order = sorted(range(len(anchors)), key=lambda k: anchors[k][1])
    disp = [ax.transData.transform((anchors[k][0], anchors[k][1]))[1] for k in order]

    min_sep_px = min_sep_pts * dpi / 72.0
    placed = list(disp)
    for k in range(1, len(placed)):
        if placed[k] - placed[k - 1] < min_sep_px:
            placed[k] = placed[k - 1] + min_sep_px

    dx = 5 if side == "right" else -5
    ha = "left" if side == "right" else "right"
    for slot, k in enumerate(order):
        x, y, text, color = anchors[k]
        dy_pts = (placed[slot] - disp[slot]) * 72.0 / dpi
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(dx, dy_pts), fontsize=8, color=color,
                    va="center", ha=ha)


def plot_profiles(grid: dict) -> str:
    """Plot the ten N=PROFILE_N profiles, absolute and normalized."""
    plt = ex._mpl()
    res = grid["results"][PROFILE_N]
    n = len(res)
    cmap = plt.get_cmap("viridis")
    chi = grid["config"]["chi_max"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    anchors0, anchors1, norm_vals = [], [], []
    for s, r_ in enumerate(res):
        color = cmap(s / max(n - 1, 1))
        radii = [r for r, _ in r_["profile"]]
        vals = [v for _, v in r_["profile"]]
        # Normalize to each sample's own mean, not to its endpoint: dividing by
        # R(i, r_max) would force every curve through 1.0 at the right edge,
        # manufacturing a convergence point that is an artefact of the choice.
        mean = sum(vals) / len(vals)
        norm = [v / mean for v in vals]
        norm_vals.extend(norm)
        label = f"{s}" + ("*" if max(r_["final_bond_dims"]) >= chi else "")

        ax0.plot(radii, vals, "-o", color=color, lw=1.8, ms=4, alpha=0.9)
        ax1.plot(radii, norm, "-o", color=color, lw=1.8, ms=4, alpha=0.9)
        anchors0.append((radii[-1], vals[-1], label, color))
        anchors1.append((radii[0], norm[0], label, color))  # left: curves widest here

    ax0.axhline(CONTROL_FLOOR, ls="--", color=CAPPED_COLOR, lw=1.5)
    ax0.annotate(r"$L''=0$ control floor", (radii[0], CONTROL_FLOOR),
                 textcoords="offset points", xytext=(2, 4), fontsize=8,
                 color=CAPPED_COLOR)
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$R(i,\ r)$")
    ax0.set_title("absolute (curves separate by magnitude)")

    ax1.axhline(1.0, ls=":", color="grey", lw=1.5)
    ax1.set_ylabel(r"$R(i,\ r)\ /\ \langle R(i,\cdot)\rangle$")
    pad = 0.08 * (max(norm_vals) - min(norm_vals))
    ax1.set_ylim(min(norm_vals) - pad, max(norm_vals) + pad)
    ax1.set_title("normalized to each sample's mean (flat = long-range order)")

    for ax in (ax0, ax1):
        ax.set_xlabel(f"site r   (reference site i = {res[0]['i']})")
        ax.set_xticks(radii[::2])
        ax.grid(True, alpha=0.25)
    ax0.margins(x=0.09)
    ax1.margins(x=0.09)

    fig.canvas.draw()  # transforms must be final before labels are placed
    _label_curve_ends(ax0, anchors0, "right")
    _label_curve_ends(ax1, anchors1, "left")

    fig.suptitle(rf"Renyi-2 profile at $N={PROFILE_N}$, $\chi={chi}$, "
                 rf"{n} random $L''$ ($\epsilon={grid['config']['epsilon']}$, "
                 rf"init {init_label(grid)})   "
                 rf"— curve labels are sample indices, * = $\chi$-limited")
    fig.tight_layout()
    path = fig_path("renyi2_swssb_n20_profiles", grid["config"]["init"])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_size_scaling(grid: dict) -> str:
    """Plot R(N/4, 3N/4) against system size for every sample.

    Left: absolute values at chi=128, with cap-bound points drawn open (their
    correlators are lower bounds) and the L''=0 control floor for scale.

    Right: the same curves normalized to each sample's own N=12 value, with the
    chi=32 data behind them. Normalizing is what makes the comparison legible --
    the samples span a factor of 15 in magnitude, and the question is not how
    big R is but whether it falls with N. The chi=32 curves are the point of the
    panel: they decay convincingly, and that decay is an artefact of truncation.
    """
    plt = ex._mpl()
    sizes = grid["config"]["sizes"]
    chi = grid["config"]["chi_max"]
    n = grid["config"]["n_samples"]
    cmap = plt.get_cmap("viridis")
    prev32 = grid.get("previous_chi", {}).get(32, {})

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    anchors, ratios_lo = [], []
    for s in range(n):
        color = cmap(s / max(n - 1, 1))
        runs = [grid["results"][N][s] for N in sizes]
        vals = [r["correlator"] for r in runs]
        capped = [is_cap_bound(max(r["final_bond_dims"]), chi) for r in runs]

        ax0.plot(sizes, vals, "-", color=color, lw=1.8, zorder=2)
        for N, v, cap in zip(sizes, vals, capped):
            ax0.plot([N], [v], "o", ms=7, mew=1.8, zorder=3, color=color,
                     mfc="none" if cap else color)
        label = f"{s}" + ("*" if any(capped) else "")
        anchors.append((sizes[-1], vals[-1], label, color))

        base = vals[sizes.index(NORM_N)]
        ax1.plot(sizes, [v / base for v in vals], "-o", color=color,
                 lw=1.8, ms=5, zorder=3)
        if prev32:
            lo_sizes = [N for N in sizes if N in prev32]
            lo = [prev32[N][s] for N in lo_sizes]
            if NORM_N in lo_sizes:
                base_lo = lo[lo_sizes.index(NORM_N)]
                ax1.plot(lo_sizes, [v / base_lo for v in lo], "--",
                         color=color, lw=1.1, alpha=0.45, zorder=1)
                ratios_lo.extend(v / base_lo for v in lo)

    ax0.axhline(CONTROL_FLOOR, ls="--", color=CAPPED_COLOR, lw=1.5)
    ax0.annotate(r"$L''=0$ control floor", (sizes[0], CONTROL_FLOOR),
                 textcoords="offset points", xytext=(2, 4), fontsize=8,
                 color=CAPPED_COLOR)
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax0.set_title(rf"absolute at $\chi={chi}$ (open marker = cap-bound)")

    ax1.axhline(1.0, ls=":", color="grey", lw=1.5)
    solid = plt.Line2D([], [], color="0.35", lw=1.8, marker="o", ms=5)
    dashed = plt.Line2D([], [], color="0.35", lw=1.1, ls="--", alpha=0.6)
    ax1.legend([solid, dashed],
               [rf"$\chi={chi}$ (converged): flat", r"$\chi=32$: decay is truncation"],
               fontsize=8, loc="lower left")
    ax1.set_ylabel(rf"$R(N)\ /\ R(N={NORM_N})$")
    ax1.set_ylim(min(ratios_lo + [0.9]) - 0.10, max(ratios_lo + [1.1]) + 0.10)
    ax1.set_title(rf"normalized to each sample's $N={NORM_N}$ value")

    for ax in (ax0, ax1):
        ax.set_xlabel("system size N")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.12)

    fig.canvas.draw()
    _label_curve_ends(ax0, anchors, "right")

    fig.suptitle(rf"SWSSB correlator vs system size, {n} random $L''$ "
                 rf"($\epsilon={grid['config']['epsilon']}$, init "
                 rf"{init_label(grid)})   — labels are sample indices")
    fig.tight_layout()
    path = fig_path("renyi2_swssb_size_scaling", grid["config"]["init"])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_convergence(sweep: dict, grid: dict) -> str:
    """Redraw the chi sweep, plus per-sample bond dimension at chi=128."""
    plt = ex._mpl()
    chis = sorted(sweep["results"])
    R = [sweep["results"][c]["correlator"] for c in chis]
    capped = [is_cap_bound(max(sweep["results"][c]["final_bond_dims"]), c) for c in chis]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5.2))

    # Panel 0: R vs chi. Marking which points are cap-bound is the whole point --
    # every chi <= 96 is a lower bound, which is what made the early decay look real.
    ax0.plot(chis, R, "-", color=OK_COLOR, lw=1.8, zorder=1)
    for c, r, cap in zip(chis, R, capped):
        ax0.plot([c], [r], "o", ms=9, zorder=2, mew=2,
                 color=CAPPED_COLOR if cap else OK_COLOR,
                 mfc="none" if cap else OK_COLOR)
    ax0.annotate("open = cap-bound,\nlower bounds only",
                 (chis[2], R[2]), textcoords="offset points", xytext=(12, -30),
                 fontsize=8, color=CAPPED_COLOR)
    ax0.annotate("cutoff-bound", (chis[-1], R[-1]), textcoords="offset points",
                 xytext=(-8, -20), fontsize=8, color=OK_COLOR, ha="right")
    ax0.margins(y=0.12)
    ax0.set_xlabel(r"bond dimension $\chi$")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax0.set_xticks(chis)
    ax0.grid(True, alpha=0.25)
    ax0.set_title(f"sample {sweep['config']['conv_sample']} only, "
                  f"N={sweep['config']['conv_N']}")

    # Panel 1: profile at each chi. chi is an ordered quantity, so a sequential
    # ramp is the right encoding here (unlike the sample index in panel 2).
    cmap = plt.get_cmap("plasma")
    for k, c in enumerate(chis):
        prof = sweep["results"][c]["profile"]
        ax1.plot([r for r, _ in prof], [v for _, v in prof], "-o",
                 color=cmap(k / max(len(chis) - 1, 1)), lw=1.6, ms=4,
                 label=rf"$\chi={c}$")
    ax1.set_xlabel("site r")
    ax1.set_ylabel(r"$R(i,\ r)$")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_title(r"profile vs $\chi$ (decay at low $\chi$ is truncation)")

    # Panel 2: the new information -- is chi=128 enough for every sample?
    chi_max = grid["config"]["chi_max"]
    res = grid["results"][PROFILE_N]
    bonds = [max(r["final_bond_dims"]) for r in res]
    colors = [CAPPED_COLOR if is_cap_bound(b, chi_max) else OK_COLOR for b in bonds]
    ax2.bar(range(len(bonds)), bonds, color=colors, width=0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OK_COLOR, CAPPED_COLOR)]
    ax2.legend(handles, ["cutoff-bound (converged)", "cap-bound (lower bound)"],
               fontsize=8, loc="lower right")
    ax2.axhline(chi_max, ls="--", color=CAPPED_COLOR, lw=1.5)
    ax2.annotate(rf"cap $\chi={chi_max}$", (len(bonds) - 0.5, chi_max),
                 textcoords="offset points", xytext=(-4, 5), fontsize=8,
                 color=CAPPED_COLOR, ha="right")
    for k, (b, c) in enumerate(zip(bonds, colors)):
        ax2.annotate(f"{b}", (k, b), textcoords="offset points", xytext=(0, 3),
                     fontsize=8, ha="center", color=c)
    ax2.set_xlabel("L'' sample")
    ax2.set_ylabel("achieved max bond dimension")
    ax2.set_xticks(range(len(bonds)))
    ax2.set_ylim(0, chi_max * 1.18)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.set_title(f"all samples at N={PROFILE_N}: red = within "
                  f"{100*(1-CAP_BOUND_FRAC):.0f}% of the cap")

    fig.suptitle(r"Bond-dimension convergence: the $\chi$ sweep was run on one "
                 r"sample, and that sample is not the hardest")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi_extended.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compare_inits(g_zero: dict, g_neel: dict) -> str:
    """Compare the two initial states -- the steady-state uniqueness test.

    Both starts lie in the (+,+) parity sector, so a unique steady state there
    forces the same R from either. A deviation is therefore either genuine
    degeneracy or, far more likely, one of the two runs not having relaxed:
    'zero' begins at the baseline's dark state and climbs from R = 0, so
    under-relaxation makes it read low, whereas 'neel' begins far away carrying
    real correlations. Where they agree, the value is converged from both
    directions; where they split, the larger one bounds the truth from below.
    """
    plt = ex._mpl()
    sizes = [N for N in g_zero["config"]["sizes"] if N in g_neel["config"]["sizes"]]
    n = g_zero["config"]["n_samples"]
    chi = g_zero["config"]["chi_max"]
    cmap = plt.get_cmap("viridis")

    print(f"\n=== zero vs neel at chi={chi}: relative difference "
          f"|z-n| / max(|z|,|n|) ===")
    print(f"{'s':>2} " + " ".join(f"{'N=%d' % N:>9}" for N in sizes))
    for s in range(n):
        diffs = []
        for N in sizes:
            z = g_zero["results"][N][s]["correlator"]
            v = g_neel["results"][N][s]["correlator"]
            diffs.append(abs(z - v) / max(abs(z), abs(v)))
        print(f"{s:>2} " + " ".join(f"{d:9.2%}" for d in diffs))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    anchors = []
    for s in range(n):
        color = cmap(s / max(n - 1, 1))
        z = [g_zero["results"][N][s]["correlator"] for N in sizes]
        v = [g_neel["results"][N][s]["correlator"] for N in sizes]
        ax0.plot(sizes, z, "-o", color=color, lw=1.8, ms=5)
        ax0.plot(sizes, v, "--s", color=color, lw=1.4, ms=5, mfc="none", alpha=0.85)
        ax1.plot(sizes, [b / a for a, b in zip(z, v)], "-o", color=color,
                 lw=1.8, ms=5)
        anchors.append((sizes[-1], v[-1] / z[-1], f"{s}", color))

    ax0.set_yscale("log")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    solid = plt.Line2D([], [], color="0.35", lw=1.8, marker="o", ms=5)
    dashed = plt.Line2D([], [], color="0.35", lw=1.4, ls="--", marker="s",
                        ms=5, mfc="none")
    ax0.legend([solid, dashed], [r"init $|0\ldots0\rangle$",
                                 r"init $|0101\ldots\rangle$"],
               fontsize=9, loc="best")
    ax0.set_title(rf"both starts at $\chi={chi}$")

    ax1.axhline(1.0, ls=":", color="grey", lw=1.5)
    ax1.set_ylabel(r"$R_{\mathrm{neel}}\ /\ R_{\mathrm{zero}}$")
    ax1.set_title("agreement (1.0 = same steady state from both starts)")

    for ax in (ax0, ax1):
        ax.set_xlabel("system size N")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.12)

    fig.canvas.draw()
    _label_curve_ends(ax1, anchors, "right")

    fig.suptitle(rf"Steady-state uniqueness in the $(+,+)$ sector: "
                 rf"does the answer depend on where you start? "
                 rf"($\epsilon={g_zero['config']['epsilon']}$, $\chi={chi}$)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_init_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved init comparison -> {path}")
    return path


def main() -> None:
    grids = {}
    for init in INITS:
        tail = "" if init == "zero" else f"_{init}"
        name = f"renyi2_swssb_chi128{tail}.pkl"
        if not os.path.exists(os.path.join(ex.RESULTS_DIR, name)):
            print(f"(no grid for init=|{init}> yet, skipping: {name})")
            continue
        grids[init] = grid = load(name)
        print_profile_table(grid)
        print(f"\nSaved profiles      -> {plot_profiles(grid)}")
        print(f"Saved size scaling  -> {plot_size_scaling(grid)}")

    # The chi sweep itself only exists for the zero start (one sample, many chi),
    # so its first two panels are zero-specific; the third panel is drawn from
    # whichever grid is available, preferring zero for continuity.
    if grids:
        base = grids.get("zero") or next(iter(grids.values()))
        print(f"Saved chi sweep     -> {plot_convergence(load(SWEEP_PKL), base)}")

    if len(grids) == 2:
        compare_inits(grids["zero"], grids["neel"])


if __name__ == "__main__":
    main()
