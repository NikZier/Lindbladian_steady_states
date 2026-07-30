"""All 10 samples, r=50 and r=100 only, one shared y-axis for direct comparison.

Why
---
Pure post-processing of the trajectories already on disk (same two source
pickles as imps_all10_samples_neel_summary.py) -- no new TEBD. That earlier
plot auto-scaled each panel's y-axis to its OWN 1st-99th percentile, which
hides how differently samples sit in absolute terms (e.g. sample 6 running
~20x higher than sample 8) and makes oscillation AMPLITUDE impossible to
compare across samples by eye. One shared y-axis across every panel fixes
that at the cost of some samples looking flatter than their own
self-scaled plot suggested -- that's the point, not a bug.

Samples 2 and 4 spend the last 24%/31% of their run with bond dimension
frozen at exactly 20, decaying smoothly and monotonically toward zero --
established as a numerical artifact (state trapped on a spurious low-rank
manifold), not physics, given the finite-N study's nonzero R for both.
Samples 5 and 9 show a milder version confined to the last ~185 time units.
That region is shaded on each affected panel rather than silently included
or excluded, so it reads as "known-unreliable" rather than either hidden or
mistaken for real late-time oscillation.

Outputs (experiments/results/):
    imps_all10_r50_r100_shared_axis.png -- one row per sample (10 rows),
        two columns (r=50, r=100), single shared y-axis, rolling mean and
        median plotted (faint raw trace for context), lock-in region shaded
        where present.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import renyi2_swssb as ex

REFERENCE_R = [50, 100]
ROLLING_WINDOW = 1000.0  # widened from 200: validated to cut CV 2-5x without shifting the mean
                          # (checked directly against these trajectories), while staying well
                          # under 8700/5 -- the user's concern that 2000 ate too much of the run

SOURCE_FILES = [
    "imps_all_samples_neel_timescale.pkl",
    "imps_remaining_samples_neel_timescale.pkl",
]


def load_all_runs() -> dict:
    runs = {}
    for fname in SOURCE_FILES:
        with open(os.path.join(ex.RESULTS_DIR, fname), "rb") as f:
            grid = pickle.load(f)
        runs.update(grid["runs"])
    return dict(sorted(runs.items()))


def rolling_mean_median(t: np.ndarray, v: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    half = window / 2.0
    mean = np.empty_like(v)
    median = np.empty_like(v)
    lo = hi = 0
    n = len(t)
    for i in range(n):
        while lo < n and t[lo] < t[i] - half:
            lo += 1
        while hi < n and t[hi] <= t[i] + half:
            hi += 1
        window_vals = v[lo:hi]
        mean[i] = np.nanmean(window_vals)
        median[i] = np.nanmedian(window_vals)
    return mean, median


def finite_reference(sample: int) -> tuple[float | None, str]:
    """Finite-N=20, chi=128, |neel> correlator for this sample, as a reference line.

    Prefers the long-schedule value where it exists (samples 0 and 8 only --
    those were the two re-run at 3.2x the schedule after the standard N=20
    grid turned out to be under-relaxed), else the standard chi=128 grid.

    Caveat worth keeping in view when reading the plot: the finite value is
    R(N/4, 3N/4) = R(5,15), i.e. separation 10, whereas the iMPS curves are
    at separation 50 and 100. Those are only comparable if the correlator is
    genuinely flat in separation -- which is the SWSSB claim being tested, so
    the comparison is meaningful but not a like-for-like identity. And the
    standard-schedule N=20 values (all samples except 0 and 8) are themselves
    known to be under-relaxed.

    Output: (value or None, label describing which schedule it came from).
    """
    long_path = os.path.join(ex.CACHE_DIR, f"long_sample{sample}_neel_N20_chi128.pkl")
    if os.path.exists(long_path):
        with open(long_path, "rb") as f:
            return pickle.load(f)["result"]["correlator"], "N=20 (long sched.)"
    std_path = os.path.join(ex.CACHE_DIR, f"chi128_sample{sample}_neel_N20_chi128.pkl")
    if os.path.exists(std_path):
        with open(std_path, "rb") as f:
            return pickle.load(f)["result"]["correlator"], "N=20 (std, under-relaxed)"
    return None, ""


def lock_in_start(bond: np.ndarray, t: np.ndarray) -> float | None:
    """Evolved time at which the trailing run of constant bond dimension
    begins, or None if the tail run is too short to call a lock-in (<5% of
    the trajectory -- the same threshold that separated samples 2/4/5/9 from
    the harmless few-stage tail coincidences in the other six samples)."""
    last_val = bond[-1]
    i = len(bond) - 2
    while i >= 0 and bond[i] == last_val:
        i -= 1
    run_len = len(bond) - 1 - i
    if run_len / len(bond) < 0.05:
        return None
    return float(t[i + 1])


def main() -> None:
    runs = load_all_runs()
    samples = sorted(runs.keys())

    # First pass: compute rolling series and the global shared y-range.
    series = {}
    all_roll = []
    for s in samples:
        traj = runs[s]["trajectory"]
        t = np.array([pt["t"] for pt in traj])
        bond = np.array([pt["bond"] for pt in traj])
        lock_t = lock_in_start(bond, t)
        series[s] = {"t": t, "lock_t": lock_t}
        for r in REFERENCE_R:
            v = np.array([pt["R"][r] for pt in traj])
            mean, median = rolling_mean_median(t, v, ROLLING_WINDOW)
            series[s][r] = {"raw": v, "mean": mean, "median": median}
            all_roll.append(mean)
            all_roll.append(median)

    # Finite-N reference per sample; include them in the shared y-range so no
    # reference line ends up off-scale.
    refs = {s: finite_reference(s) for s in samples}

    pooled = np.concatenate(all_roll)
    pooled = pooled[np.isfinite(pooled)]
    ylo, yhi = np.nanpercentile(pooled, [0.5, 99.5])
    ref_vals = [v for v, _ in refs.values() if v is not None]
    if ref_vals:
        ylo = min(ylo, min(ref_vals))
        yhi = max(yhi, max(ref_vals))
    pad = 0.1 * max(yhi - ylo, 1e-8)
    ylim = (ylo - pad, yhi + pad)
    print(f"shared y-axis: [{ylim[0]:.4e}, {ylim[1]:.4e}]  (0.5-99.5 percentile of all rolling series)")

    plt = ex._mpl()
    fig, axes = plt.subplots(len(samples), len(REFERENCE_R),
                              figsize=(6.0 * len(REFERENCE_R), 2.4 * len(samples)), squeeze=False)

    for row, s in enumerate(samples):
        t = series[s]["t"]
        lock_t = series[s]["lock_t"]
        for col, r in enumerate(REFERENCE_R):
            ax = axes[row][col]
            d = series[s][r]
            ax.plot(t, d["raw"], "-", color="tab:red", lw=0.4, alpha=0.15)
            ax.plot(t, d["mean"], "-", color="tab:red", lw=1.2, alpha=0.85, label="rolling mean")
            ax.plot(t, d["median"], "--", color="tab:blue", lw=1.3, alpha=0.9, label="rolling median")
            if lock_t is not None:
                ax.axvspan(lock_t, t[-1], color="grey", alpha=0.25, lw=0,
                           label="bond-dim locked (artifact)")
            ref_val, ref_label = refs[s]
            if ref_val is not None:
                ax.axhline(ref_val, color="black", ls="-", lw=1.4, alpha=0.8,
                           label=f"finite {ref_label}")
            ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
            ax.set_ylim(*ylim)
            ax.set_xlabel("evolved time", fontsize=8)
            ax.set_ylabel(rf"$R(0,{r})$", fontsize=8)
            ax.set_title(f"sample {s}, r = {r}" + ("  [LOCK-IN]" if lock_t is not None else ""),
                         fontsize=9, color="firebrick" if lock_t is not None else "black")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)
            elif ref_val is not None and col == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(r"All 10 samples, |neel>, r=50 and r=100, SHARED y-axis "
                 f"[{ylim[0]:.2e}, {ylim[1]:.2e}]"
                 "\nblack = finite N=20 reference (separation 10, not 50/100); "
                 "shaded = bond-dimension lock-in (not physical)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_all10_r50_r100_shared_axis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")


if __name__ == "__main__":
    main()
