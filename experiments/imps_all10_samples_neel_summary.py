"""Combined summary across all 10 samples, |neel> only, 8700-unit trajectories.

Why
---
Pure post-processing of the two runs already on disk (imps_all_samples_neel_
timescale.pkl for samples 8/6/0, imps_remaining_samples_neel_timescale.pkl
for samples 1/2/3/4/5/7/9) -- no new TEBD. They were split into two runs
only to avoid a cache-write race with an already-in-flight job; both used
the identical schedule/chi/methodology (see imps_sample8_timescale_long.py
for the original rationale), so combining them is just concatenation.

NaN-robust on purpose: sample 9's trajectory has one non-finite stage (a
transient at t=20, very early, when its bond dimension briefly spiked to
157 before canonicalize() settled it back down -- one bad point out of 2175,
not a systemic problem). np.median propagates a single NaN across an entire
window, which is what corrupted sample 9's first-third numbers in the
original run's printed table and crashed its plot entirely (matplotlib
rejects NaN axis limits). Using nan-aware statistics throughout here is the
fix -- ignore the one bad point rather than let it corrupt everything
downstream of it.

Outputs (experiments/results/):
    imps_all10_samples_neel_summary.pkl -- combined median-by-thirds table
        for all 10 samples.
    imps_all10_samples_neel_summary.png -- rolling mean/median R(r) vs t,
        one row per sample (10 rows), one column per reference r.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import renyi2_swssb as ex

REFERENCE_R = [1, 5, 20, 50, 100]
ROLLING_WINDOW = 200.0

SOURCE_FILES = [
    "imps_all_samples_neel_timescale.pkl",          # samples 8, 6, 0
    "imps_remaining_samples_neel_timescale.pkl",     # samples 1,2,3,4,5,7,9
]


def load_all_runs() -> dict:
    """Load and merge both source pickles into one {sample: run_dict} map."""
    runs = {}
    for fname in SOURCE_FILES:
        with open(os.path.join(ex.RESULTS_DIR, fname), "rb") as f:
            grid = pickle.load(f)
        runs.update(grid["runs"])
    return dict(sorted(runs.items()))


def rolling_mean_median(t: np.ndarray, v: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-windowed rolling mean/median, NaN-robust (nanmean/nanmedian --
    a window straddling sample 9's one bad point ignores that point instead
    of returning NaN for every window it touches)."""
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


def median_by_thirds(trajectory: list[dict], r: int) -> list[float]:
    """NaN-robust median of R(r) in each of three equal-count windows."""
    v = np.array([pt["R"][r] for pt in trajectory])
    n = len(v)
    return [float(np.nanmedian(v[lo:hi])) for lo, hi in [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]]


def build_summary(runs: dict) -> dict:
    samples = sorted(runs.keys())
    return {s: {r: median_by_thirds(runs[s]["trajectory"], r) for r in REFERENCE_R} for s in samples}


def print_table(summary: dict) -> None:
    print(f"\n{'='*100}\nall 10 samples, init=|neel>, chi=256, 8700 time units -- "
          f"median-by-thirds drift (NaN-robust)\n{'='*100}")
    print(f"\n{'sample':>6} {'r':>4}  {'1st third':>11} {'2nd third':>11} {'3rd third':>11}  {'drift 1st->3rd':>14}")
    for s in sorted(summary):
        for r in REFERENCE_R:
            thirds = summary[s][r]
            drift = (thirds[2] / thirds[0] - 1) if thirds[0] and np.isfinite(thirds[0]) else float("nan")
            print(f"{s:>6} {r:>4}  {thirds[0]:11.3e} {thirds[1]:11.3e} {thirds[2]:11.3e}  {drift:+13.0%}")
        print()

    print("Reference: finite-N converged plateau values (N<=12, chi=128, established earlier this "
          "session) -- sample 8: 2.21e-4. Other samples' finite-N plateaus were not individually "
          "re-quoted here; compare orders of magnitude and cross-sample consistency instead.")
    print("\nsample 9 note: one transient non-finite stage at t=20 (very early, bond dimension "
          "briefly spiked to 157) is excluded via nanmedian rather than corrupting its whole "
          "first-third column, as it did in the original per-run printout.", flush=True)


def plot(runs: dict) -> str:
    plt = ex._mpl()
    samples = sorted(runs.keys())

    fig, axes = plt.subplots(len(samples), len(REFERENCE_R),
                              figsize=(4.2 * len(REFERENCE_R), 3.4 * len(samples)), squeeze=False)
    for row, s in enumerate(samples):
        traj = runs[s]["trajectory"]
        t = np.array([pt["t"] for pt in traj])
        for col, r in enumerate(REFERENCE_R):
            ax = axes[row][col]
            v = np.array([pt["R"][r] for pt in traj])
            mean, median = rolling_mean_median(t, v, ROLLING_WINDOW)

            finite = np.isfinite(v)
            ax.plot(t[finite], v[finite], "-", color="tab:red", lw=0.4, alpha=0.15)
            ax.plot(t, mean, "-", color="tab:red", lw=1.3, alpha=0.85, label="rolling mean")
            ax.plot(t, median, "--", color="tab:red", lw=1.5, alpha=0.9, label="rolling median")

            roll_concat = np.concatenate([mean, median])
            roll_concat = roll_concat[np.isfinite(roll_concat)]
            if len(roll_concat):
                lo, hi = np.nanpercentile(roll_concat, [1, 99])
                pad = 0.15 * max(hi - lo, 1e-8)
                ax.set_ylim(lo - pad, hi + pad)
            ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
            ax.set_xlabel("evolved time", fontsize=8)
            ax.set_ylabel(rf"$R(0,{r})$", fontsize=8)
            ax.set_title(f"sample {s}, r = {r}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    fig.suptitle(r"All 10 samples, |neel> only: rolling mean/median of $R(r)$ "
                 f"(window = {ROLLING_WINDOW:.0f} time units), faint = raw trajectory")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_all10_samples_neel_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    runs = load_all_runs()
    print(f"Loaded {len(runs)} samples: {sorted(runs.keys())}", flush=True)

    summary = build_summary(runs)
    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(ex.RESULTS_DIR, "imps_all10_samples_neel_summary.pkl"), "wb") as f:
        pickle.dump({"summary": summary, "reference_r": REFERENCE_R}, f)

    print_table(summary)
    print(f"\nSaved plot -> {plot(runs)}", flush=True)


if __name__ == "__main__":
    main()
