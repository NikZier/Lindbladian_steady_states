"""Rolling average/median of sample 8's R(r) trajectory, both timescale runs.

Why
---
The raw R(r,t) plots (imps_sample8_timescale.png / imps_sample8_timescale_long.png)
are dominated by occasional spikes of order 1e-2, which forces a linear y-axis
wide enough to hide the order-1e-4 structure the spikes are sitting on top of
-- exactly the structure the SWSSB question is actually about. This is pure
post-processing of the trajectories already saved in
imps_sample8_timescale.pkl / imps_sample8_timescale_long.pkl -- no new TEBD.

Rolling window is TIME-based (a fixed window in evolved-time units), not a
fixed number of stages: both schedules anneal dt from 0.1 down to 0.005, so a
fixed stage-count window would average over wildly different amounts of
evolved time depending on where in the schedule it sits (concretely: the
previous "last 30% of stages" summary for the long run turned out to cover
only the last 13.1% of evolved TIME -- see the session's write-up -- exactly
the kind of mismatch a time-based window avoids).

Both mean and median are shown: the spikes are heavy-tailed (a single 1e-2
event among ~1e-4 neighbours), so a rolling MEAN is still visibly dragged
around by them, while a rolling MEDIAN is close to immune -- comparing the
two on the same axes is itself informative about how much the spikes are
actually contaminating the naive average.

Outputs (experiments/results/):
    imps_sample8_rolling_average.png -- one row per run (1740-unit,
        8700-unit), one column per reference r, rolling mean and median
        overlaid on the (faint) raw trace, y-axis scaled to the rolling
        quantities' own range rather than the raw spikes' range.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import renyi2_swssb as ex

REFERENCE_R = [1, 5, 20, 50, 100]
WINDOW = 200.0  # time-window half-width is WINDOW/2 either side of each point

RUNS = [
    ("1740-unit run", "imps_sample8_timescale.pkl"),
    ("8700-unit run", "imps_sample8_timescale_long.pkl"),
]


def rolling_mean_median(t: np.ndarray, v: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-windowed rolling mean and median (not stage-count-based).

    For each point t[i], averages/medians over all points with
    |t[j] - t[i]| <= window/2 -- a fixed span of EVOLVED TIME regardless of
    how densely stages are packed in that span (which varies a lot across an
    annealed schedule).

    Input: t, v: equal-length 1D arrays (time, value) -- assumed t sorted ascending.
    Output: (mean, median), same length as t.
    """
    half = window / 2.0
    mean = np.empty_like(v)
    median = np.empty_like(v)
    lo = 0
    hi = 0
    n = len(t)
    for i in range(n):
        while lo < n and t[lo] < t[i] - half:
            lo += 1
        while hi < n and t[hi] <= t[i] + half:
            hi += 1
        window_vals = v[lo:hi]
        mean[i] = np.mean(window_vals)
        median[i] = np.median(window_vals)
    return mean, median


def load(pkl_name: str) -> dict:
    with open(os.path.join(ex.RESULTS_DIR, pkl_name), "rb") as f:
        return pickle.load(f)


def plot() -> str:
    plt = ex._mpl()
    colors = {"zero": "tab:blue", "neel": "tab:red"}

    fig, axes = plt.subplots(len(RUNS), len(REFERENCE_R), figsize=(4.2 * len(REFERENCE_R), 4.0 * len(RUNS)),
                              squeeze=False)

    for row, (label, pkl_name) in enumerate(RUNS):
        grid = load(pkl_name)
        for col, r in enumerate(REFERENCE_R):
            ax = axes[row][col]
            roll_vals_for_ylim = []
            for init in ("zero", "neel"):
                traj = grid["runs"][init]["trajectory"]
                t = np.array([pt["t"] for pt in traj])
                v = np.array([pt["R"][r] for pt in traj])
                mean, median = rolling_mean_median(t, v, WINDOW)
                roll_vals_for_ylim.append(mean)
                roll_vals_for_ylim.append(median)

                ax.plot(t, v, "-", color=colors[init], lw=0.4, alpha=0.15)
                ax.plot(t, mean, "-", color=colors[init], lw=1.6, alpha=0.9,
                        label=f"|{init}> rolling mean")
                ax.plot(t, median, "--", color=colors[init], lw=1.6, alpha=0.9,
                        label=f"|{init}> rolling median")

            # y-limits from the ROLLING quantities' own spread, not the raw
            # spikes -- this is the whole point: reveal the 1e-4 structure.
            all_roll = np.concatenate(roll_vals_for_ylim)
            lo, hi = np.percentile(all_roll, [1, 99])
            pad = 0.15 * max(hi - lo, 1e-8)
            ax.set_ylim(lo - pad, hi + pad)

            ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
            ax.set_xlabel("evolved time")
            ax.set_ylabel(rf"$R(0,{r})$")
            ax.set_title(f"{label}, r = {r}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, ncol=1)

    fig.suptitle(rf"Sample 8: rolling mean/median of $R(r)$ (window = {WINDOW:.0f} time units), "
                 "faint line = raw trajectory")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_sample8_rolling_average.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_final_rolling_values() -> None:
    """Print the rolling mean/median at the END of each trajectory -- the
    best available point estimate given the window."""
    print(f"{'run':>14} {'r':>4} {'init':>5}  {'roll. mean (final)':>19}  {'roll. median (final)':>20}")
    for label, pkl_name in RUNS:
        grid = load(pkl_name)
        for r in REFERENCE_R:
            for init in ("zero", "neel"):
                traj = grid["runs"][init]["trajectory"]
                t = np.array([pt["t"] for pt in traj])
                v = np.array([pt["R"][r] for pt in traj])
                mean, median = rolling_mean_median(t, v, WINDOW)
                print(f"{label:>14} {r:>4} {init:>5}  {mean[-1]:19.4e}  {median[-1]:20.4e}")


if __name__ == "__main__":
    print_final_rolling_values()
    print(f"\nSaved plot -> {plot()}")
