"""The headline run: genericity of the SWSSB signal at converged bond dimension.

Everything established so far rests on a single L'' sample. At chi=128 that one
sample gives a Renyi-2 correlator flat in separation (7.14e-04 at r=6 ->
7.09e-04 at r=19, N=20) and flat in system size (7.12e-04 at N=16, 7.04e-04 at
N=20) -- consistent with SWSSB. At chi=32 and chi=64 the same sample shows a
convincing exponential decay in both, which is entirely fabricated by
truncation. So the bond dimension is settled and the sample count is not: this
script runs all 10 L'' samples at N in {12, 16, 20}, chi=128, from the 'zero'
(dark) start, which is what turns one sample into a result.

Nothing is aliased in from the existing cache even where a matching run exists
(`conv_chi128_zero_N20_chi128`, `probe_chi128_zero_N16_chi128`, both sample 0).
Those two were computed on a different machine, and at chi=32 the correlator
differs 5-12% across machines -- far more than the ~1% quoted for run-to-run
scatter on one machine. Two foreign points in an otherwise self-consistent grid
are not worth the ~90 minutes they would save. They also predate the
convergence diagnostics below.

Reading the output
------------------
Every run reports, besides R:

  drift=...   relative change in R over the final dt stage. A trailing '!'
              means it exceeds STAGE_DRIFT_TOL, i.e. the run is still moving
              and has NOT converged in time. The older 'converged' flag cannot
              see this: it measures one Trotter step, which shrinks with dt
              regardless of distance to the fixed point. The L''=0 'neel'
              control reports converged=True at N=20 while sitting at bond
              dimension 56, whose exact steady state has bond dimension 1.
  NEG         some R on the profile went negative. R = Tr[(A rho A) rho] with
              A Hermitian and unitary is a trace of two positive semidefinite
              matrices, so this is impossible for a physical rho and measures
              how much positivity truncation has destroyed.

Also check max(final_bond_dims) against chi_max on every run: equality means
the cap is binding and the number is a lower bound, not a result.

For scale, the L''=0 control at chi=128 (cache: basectrl_baseline_neel_N*)
puts the pipeline's false-positive floor at ~4e-06 -- the worst point on its
profile, against a signal of 7.04e-04. A sample whose R lands near that floor
is not evidence of anything.

Outputs (experiments/results/):
    renyi2_swssb_chi128.pkl -- the grid, the chi=32/chi=64 values for the same
        samples where available, and the L''=0 control.
    renyi2_swssb_chi128.png -- R vs N per sample, and the size-to-size ratio
        (1.0 = plateau = SWSSB).
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import renyi2_swssb as ex

CHI = 128
# N=4 and N=8 were added after the first pass. They are cheap -- an N-site chain
# of physical dimension 4 cannot exceed bond dimension 4^min(k, N-k), so at N=4
# the geometric bound is 16 and chi never binds at all (the chi=32 runs already
# settled at 14-16, i.e. they were exact). They anchor the small-N end of the
# size scaling, where the answer is not truncation-limited by construction.
SIZES = [4, 8, 12, 16, 20]

# Both strongly-symmetric starts, run on the same grid. They share the (+,+)
# parity sector, so a unique steady state there must give the same R from
# either -- which is the point of running both. 'zero' is the baseline's dark
# state and approaches the perturbed steady state from R = 0, i.e. from below;
# 'neel' starts far away carrying real correlations and approaches from a
# different direction, so together they bracket rather than merely repeat.
INITS = ["zero", "neel"]
CONTROL_KIND = "basectrl"  # the L''=0 neel runs at CHI, already in the cache


def suffix(init: str) -> str:
    """Output-filename suffix for an initial state ('' for the original zero run)."""
    return "" if init == "zero" else f"_{init}"


def build_jobs(samples: list[dict], init: str) -> list[dict]:
    """Enumerate the 10 x |SIZES| grid for one initial state, longest job first.

    Input: samples, from ex.draw_samples(); init, 'zero' or 'neel'.
    Output: list of job dicts for ex.run_job(), sorted by descending cost so
        the worker pool has a short tail.
    """
    jobs = [
        {"kind": f"chi{CHI}", "label": f"sample{s}", "sample": s,
         "L_pp": sample["L_pp"], "N": N, "init": init, "chi_max": CHI}
        for s, sample in enumerate(samples)
        for N in SIZES
    ]
    jobs.sort(key=ex.estimated_cost, reverse=True)
    return jobs


def load_control() -> dict:
    """Pull the L''=0 chi=128 control runs from the cache, if present.

    These are 'neel' runs whatever grid they are being compared against: the
    'zero' baseline is the dark state itself and is exactly 0 at every N, so it
    measures nothing. Only the neel baseline has a nontrivial transient and
    therefore a meaningful truncation-noise floor.

    Output: dict N -> result, empty if the control has not been run.
    """
    control = {}
    for N in SIZES:
        path = os.path.join(ex.CACHE_DIR, f"{CONTROL_KIND}_baseline_neel_N{N}_chi{CHI}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                control[N] = pickle.load(f)["result"]
    return control


def previous_grids(init: str) -> dict:
    """Load the chi=32 and chi=64 correlators for the same samples, for comparison.

    Output: dict chi -> {N -> [R per sample]}, omitting anything not on disk.
    """
    out = {}
    main_path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb.pkl")
    if os.path.exists(main_path):
        with open(main_path, "rb") as f:
            main = pickle.load(f)
        out[32] = {N: [s["results"][init][N]["correlator"] for s in main["samples"]]
                   for N in main["config"]["sizes"] if N in SIZES}
    grid64_path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi64.pkl")
    if os.path.exists(grid64_path):
        with open(grid64_path, "rb") as f:
            g64 = pickle.load(f)
        out[64] = {N: [r["correlator"] if r else None for r in g64["chi64"][N]]
                   for N in g64["chi64"] if N in SIZES}
    return out


def run_grid(init: str) -> dict:
    """Run every grid job for one initial state on ex.N_WORKERS processes.

    Output: pickle-ready dict with the grid, the control and the earlier chis.
    """
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()
    jobs = build_jobs(samples, init)

    grid = {
        "config": {
            "epsilon": ex.EPSILON, "chi_max": CHI, "sizes": SIZES, "init": init,
            "n_samples": len(samples), "base_seed": ex.BASE_SEED,
            "cutoff": ex.CUTOFF, "dt_schedule": ex.DT_SCHEDULE,
            "steps_per_dt": ex.STEPS_PER_DT,
            "recanonicalize_every": ex.RECANON_EVERY,
            "stage_drift_tol": ex.STAGE_DRIFT_TOL,
            "site_rule": "i = N//4, j = 3N//4", "order_parameter": "X",
        },
        "descriptions": [s["description"] for s in samples],
        "results": {N: [None] * len(samples) for N in SIZES},
        "control_Lpp0": load_control(),
        "previous_chi": previous_grids(init),
    }

    print(f"{len(jobs)} runs queued on {ex.N_WORKERS} workers: "
          f"{len(samples)} samples x N in {SIZES}, chi={CHI}, init=|{init}>.",
          flush=True)
    print(f"Watch for '!' (still drifting) and 'NEG' (positivity lost); "
          f"control floor is ~4e-06.\n", flush=True)

    t0 = time.perf_counter()
    done, failures = 0, []
    with mp.Pool(processes=ex.N_WORKERS) as pool:
        for out in pool.imap_unordered(ex.run_job, jobs):
            done += 1
            tag = f"  [{done:2d}/{len(jobs)}] {out['label']:>9s} N={out['N']:2d}"
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            res = out["result"]
            grid["results"][out["N"]][out["sample"]] = res
            timing = "cached" if out["cached"] else f"{out['seconds']:.0f}s"
            maxbond = max(res["final_bond_dims"])
            print(f"{tag}  R={res['correlator']:.6e}  "
                  f"{ex.format_diagnostics(res)}"
                  f"bond={maxbond:3d}/{CHI}{'!' if maxbond >= CHI else ' '} "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]",
                  flush=True)

    grid["failures"] = [{k: v for k, v in f.items() if k != "L_pp"} for f in failures]
    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t0)/60:.1f} min.", flush=True)
    for f in failures:
        print(f"  FAILED: {ex.job_key(f)}", flush=True)
    return grid


def summarize(grid: dict) -> None:
    """Print the size-to-size ratios and every convergence flag that fired."""
    n = grid["config"]["n_samples"]
    print(f"\n=== init |{grid['config']['init']}> ===")
    ratio_label = f"R{SIZES[-1]}/R{SIZES[0]}"
    print(f"\n{'sample':>7} " + " ".join(f"{'R(N=%d)' % N:>12}" for N in SIZES)
          + f" {ratio_label:>9}  flags")
    for s in range(n):
        row = [grid["results"][N][s] for N in SIZES]
        if any(r is None for r in row):
            print(f"{s:>7}  (incomplete)")
            continue
        Rs = [r["correlator"] for r in row]
        ratio = Rs[-1] / Rs[0] if Rs[0] else float("nan")
        flags = []
        for N, r in zip(SIZES, row):
            if not r.get("time_converged", True):
                flags.append(f"drift@N{N}")
            if r.get("positivity_violation"):
                flags.append(f"neg@N{N}")
            if max(r["final_bond_dims"]) >= grid["config"]["chi_max"]:
                flags.append(f"CHI-LIMITED@N{N}")
        print(f"{s:>7} " + " ".join(f"{R:12.4e}" for R in Rs)
              + f" {ratio:9.3f}  {' '.join(flags) if flags else 'ok'}")

    ctrl = grid["control_Lpp0"]
    if ctrl:
        floor = max(abs(min(v for _, v in r["profile"])) for r in ctrl.values())
        print(f"\n  L''=0 control floor (worst profile point over N): {floor:.2e}")
    print("  ratio ~ 1.0 across samples = R independent of N = SWSSB; "
          "ratio << 1 = decay.", flush=True)


def plot_grid(grid: dict) -> str:
    """Plot R vs N per sample at chi=128, and the size-to-size ratio."""
    plt = ex._mpl()
    n = grid["config"]["n_samples"]
    cmap = plt.get_cmap("viridis")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for s in range(n):
        color = cmap(s / max(n - 1, 1))
        pts = [(N, grid["results"][N][s]["correlator"]) for N in SIZES
               if grid["results"][N][s] is not None]
        if not pts:
            continue
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

    ax0.set_xlabel("system size N")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax0.set_yscale("log")
    ax0.set_xticks(SIZES)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, ncol=2)
    ax0.set_title(rf"$\chi={grid['config']['chi_max']}$, "
                  rf"init $|{grid['config']['init']}\rangle$")

    ax1.axhline(1.0, ls=":", color="grey", lw=1.5)
    ax1.set_xlabel("system size N")
    ax1.set_ylabel(r"$R(N)/R(N-4)$")
    ax1.set_xticks(SIZES[1:])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("size-to-size ratio (1.0 = plateau = SWSSB)")

    fig.suptitle(rf"Genericity of the SWSSB correlator over {n} random $L''$ "
                 rf"($\epsilon={grid['config']['epsilon']}$, "
                 rf"$\chi={grid['config']['chi_max']}$, "
                 rf"init $|{grid['config']['init']}\rangle$)")
    fig.tight_layout()
    path = os.path.join(
        ex.RESULTS_DIR, f"renyi2_swssb_chi128{suffix(grid['config']['init'])}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_one_init(init: str) -> None:
    """Run, save, summarize and plot the grid for a single initial state."""
    grid = run_grid(init)

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, f"renyi2_swssb_chi128{suffix(init)}.pkl")
    with open(path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\nSaved grid -> {path}", flush=True)

    try:
        summarize(grid)
    except Exception:
        print(f"!! summary failed (data is still saved):\n{traceback.format_exc()}",
              flush=True)
    try:
        print(f"Saved plot -> {plot_grid(grid)}", flush=True)
    except Exception:
        print(f"!! plot failed (data is still saved):\n{traceback.format_exc()}",
              flush=True)


def main() -> None:
    """Run the grid for every initial state in INITS, one after the other.

    Anything already cached returns immediately, so re-running after adding an
    initial state costs only the new work.
    """
    for init in INITS:
        print(f"\n{'='*70}\ninitial state |{init}>\n{'='*70}", flush=True)
        run_one_init(init)


if __name__ == "__main__":
    mp.freeze_support()
    main()
