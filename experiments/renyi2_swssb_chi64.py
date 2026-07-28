"""Bond-dimension extension of the SWSSB study.

The chi=32 sweep (renyi2_swssb.py) is not converged at the sizes that matter:
at N=20 the correlator reads 1.82e-04 at chi=32 but 3.49e-04 at chi=64 and is
still rising, and the apparent decay of R with N tracks the growth of the
discarded weight almost exactly. So the plateau-vs-decay question -- whether
SWSSB survives -- cannot be answered from that data. This script supplies the
two things needed to settle it:

  1. chi=64 at N=16 and N=20 for all 10 L'' samples, from the 'zero' (dark)
     start, so the N-scaling can be redone at a bond dimension where the
     truncation error is ~3x smaller.
  2. chi in {96, 128} at N=20 on the convergence sample, to find where R
     actually saturates (the existing sweep only brackets it from below).

Runs are dispatched through renyi2_swssb.run_job, so they share that module's
TEBD settings and its on-disk cache: anything already computed (notably
chi=64 at N=20 for the convergence sample) is reused rather than repeated,
and an interrupted run resumes where it stopped.

Outputs (experiments/results/):
    renyi2_swssb_chi64.pkl     -- the chi=64 grid, alongside the chi=32
        values from the main study for direct comparison.
    renyi2_swssb_chi_extended.pkl -- the full chi sweep at N=20
        {8,16,32,48,64,96,128}. The original convergence pickle is left
        untouched.
    renyi2_swssb_chi64.png     -- R vs N at chi=32 vs chi=64.
    renyi2_swssb_chi_extended.png -- R vs chi, and the profile at each chi.
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

CHI_HIGH = 64
GRID_SIZES = [16, 20]
GRID_INIT = "zero"
EXTRA_CHIS = [96, 128]


def build_extension_jobs(samples: list[dict]) -> list[dict]:
    """Enumerate the chi=64 grid plus the chi in {96,128} convergence points.

    The (sample CONV_SAMPLE, N=CONV_N, chi=64) run already exists in the cache
    under the convergence sweep's key, so that one job is emitted with the
    convergence key to score a cache hit instead of recomputing 18 minutes of
    identical work.

    Input: samples, from ex.draw_samples().
    Output: list of job dicts for ex.run_job(), longest first.
    """
    jobs = []
    for s, sample in enumerate(samples):
        for N in GRID_SIZES:
            is_existing = (s == ex.CONV_SAMPLE and N == ex.CONV_N)
            jobs.append({
                "kind": "conv" if is_existing else "chi64",
                "label": f"chi{CHI_HIGH}" if is_existing else f"sample{s}",
                "sample": s,
                "L_pp": sample["L_pp"],
                "N": N,
                "init": GRID_INIT,
                "chi_max": CHI_HIGH,
            })

    for chi in EXTRA_CHIS:
        jobs.append({
            "kind": "conv", "label": f"chi{chi}", "sample": ex.CONV_SAMPLE,
            "L_pp": samples[ex.CONV_SAMPLE]["L_pp"], "N": ex.CONV_N,
            "init": ex.CONV_INIT, "chi_max": chi,
        })

    jobs.sort(key=ex.estimated_cost, reverse=True)
    return jobs


def run_extension() -> tuple[dict, dict]:
    """Run every extension job on ex.N_WORKERS processes.

    Output: (grid, extended) -- the chi=64 grid and the full N=20 chi sweep,
        both pickle-ready.
    """
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()
    jobs = build_extension_jobs(samples)

    main = pickle.load(open(os.path.join(ex.RESULTS_DIR, "renyi2_swssb.pkl"), "rb"))
    prev_conv = pickle.load(
        open(os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi_convergence.pkl"), "rb")
    )

    grid = {
        "config": dict(main["config"], chi_high=CHI_HIGH, grid_sizes=GRID_SIZES,
                       grid_init=GRID_INIT),
        "descriptions": [s["description"] for s in samples],
        "chi32": {N: [s["results"][GRID_INIT][N] for s in main["samples"]]
                  for N in main["config"]["sizes"]},
        "chi64": {N: [None] * len(samples) for N in GRID_SIZES},
    }
    extended = {
        "config": dict(prev_conv["config"], chi_convergence=
                       sorted(set(prev_conv["config"]["chi_convergence"]) | set(EXTRA_CHIS))),
        "description": prev_conv["description"],
        "results": dict(prev_conv["results"]),
    }

    print(f"{len(jobs)} extension runs queued on {ex.N_WORKERS} workers.", flush=True)
    print(f"  chi={CHI_HIGH} grid: {len(samples)} samples x N in {GRID_SIZES}, "
          f"init=|{GRID_INIT}>", flush=True)
    print(f"  chi sweep extension: {EXTRA_CHIS} at N={ex.CONV_N}\n", flush=True)

    t0 = time.perf_counter()
    done, failures = 0, []
    with mp.Pool(processes=ex.N_WORKERS) as pool:
        for out in pool.imap_unordered(ex.run_job, jobs):
            done += 1
            tag = (f"  [{done:2d}/{len(jobs)}] {out['label']:>9s} N={out['N']:2d} "
                   f"chi={out['chi_max']:3d}")
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            res = out["result"]
            if out["chi_max"] == CHI_HIGH and out["N"] in GRID_SIZES:
                grid["chi64"][out["N"]][out["sample"]] = res
            if out["kind"] == "conv":
                extended["results"][out["chi_max"]] = res
            timing = "cached" if out["cached"] else f"{out['seconds']:.0f}s"
            print(f"{tag}  R({res['i']},{res['j']})={res['correlator']:.6e}  "
                  f"maxdisc={res['max_discarded_weight']:.1e}  "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]", flush=True)

    grid["failures"] = [{k: v for k, v in f.items() if k != "L_pp"} for f in failures]
    extended["failures"] = grid["failures"]
    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t0)/60:.1f} min.", flush=True)
    for f in failures:
        print(f"  FAILED: {ex.job_key(f)}", flush=True)
    return grid, extended


def plot_grid(grid: dict) -> str:
    """Plot R vs N at chi=32 against chi=64 for the sizes recomputed."""
    plt = ex._mpl()
    sizes32 = grid["config"]["sizes"]
    cmap = plt.get_cmap("viridis")
    n = len(grid["descriptions"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for s in range(n):
        color = cmap(s / max(n - 1, 1))
        y32 = [grid["chi32"][N][s]["correlator"] for N in sizes32]
        ax0.plot(sizes32, y32, "-o", color=color, alpha=0.5, lw=1.2)
        pts = [(N, grid["chi64"][N][s]["correlator"]) for N in GRID_SIZES
               if grid["chi64"][N][s] is not None]
        if pts:
            ax0.plot([p[0] for p in pts], [p[1] for p in pts], "--s",
                     color=color, lw=2.2, ms=7)
        # ratio of successive sizes, chi=32 vs chi=64 where available
        ax1.plot(sizes32[1:], [y32[i] / y32[i - 1] for i in range(1, len(y32))],
                 "-o", color=color, alpha=0.5, lw=1.2)
    if all(grid["chi64"][N][s] is not None for N in GRID_SIZES for s in range(n)):
        ratios = [grid["chi64"][20][s]["correlator"] / grid["chi64"][16][s]["correlator"]
                  for s in range(n)]
        ax1.plot([20] * n, ratios, "s", color="crimson", ms=8,
                 label=r"$16\to20$ at $\chi=64$")
        ax1.legend(fontsize=9)

    ax0.set_xlabel("system size N")
    ax0.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax0.set_yscale("log")
    ax0.set_xticks(sizes32)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(r"solid/round $\chi=32$,  dashed/square $\chi=64$")
    ax1.axhline(1.0, ls=":", color="grey")
    ax1.set_xlabel("system size N")
    ax1.set_ylabel(r"$R(N)/R(N-4)$")
    ax1.set_xticks(sizes32[1:])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("size-to-size ratio (1.0 = plateau)")

    fig.suptitle(rf"Bond-dimension effect on the N-scaling, init $|0\ldots0\rangle$ "
                 rf"($\epsilon={grid['config']['epsilon']}$)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi64.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid, extended = run_extension()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    p1 = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi64.pkl")
    p2 = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi_extended.pkl")
    with open(p1, "wb") as f:
        pickle.dump(grid, f)
    with open(p2, "wb") as f:
        pickle.dump(extended, f)
    print(f"\nSaved chi64 grid  -> {p1}", flush=True)
    print(f"Saved chi sweep   -> {p2}", flush=True)

    for name, fn, arg, out in (
        ("chi64 plot", plot_grid, grid, None),
        ("chi sweep plot", ex.plot_convergence, extended,
         os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi.png")),
    ):
        try:
            path = fn(arg)
            if out is not None:  # ex.plot_convergence writes the study-wide name
                os.replace(path, os.path.join(
                    ex.RESULTS_DIR, "renyi2_swssb_chi_extended.png"))
                path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi_extended.png")
            print(f"Saved {name:<15s} -> {path}", flush=True)
        except Exception:
            print(f"!! {name} failed (data is still saved):\n{traceback.format_exc()}",
                  flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
