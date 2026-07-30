"""Sample 8 at N in {12, 16, 20} on a 3.2x longer TEBD schedule, both starts.

Why
---
Sample 8 is the one run in the headline chi=128 grid that looks like it loses
long-range order: from 'zero' it reads 2.2153, 2.2156, 2.2093, 2.0603, 1.9165
(x1e-4) across N = 4...20, flat to 0.3% up to N=12 and then falling 13%. It is
not chi-limited -- its bonds settle at 48 of 128 -- so truncation cannot be the
cause. From 'neel' the same sample reads 2.5976e-04 at N=20, 17% *above* the
plateau, and its profile *rises* with separation where the 'zero' profile
falls. Two starts straddling a flat value is the signature of a run stopped
mid-relaxation, not of a sample without order: a sample genuinely losing order
would decay from both.

The per-stage correlators say the same thing directly. At N=20 the last three
stages run 1.716 -> 1.848 -> 1.916 (from 'zero', still climbing) and
2.897 -> 2.653 -> 2.598 (from 'neel', still falling). Neither has stopped.

The fix is more evolved time, and the standard schedule has very little of it:
[0.1, 0.05, 0.02, 0.01, 0.005] x 300 steps is 55.5 time units, 45 of which sit
in the two coarse stages, while the final stage advances only 1.5 units against
a relaxation time of ~7-10 (fitted from the tail of the 'neel' approach).

The schedule
------------
Trotter cost is per *step* and evolved time is dt x steps, so time is cheapest
at large dt while accuracy of the fixed point needs small dt -- which is what
the annealing is for. LONG_DT_SCHEDULE therefore repeats each dt instead of
lengthening the tail: 201 time units in 4800 steps, versus 55.5 in 1500, i.e.
3.6x the time for 3.2x the cost. 189 of those units (~20 relaxation times)
are spent at or above dt=0.02, before the fine stages, whose job is only to
remove the O(dt^2) Trotter bias -- a shift measured at ~0.4% on the converged
N=12 run, so lagging it slightly costs nothing.

Repeating a dt also buys a free trajectory: `stage_correlators` is sampled once
per schedule entry, so 16 entries give R at 16 points in evolved time instead
of 5. That is what the plots show, and it answers the follow-up question of how
long a schedule N=20 actually needs.

What counts as an improvement
-----------------------------
Not "R changed". The test is that the two starts stop straddling: 'zero'
rising and 'neel' falling to a common value, at N=20 the plateau 2.21e-04 set
by the converged N<=12 runs, with the N=20 profile flattening out of its
current fall (zero) and rise (neel). N=12 is included as a null: it is already
converged (the two starts agree to 0.2%), so the long schedule must leave it
where it is. Sample 0 at N=20 is included as a second check -- its two starts
disagree by 5%, the same one-sided way, so if lengthening the schedule is the
general fix for N=20 rather than a sample-8 patch, its gap must close too.

Cache
-----
Runs are written under kind='long', a different cache key from the headline
grid's 'chi128', so nothing here can overwrite it -- important, since these are
the same (sample, N, init, chi) triples under a different schedule, and the
run_config guard would otherwise force a recompute of the grid on its next run.

Outputs (experiments/results/):
    renyi2_swssb_longrun.pkl -- the long runs, the schedule, and the standard-
        schedule runs they are compared against.
    renyi2_swssb_longrun.png -- R vs evolved time per size (the money plot),
        R vs N before/after, the N=20 profile before/after.
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
KIND = "long"  # cache namespace, kept distinct from the headline grid's 'chi128'
SAMPLE = 8
SIZES = [12, 16, 20]  # 12 is the null: already converged, must not move
INITS = ["zero", "neel"]

# A second sample whose two starts disagree at N=20 (7.172e-04 vs 7.532e-04,
# 5.0%, neel high -- the same one-sided under-relaxation signature). Run only
# at N=20, to distinguish "the N=20 schedule is too short" from "sample 8 is
# special". Set to [] to run sample 8 alone.
CONTROL_SAMPLES = [0]
CONTROL_N = 20

# Each entry is one stage of ex.STEPS_PER_DT (300) steps. See the module
# docstring: repeats of a dt, not a longer fine tail, because evolved time per
# unit cost is dt and the fine stages exist only to remove Trotter bias.
LONG_DT_SCHEDULE = [0.1] * 4 + [0.05] * 3 + [0.02] * 4 + [0.01] * 3 + [0.005] * 2

REFERENCE_KIND = "chi128"  # the standard-schedule runs to compare against


def elapsed_times(dt_schedule: list[float], steps_per_dt: int) -> list[float]:
    """Evolved time at the end of each stage of a schedule.

    Input: dt_schedule, one dt per stage; steps_per_dt, steps in every stage.
    Output: list of cumulative times, one per stage, same length as the schedule.
    """
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def run_job_long(job: dict) -> dict:
    """Worker entry point: pin the long schedule, then run the job normally.

    The schedule is a module-level constant of renyi2_swssb, read at call time
    both by run_steady_state_correlator (which uses it) and by run_config()
    (which stores it in the cache entry and compares it on reuse). Windows
    spawns workers that import this module fresh rather than inheriting the
    parent's memory, so the override has to happen inside the worker; setting
    it here makes that explicit instead of relying on import-time side effects.

    Input: job, a dict for ex.run_job().
    Output: ex.run_job()'s output, whose 'run_config' the parent verifies.
    """
    ex.DT_SCHEDULE = LONG_DT_SCHEDULE
    return ex.run_job(job)


def build_jobs(samples: list[dict]) -> list[dict]:
    """Enumerate the long-schedule runs, longest job first.

    Input: samples, from ex.draw_samples().
    Output: list of job dicts for run_job_long(), sorted by descending cost.
    """
    jobs = [
        {"kind": KIND, "label": f"sample{SAMPLE}", "sample": SAMPLE,
         "L_pp": samples[SAMPLE]["L_pp"], "N": N, "init": init, "chi_max": CHI}
        for N in SIZES
        for init in INITS
    ]
    jobs += [
        {"kind": KIND, "label": f"sample{s}", "sample": s,
         "L_pp": samples[s]["L_pp"], "N": CONTROL_N, "init": init, "chi_max": CHI}
        for s in CONTROL_SAMPLES
        for init in INITS
    ]
    # A control sample/size can coincide with the main grid. Two jobs sharing a
    # cache key are the same run: without this they would be computed twice,
    # concurrently, each writing the file the other is also writing.
    unique = {ex.job_key(job): job for job in jobs}
    jobs = sorted(unique.values(), key=ex.estimated_cost, reverse=True)
    return jobs


def reference_result(sample: int, N: int, init: str) -> dict | None:
    """Load the standard-schedule run of the same (sample, N, init) from the cache.

    Output: dict with 'result' and 'run_config' (the schedule it was run on,
        falling back to ex.LEGACY_RUN_CONFIG for entries written before that
        field existed), or None if the run is not cached.
    """
    path = os.path.join(ex.CACHE_DIR,
                        f"{REFERENCE_KIND}_sample{sample}_{init}_N{N}_chi{CHI}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        out = pickle.load(f)
    return {"result": out["result"],
            "run_config": out.get("run_config", ex.LEGACY_RUN_CONFIG)}


def run_all() -> dict:
    """Run every long-schedule job on a worker pool and collect the results.

    Verifies on every returned job that the long schedule actually reached the
    worker: a silently-unoverridden run would reproduce the standard schedule
    exactly and look like "lengthening the run changes nothing".

    Output: pickle-ready dict with the runs, their references and the config.
    """
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()
    jobs = build_jobs(samples)

    total_time = sum(LONG_DT_SCHEDULE) * ex.STEPS_PER_DT
    old_time = sum(ex.DT_SCHEDULE) * ex.STEPS_PER_DT
    n_steps = len(LONG_DT_SCHEDULE) * ex.STEPS_PER_DT
    old_steps = len(ex.DT_SCHEDULE) * ex.STEPS_PER_DT

    grid = {
        "config": {
            "epsilon": ex.EPSILON, "chi_max": CHI, "sample": SAMPLE,
            "sizes": SIZES, "inits": INITS, "kind": KIND,
            "control_samples": CONTROL_SAMPLES, "control_N": CONTROL_N,
            "dt_schedule": list(LONG_DT_SCHEDULE),
            "steps_per_dt": ex.STEPS_PER_DT,
            "recanonicalize_every": ex.RECANON_EVERY, "cutoff": ex.CUTOFF,
            "evolved_time": total_time, "n_steps": n_steps,
            "reference_dt_schedule": list(ex.DT_SCHEDULE),
            "reference_evolved_time": old_time,
            "site_rule": "i = N//4, j = 3N//4", "order_parameter": "X",
        },
        "descriptions": [s["description"] for s in samples],
        "runs": {},        # (sample, N, init) -> result
        "reference": {},   # (sample, N, init) -> {'result', 'run_config'}
    }

    print(f"{len(jobs)} long runs queued: sample {SAMPLE} x N in {SIZES} x "
          f"{INITS}" + (f", control sample(s) {CONTROL_SAMPLES} at N={CONTROL_N}"
                        if CONTROL_SAMPLES else "") + f", chi={CHI}.", flush=True)
    print(f"schedule: {len(LONG_DT_SCHEDULE)} stages, {n_steps} steps, "
          f"{total_time:.1f} time units "
          f"({n_steps/old_steps:.1f}x the cost and {total_time/old_time:.1f}x "
          f"the evolved time of the standard {old_steps}-step schedule).",
          flush=True)
    print("Expect the N=20 runs to take ~5 h each; they run in parallel.\n",
          flush=True)

    t0 = time.perf_counter()
    done, failures = 0, []
    with mp.Pool(processes=min(ex.N_WORKERS, len(jobs))) as pool:
        for out in pool.imap_unordered(run_job_long, jobs):
            done += 1
            key = (out["sample"], out["N"], out["init"])
            tag = (f"  [{done:2d}/{len(jobs)}] {out['label']:>9s} "
                   f"{out['init']:>4s} N={out['N']:2d}")
            if "error" in out:
                failures.append(out)
                print(f"{tag}  *** FAILED ***  "
                      f"{out['error'].strip().splitlines()[-1]}", flush=True)
                continue

            assert out["run_config"]["dt_schedule"] == LONG_DT_SCHEDULE, (
                f"{ex.job_key(out)} ran on {out['run_config']['dt_schedule']}, "
                f"not the long schedule -- the worker override did not apply"
            )
            res = out["result"]
            grid["runs"][key] = res
            grid["reference"][key] = reference_result(*key)
            ref = grid["reference"][key]
            change = (f"  ({res['correlator']/ref['result']['correlator'] - 1:+6.1%} "
                      f"vs standard)" if ref else "")
            timing = "cached" if out["cached"] else f"{out['seconds']/60:.0f}min"
            print(f"{tag}  R={res['correlator']:.6e}{change}  "
                  f"{ex.format_diagnostics(res)}"
                  f"bond={max(res['final_bond_dims']):3d}/{CHI}"
                  f"{'!' if max(res['final_bond_dims']) >= CHI else ' '} "
                  f"[{timing}, elapsed {(time.perf_counter()-t0)/60:.1f}min]",
                  flush=True)

    grid["failures"] = [{k: v for k, v in f.items() if k != "L_pp"} for f in failures]
    print(f"\n{done - len(failures)}/{len(jobs)} runs succeeded in "
          f"{(time.perf_counter()-t0)/60:.1f} min.", flush=True)
    for f in failures:
        print(f"  FAILED: {ex.job_key(f)}", flush=True)
    return grid


def plateau(grid: dict) -> float | None:
    """The converged reference value: R of sample SAMPLE at the smallest size.

    The N<=12 runs relax inside the standard schedule (their two starts agree
    to 0.2%), so their common value is what the larger sizes should approach.

    Output: mean of the two starts at SIZES[0], or None if those runs are absent.
    """
    vals = [grid["runs"][(SAMPLE, SIZES[0], init)]["correlator"]
            for init in INITS if (SAMPLE, SIZES[0], init) in grid["runs"]]
    return sum(vals) / len(vals) if vals else None


def summarize(grid: dict) -> None:
    """Print, per (sample, N), the two starts before and after and the gap between them.

    The gap -- 'neel' over 'zero' minus one -- is the diagnostic: a unique
    steady state in the (+,+) sector forces it to zero, and it is one-sided
    (neel high) wherever the runs are under-relaxed.
    """
    ref_val = plateau(grid)
    print(f"\n{'='*78}\nzero/neel agreement, standard schedule vs long schedule")
    if ref_val is not None:
        print(f"converged plateau (sample {SAMPLE}, N={SIZES[0]}): {ref_val:.4e}")
    print(f"{'='*78}")
    print(f"{'sample':>6} {'N':>3}  {'schedule':>8}  {'R(zero)':>11} {'R(neel)':>11} "
          f"{'gap':>7}  {'vs plateau':>21}")

    groups = [(SAMPLE, N) for N in SIZES]
    groups += [(s, CONTROL_N) for s in CONTROL_SAMPLES]
    for sample, N in groups:
        for which, store in (("standard", "reference"), ("long", "runs")):
            row = {}
            for init in INITS:
                entry = grid[store].get((sample, N, init))
                if entry is not None:
                    row[init] = (entry["result"] if which == "standard" else entry)
            if len(row) < len(INITS):
                print(f"{sample:>6} {N:>3}  {which:>8}  (incomplete)")
                continue
            Rz, Rn = row["zero"]["correlator"], row["neel"]["correlator"]
            gap = Rn / Rz - 1.0
            if ref_val and sample == SAMPLE:
                dev = (f"{Rz/ref_val - 1:+6.1%} / {Rn/ref_val - 1:+6.1%}")
            else:
                dev = ""
            print(f"{sample:>6} {N:>3}  {which:>8}  {Rz:11.4e} {Rn:11.4e} "
                  f"{gap:+7.1%}  {dev:>21}")
        print()

    print("Per-stage trajectory (R at the end of each stage, x1e-4):")
    for sample, N in groups:
        for init in INITS:
            res = grid["runs"].get((sample, N, init))
            if res is None:
                continue
            times = elapsed_times(grid["config"]["dt_schedule"],
                                  grid["config"]["steps_per_dt"])
            vals = [v for _, v in res["stage_correlators"]]
            tail = vals[-4:]
            spread = (max(tail) - min(tail)) / max(abs(v) for v in tail)
            print(f"  sample{sample} N={N:2d} {init:>4s}: "
                  + " ".join(f"{v*1e4:.3f}" for v in vals)
                  + f"   | last-4-stage spread {spread:.2%} "
                    f"(t = {times[-4]:.0f}...{times[-1]:.0f})")
    print("\nA run that has relaxed shows a flat tail AND a gap near zero; a gap "
          "that is still one-sided (neel high) means still under-relaxed.",
          flush=True)


def plot(grid: dict) -> str:
    """Plot R vs evolved time per size, R vs N before/after, and the N=20 profile.

    Output: path of the written PNG.
    """
    plt = ex._mpl()
    cfg = grid["config"]
    colors = {"zero": "tab:blue", "neel": "tab:red"}
    # Different widths, so that where the two starts agree exactly -- which is
    # the outcome being tested for -- the lower curve is still visible under
    # the upper one rather than being hidden by it.
    widths = {"zero": 3.0, "neel": 1.5}
    ref_val = plateau(grid)
    t_long = elapsed_times(cfg["dt_schedule"], cfg["steps_per_dt"])

    panels = [(SAMPLE, N) for N in SIZES] + [(s, CONTROL_N) for s in CONTROL_SAMPLES]
    ncol = max(len(panels), 3)
    fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol, 9), squeeze=False)

    # Row 1: the trajectory in evolved time, long run against the standard run.
    for col, (sample, N) in enumerate(panels):
        ax = axes[0][col]
        for init in INITS:
            res = grid["runs"].get((sample, N, init))
            if res is not None:
                vals = [v for _, v in res["stage_correlators"]]
                ax.plot(t_long[:len(vals)], vals, "-o", color=colors[init],
                        lw=widths[init], ms=4, label=f"|{init}>, long")
            ref = grid["reference"].get((sample, N, init))
            if ref:
                t_ref = elapsed_times(ref["run_config"]["dt_schedule"],
                                      ref["run_config"]["steps_per_dt"])
                vals = [v for _, v in ref["result"]["stage_correlators"]]
                ax.plot(t_ref[:len(vals)], vals, "--s", color=colors[init], lw=1.5,
                        ms=4, alpha=0.55, mfc="none", label=f"|{init}>, standard")
        if ref_val is not None and sample == SAMPLE:
            ax.axhline(ref_val, ls=":", color="k", lw=1.5,
                       label=f"N={SIZES[0]} plateau")
        ax.set_xlabel("evolved time")
        ax.set_ylabel(r"$R(N/4,\ 3N/4)$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"sample {sample}, N = {N}")
        if col == 0:
            ax.legend(fontsize=8)

    # Row 2, panel 0: R vs N for sample SAMPLE, both schedules.
    ax = axes[1][0]
    for init in INITS:
        for which, store, style in (("standard", "reference", "--s"),
                                    ("long", "runs", "-o")):
            pts = []
            for N in SIZES:
                entry = grid[store].get((SAMPLE, N, init))
                if entry is not None:
                    pts.append((N, (entry["result"] if which == "standard"
                                    else entry)["correlator"]))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        color=colors[init],
                        lw=widths[init] if which == "long" else 1.5,
                        ms=5, alpha=1.0 if which == "long" else 0.55,
                        mfc=None if which == "long" else "none",
                        label=f"|{init}>, {which}")
    if ref_val is not None:
        ax.axhline(ref_val, ls=":", color="k", lw=1.5)
    ax.set_xlabel("system size N")
    ax.set_ylabel(r"$R(N/4,\ 3N/4)$")
    ax.set_xticks(SIZES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"sample {SAMPLE}: size scaling, both schedules")

    # Row 2, panel 1: the profile at the largest size, both schedules.
    ax = axes[1][1]
    N = SIZES[-1]
    for init in INITS:
        res = grid["runs"].get((SAMPLE, N, init))
        if res is not None:
            ax.plot([r for r, _ in res["profile"]], [v for _, v in res["profile"]],
                    "-o", color=colors[init], lw=widths[init], ms=4,
                    label=f"|{init}>, long")
        ref = grid["reference"].get((SAMPLE, N, init))
        if ref:
            prof = ref["result"]["profile"]
            ax.plot([r for r, _ in prof], [v for _, v in prof], "--s",
                    color=colors[init], lw=1.5, ms=4, alpha=0.55, mfc="none",
                    label=f"|{init}>, standard")
    if ref_val is not None:
        ax.axhline(ref_val, ls=":", color="k", lw=1.5, label=f"N={SIZES[0]} plateau")
    ax.set_xlabel(f"site r   (reference site i = {N//4})")
    ax.set_ylabel(r"$R(i,\ r)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"sample {SAMPLE}, N = {N}: profile")

    for col in range(2, ncol):
        axes[1][col].axis("off")

    fig.suptitle(f"Sample {SAMPLE} on a longer TEBD schedule: "
                 f"{cfg['evolved_time']:.0f} time units in {cfg['n_steps']} steps "
                 f"vs {cfg['reference_evolved_time']:.0f} in "
                 f"{len(cfg['reference_dt_schedule'])*cfg['steps_per_dt']} "
                 rf"($\epsilon={cfg['epsilon']}$, $\chi={CHI}$)")
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_longrun.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grid = run_all()

    os.makedirs(ex.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_longrun.pkl")
    with open(path, "wb") as f:
        pickle.dump(grid, f)
    print(f"\nSaved runs -> {path}", flush=True)

    try:
        summarize(grid)
    except Exception:
        print(f"!! summary failed (data is still saved):\n{traceback.format_exc()}",
              flush=True)
    try:
        print(f"Saved plot -> {plot(grid)}", flush=True)
    except Exception:
        print(f"!! plot failed (data is still saved):\n{traceback.format_exc()}",
              flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
