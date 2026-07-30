"""How much evolved time does the CORRECTED iTEBD actually need?

Why
---
The 8700-time-unit schedule used before was sized to fight oscillations and
apparent non-convergence in R(r) -- both of which turned out to be
overwhelmingly a MEASUREMENT artifact, not slow physics: correlator_profile
used the wrong Lambda-weighting for its sweep, and canonicalize() left the
per-site Gamma/Lambda decomposition non-Vidal (merged-cell conditions held to
1e-7 while per-site conditions were violated by ~1e+7). Both are fixed and
cross-validated against the finite-chain code now.

So the schedule length is an open question again, and inheriting 8700 would
be spending hours to confirm what a short run may already show. In the
post-fix verification probe, t=200 at chi=48 already gave R flat to 4 digits
and matching the tiled finite chain exactly.

This measures it directly: run a few samples spanning the creation-rate range
(6 = largest, 0 = middle, 8 = smallest/slowest-relaxing) on a modest schedule
with fine stage sampling, and report R(r) versus evolved time so the plateau
time can be read off rather than guessed. Cheap by design (chi=128 -- bond
dimension never exceeded ~50 in any earlier run, so 128 is already generous
and 256 was never binding).

Outputs (experiments/results/):
    imps_schedule_length_check.pkl / .png -- R(r) vs evolved time per sample.
"""

import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import imps_swssb_infinite as ex_inf
import renyi2_swssb as ex
from lindblad_mps import iobservables, itebd, models

SAMPLES = [6, 0, 8]  # largest, middle, smallest pair-creation rate
INIT = "neel"
CHI = 128
R_MAX = 100
REFERENCE_R = [20, 50, 100]

STEPS_PER_DT = 50
# Reaches t=1200 -- 7x shorter than the previous 8700, but 6x longer than the
# t=200 at which the post-fix probe already looked converged, so it should
# bracket the plateau rather than assume it.
DT_SCHEDULE = [0.1] * 200 + [0.05] * 60 + [0.02] * 50 + [0.01] * 30 + [0.005] * 20
CANONICALIZE_EVERY = 10


def elapsed_times(dt_schedule, steps_per_dt):
    out, t = [], 0.0
    for dt in dt_schedule:
        t += dt * steps_per_dt
        out.append(t)
    return out


def run_one(job: dict) -> dict:
    """Evolve one sample, recording R(r) at every stage. Module-level for spawn."""
    s = job["sample"]
    t0 = time.perf_counter()
    trajectory = []
    times = elapsed_times(DT_SCHEDULE, STEPS_PER_DT)

    def stage_callback(stage, dt, state):
        prof = dict(iobservables.correlator_profile(state, models.X, r_max=max(REFERENCE_R)))
        trajectory.append({
            "t": times[stage],
            "R": {r: prof.get(r, float("nan")) for r in REFERENCE_R},
            "bond": max(state.bond_dims.values()),
        })

    L2 = ex.build_L2_terms(job["L_pp"])
    state, history = itebd.find_steady_state_infinite(
        H2_terms=[], H1_terms=[], L2_terms=L2, L1_terms=[],
        dt_schedule=DT_SCHEDULE, steps_per_dt=STEPS_PER_DT,
        chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
        initial_state=ex_inf.build_initial_state(INIT),
        stage_callback=stage_callback,
    )
    return {
        "sample": s, "trajectory": trajectory,
        "final_profile": iobservables.correlator_profile(state, models.X, r_max=R_MAX),
        "max_discarded_weight": max(history["discarded_weight"], default=0.0),
        "seconds": time.perf_counter() - t0,
    }


def finite_ref(sample: int) -> float | None:
    for kind in (f"long_sample{sample}_neel_N20_chi128.pkl",
                 f"chi128_sample{sample}_neel_N20_chi128.pkl"):
        p = os.path.join(ex.CACHE_DIR, kind)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)["result"]["correlator"]
    return None


def main() -> None:
    samples = ex.draw_samples()
    jobs = [{"sample": s, "L_pp": samples[s]["L_pp"]} for s in SAMPLES]
    total_t = sum(DT_SCHEDULE) * STEPS_PER_DT
    print(f"schedule-length check: samples {SAMPLES}, |{INIT}>, chi={CHI}, "
          f"{len(DT_SCHEDULE)} stages, {total_t:.0f} evolved time units.\n", flush=True)

    runs = {}
    t0 = time.perf_counter()
    with mp.Pool(processes=len(jobs)) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            runs[out["sample"]] = out
            last = out["trajectory"][-1]
            print(f"  sample{out['sample']}  done  R(100)={last['R'][100]:.4e}  "
                  f"bond={last['bond']}/{CHI}  maxdisc={out['max_discarded_weight']:.1e}  "
                  f"[{out['seconds']/60:.1f}min, elapsed {(time.perf_counter()-t0)/60:.1f}min]",
                  flush=True)

    with open(os.path.join(ex.RESULTS_DIR, "imps_schedule_length_check.pkl"), "wb") as f:
        pickle.dump({"runs": runs, "dt_schedule": DT_SCHEDULE,
                     "steps_per_dt": STEPS_PER_DT, "chi": CHI}, f)

    # How early does R(100) settle to within 1% of its final value, and stay?
    print(f"\n{'='*88}\nplateau time: earliest t after which R(100) stays within 1% of its "
          f"final value\n{'='*88}")
    print(f"{'sample':>6} {'R(100) final':>13} {'finite N=20':>13} {'ratio':>7} {'t_plateau':>10} "
          f"{'bond':>6} {'maxdisc':>9}")
    for s in SAMPLES:
        traj = runs[s]["trajectory"]
        t = np.array([p["t"] for p in traj])
        v = np.array([p["R"][100] for p in traj])
        final = v[-1]
        within = np.abs(v - final) <= 0.01 * abs(final)
        # earliest index from which ALL later points are within tolerance
        t_plat = t[-1]
        for i in range(len(v)):
            if within[i:].all():
                t_plat = t[i]
                break
        ref = finite_ref(s)
        ratio = final / ref if ref else float("nan")
        print(f"{s:>6} {final:13.4e} {ref if ref else float('nan'):13.4e} {ratio:7.2f} "
              f"{t_plat:10.0f} {traj[-1]['bond']:6d} {runs[s]['max_discarded_weight']:9.1e}")

    print("\nAlso: is the final profile FLAT in r? (the SWSSB signature, and the thing "
          "the measurement bug was destroying)")
    for s in SAMPLES:
        prof = dict(runs[s]["final_profile"])
        vals = np.array([prof[r] for r in range(20, 101)])
        print(f"  sample {s}: r=20..100  mean={vals.mean():.4e}  "
              f"spread(max-min)/mean={(vals.max()-vals.min())/abs(vals.mean()):.2%}")

    try:
        plt = ex._mpl()
        fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5.5 * len(SAMPLES), 4.5), squeeze=False)
        cmap = plt.get_cmap("viridis")
        for col, s in enumerate(SAMPLES):
            ax = axes[0][col]
            traj = runs[s]["trajectory"]
            t = [p["t"] for p in traj]
            for k, r in enumerate(REFERENCE_R):
                ax.plot(t, [p["R"][r] for p in traj], "-", lw=1.3,
                        color=cmap(k / max(len(REFERENCE_R) - 1, 1)), label=f"r={r}")
            ref = finite_ref(s)
            if ref:
                ax.axhline(ref, color="black", lw=1.4, label="finite N=20")
            ax.set_xlabel("evolved time")
            ax.set_ylabel(r"$R(0,r)$")
            ax.set_title(f"sample {s} (chi={CHI})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("Corrected iTEBD: how quickly does R(r) settle?")
        fig.tight_layout()
        path = os.path.join(ex.RESULTS_DIR, "imps_schedule_length_check.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\nSaved plot -> {path}", flush=True)
    except Exception:
        print(f"!! plot failed (data saved):\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
