"""Does repeated canonicalize() on the SAME state (no evolution in between) drift?

Why
---
The oscillation-diagnosis session established: truncation isn't the cause
(max_discarded_weight ~ machine precision for all 10 samples) and it isn't a
shared schedule artifact (bond-dimension spikes are sample-specific, not
tied to the dt-transition times). The remaining untested hypothesis is
specifically about canonicalize() itself: it finds the leading eigenvector
of a transfer operator via ARPACK, and if that eigenvalue's subspace isn't
perfectly simple (a risk flagged when this algorithm was built -- the
model's strong Z2 symmetry could in principle make the transfer operator
imprimitive), ARPACK could converge to a slightly different valid gauge on
different calls, which would show up as noise in observables even with zero
truncation and zero further time evolution.

This isolates that question cleanly: evolve a state partway (through the
same schedule prefix, dt=0.1, so it's representative of what canonicalize()
actually sees mid-run, not a trivial near-product state), then call
canonicalize() MANY TIMES IN A ROW with NO gate application in between. If
the gauge is stable, R(r) and xi must be identical (to solver/float
precision) across all repeats -- any spread here is coming from
canonicalize() itself, not from evolution.

One sample from the noisy group (sample 0) and one from the clean group
(sample 6, the outlier with the largest pair-creation rate) -- if canonicalize
were the noise source, there's no obvious reason it would respect that
distinction, so running both is also a light cross-check on the "genuine
physics" explanation.

Outputs: printed only (no plot needed for a yes/no stability question).
"""

import multiprocessing as mp
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import imps_swssb_infinite as ex_inf
import renyi2_swssb as ex
from lindblad_mps import iobservables, itebd, models

SAMPLES = [0, 6]
CHI = 256
N_REPEATS = 15  # repeated canonicalize() calls per sample, no evolution between them

# Reach a "typical mid-run" state before testing -- deep enough that the
# initial-transient bond-dimension spike (universal, t<250 in every sample)
# has passed, matching where the real oscillation lives in the actual runs.
PREFIX_SCHEDULE = [0.1] * 600  # t = 600*0.1*50 = 3000
STEPS_PER_DT = 50
CANONICALIZE_EVERY = 10


def run_one(job: dict) -> dict:
    """Evolve one sample to t=3000, then canonicalize N_REPEATS times with
    no further evolution. Module-level so it survives pickling to a worker
    process (Windows spawn)."""
    s = job["sample"]
    t0 = time.perf_counter()
    L2_terms = ex.build_L2_terms(job["L_pp"])
    state, _ = itebd.find_steady_state_infinite(
        H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[],
        dt_schedule=PREFIX_SCHEDULE, steps_per_dt=STEPS_PER_DT,
        chi_max=CHI, cutoff=ex.CUTOFF, canonicalize_every=CANONICALIZE_EVERY,
        initial_state=ex_inf.build_initial_state("neel"),
    )
    prefix_minutes = (time.perf_counter() - t0) / 60
    prefix_bond = dict(state.bond_dims)

    rows = []
    for i in range(N_REPEATS):
        diag = state.canonicalize(chi_max=CHI, cutoff=ex.CUTOFF)
        profile = iobservables.correlator_profile(state, models.X, r_max=100)
        prof_map = dict(profile)
        xi_diag = iobservables.correlation_length(state)
        rows.append({
            "eigenvalue": diag["eigenvalue_left_env"],
            "R50": prof_map[50], "R100": prof_map[100],
            "xi": xi_diag["xi"], "reason": xi_diag["reason"],
            "bond": max(state.bond_dims.values()),
        })

    return {"sample": s, "prefix_minutes": prefix_minutes, "prefix_bond": prefix_bond, "rows": rows}


def report(out: dict) -> None:
    s = out["sample"]
    rows = out["rows"]
    print(f"\n{'='*80}\nsample {s}: reached t=3000 in {out['prefix_minutes']:.1f} min "
          f"(bond dims = {out['prefix_bond']}), then canonicalized {N_REPEATS} times "
          f"with no further evolution\n{'='*80}", flush=True)

    print(f"\n  {'call':>4} {'eigenvalue-1':>14} {'R(50)':>13} {'R(100)':>13} "
          f"{'xi':>10} {'bond':>5}")
    for i, r in enumerate(rows):
        xi_str = f"{r['xi']:.3f}" if r["xi"] is not None else "None"
        print(f"  {i:>4} {r['eigenvalue']-1:>+14.3e} {r['R50']:>+13.4e} "
              f"{r['R100']:>+13.4e} {xi_str:>10} {r['bond']:>5}")

    r50_vals = np.array([r["R50"] for r in rows])
    r100_vals = np.array([r["R100"] for r in rows])
    print(f"\n  R(50)  across {N_REPEATS} repeats: mean={r50_vals.mean():.4e}  "
          f"std={r50_vals.std():.4e}  spread(max-min)={r50_vals.max()-r50_vals.min():.4e}  "
          f"relative spread={100*(r50_vals.max()-r50_vals.min())/abs(r50_vals.mean()):.4f}%")
    print(f"  R(100) across {N_REPEATS} repeats: mean={r100_vals.mean():.4e}  "
          f"std={r100_vals.std():.4e}  spread(max-min)={r100_vals.max()-r100_vals.min():.4e}  "
          f"relative spread={100*(r100_vals.max()-r100_vals.min())/abs(r100_vals.mean()):.4f}%")
    print("\n  Interpretation: if relative spread is at the ~1e-6% level (float/solver "
          "precision), the gauge is stable and canonicalize() is NOT the noise source. "
          "If it's a meaningfully larger fraction of a percent (or more), repeated "
          "canonicalize() alone is measurably perturbing the state.", flush=True)


def main() -> None:
    samples = ex.draw_samples()
    jobs = [{"sample": s, "L_pp": samples[s]["L_pp"]} for s in SAMPLES]

    print(f"{len(jobs)} gauge-stability checks queued (samples {SAMPLES}), in parallel.\n", flush=True)
    with mp.Pool(processes=len(jobs)) as pool:
        for out in pool.imap_unordered(run_one, jobs):
            try:
                report(out)
            except Exception:
                print(f"!! report failed for sample {out['sample']}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
