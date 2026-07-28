"""Probe: is N=16 cutoff-limited or chi-limited at chi=128?

At N=20 the chi sweep showed a qualitative change at chi=128: for every
chi <= 96 the final bond dimension saturated the cap (chi is the binding
constraint, so the run is not converged), whereas at chi=128 the state settled
at max bond 71 under the 1e-10 cutoff -- the first setting where the bond
dimension was genuinely sufficient.

The open question is the N-scaling ratio R(20)/R(16), which cannot be trusted
while both sizes are chi-limited (it is biased downward, more strongly at the
larger size). This runs the single missing point -- one sample, N=16,
chi=128 -- to establish two things:

  1. whether N=16 is also cutoff-limited at chi=128 (final max bond < 128),
     in which case a full chi=128 study would be worth its ~5.6 h; if N=16 is
     still chi-limited then chi=192+ is needed instead and the big run would
     be wasted;
  2. a first converged-vs-converged ratio R(20)/R(16) at chi=128, which is
     the actual plateau-vs-decay discriminator.

Dispatched through renyi2_swssb.run_job, so it shares that module's TEBD
settings and cache.

Output (experiments/results/):
    renyi2_swssb_probe_n16.pkl -- the run record plus the chi=128 ratio.
"""

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import renyi2_swssb as ex

PROBE_N = 16
PROBE_CHI = 128
PROBE_SAMPLE = 0
PROBE_INIT = "zero"


def main() -> None:
    os.makedirs(ex.CACHE_DIR, exist_ok=True)
    samples = ex.draw_samples()
    job = {
        "kind": "probe", "label": f"chi{PROBE_CHI}", "sample": PROBE_SAMPLE,
        "L_pp": samples[PROBE_SAMPLE]["L_pp"], "N": PROBE_N,
        "init": PROBE_INIT, "chi_max": PROBE_CHI,
    }
    print(f"probe: {ex.job_key(job)}", flush=True)
    print(f"  sample {PROBE_SAMPLE} (seed {samples[PROBE_SAMPLE]['seed']}), "
          f"eps={ex.EPSILON}, cutoff={ex.CUTOFF}", flush=True)
    print("  expect 1.5-2.5 h depending on whether chi or the cutoff binds\n",
          flush=True)

    t0 = time.perf_counter()
    out = ex.run_job(job)
    if "error" in out:
        print(f"FAILED after {(time.perf_counter()-t0)/60:.1f} min:\n{out['error']}",
              flush=True)
        return

    res = out["result"]
    bd = res["final_bond_dims"]
    limited_by = "chi" if max(bd) >= PROBE_CHI else "cutoff"

    print(f"finished in {(time.perf_counter()-t0)/60:.1f} min "
          f"({'cached' if out['cached'] else 'computed'})", flush=True)
    print(f"  R({res['i']},{res['j']}) = {res['correlator']:.6e}", flush=True)
    print(f"  1-|overlap|            = {1-res['final_overlap']:.2e}", flush=True)
    print(f"  max discarded weight   = {res['max_discarded_weight']:.2e}", flush=True)
    print(f"  final max bond         = {max(bd)}  ->  limited by {limited_by.upper()}",
          flush=True)
    print(f"  final bond dims        = {bd}", flush=True)

    # converged-vs-converged ratio against the existing N=20 chi=128 run
    ext = pickle.load(
        open(os.path.join(ex.RESULTS_DIR, "renyi2_swssb_chi_extended.pkl"), "rb")
    )
    r20 = ext["results"].get(PROBE_CHI)
    ratio = None
    if r20 is not None:
        ratio = r20["correlator"] / res["correlator"]
        print(f"\n  R(N=20, chi=128) = {r20['correlator']:.6e} "
              f"(max bond {max(r20['final_bond_dims'])})", flush=True)
        print(f"  R(N=16, chi=128) = {res['correlator']:.6e}", flush=True)
        print(f"  ratio R(20)/R(16) = {ratio:.3f}   "
              f"[chi=32: 0.471, chi=64: 0.591 for this sample]", flush=True)
        if ratio > 0.85:
            verdict = "PLATEAU-like -- consistent with SWSSB surviving"
        elif ratio < 0.75:
            verdict = "still decaying"
        else:
            verdict = "intermediate -- inconclusive"
        print(f"\n  {verdict}", flush=True)

    payload = {
        "config": {"N": PROBE_N, "chi": PROBE_CHI, "sample": PROBE_SAMPLE,
                   "init": PROBE_INIT, "epsilon": ex.EPSILON, "cutoff": ex.CUTOFF},
        "description": samples[PROBE_SAMPLE]["description"],
        "result": res,
        "limited_by": limited_by,
        "N20_chi128": r20,
        "ratio_20_over_16": ratio,
    }
    path = os.path.join(ex.RESULTS_DIR, "renyi2_swssb_probe_n16.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved -> {path}", flush=True)


if __name__ == "__main__":
    main()
