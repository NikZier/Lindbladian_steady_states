"""R(t) for the CORRECTED iTEBD, against exact ED dynamics and the finite chain.

Why
---
The earlier R(t) figures were made from the buggy evolution -- before the two
correlator/gauge bugs were found (wrong Lambda-weighting in the measurement
sweep, and canonicalize() leaving per-site tensors non-Vidal). Those plots
showed violent oscillation, sign flips and apparent decay, essentially all of
which was measurement artifact. This redraws the same picture from the
corrected runs and adds two references the earlier version lacked:

  - EXACT dense-ED dynamics at N=4. CLAUDE.md certifies N=4 as exact for this
    model (a 4-site chain of physical dimension 4 cannot exceed bond dimension
    4^min(k,N-k), so the middle-bond bound is 16 and chi never binds even in
    principle). Integrating vec(rho) under the full Liouvillian therefore
    gives R(t) with no truncation of any kind -- an absolute reference for
    the transient shape and the plateau value.
  - the converged finite-chain value at N=20.

and shows the iTEBD/finite RATIO explicitly, which is where the unexplained
universal factor of ~2 lives (1.965-2.012 for 8 of 10 samples, established to
be independent of positivity and of the measurement).

Caveat worth reading off the figure rather than forgetting: the exact curve is
a 4-site chain measured at separation 2, the iTEBD is the infinite chain at
separation 100, and the finite line is N=20 at separation 10. They are only
directly comparable to the extent R is flat in separation and in system size
-- which is exactly the SWSSB claim under test, so the comparison is
meaningful but is not an identity.

Rolling average uses a deliberately SMALL time window: the corrected
trajectories are far quieter than the buggy ones (0% drift, 0.00% spread over
r=20..100), so heavy smoothing is neither needed nor wanted.

Outputs (experiments/results/):
    imps_corrected_dynamics.png
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.linalg

import renyi2_swssb as ex
from lindblad_mps import exact, models, observables, vectorize

SOURCES = [
    "imps_all_samples_neel_timescale.pkl",       # samples 8, 6, 0
    "imps_remaining_samples_neel_timescale.pkl",  # samples 1,2,3,4,5,7,9
]
R_PLOT = 100          # the long-range separation from the iTEBD trajectories
ROLLING_WINDOW = 100.0  # evolved-time units -- small on purpose, see docstring
EXACT_N = 4           # the one size that is exact by construction for this model
N_EXACT_POINTS = 90   # log-spaced sample times for the ED curve


def load_runs() -> dict:
    runs = {}
    for name in SOURCES:
        path = os.path.join(ex.RESULTS_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                runs.update(pickle.load(f)["runs"])
    return dict(sorted(runs.items()))


def finite_reference(sample: int) -> float | None:
    for name in (f"long_sample{sample}_neel_N20_chi128.pkl",
                 f"chi128_sample{sample}_neel_N20_chi128.pkl"):
        path = os.path.join(ex.CACHE_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)["result"]["correlator"]
    return None


def exact_dynamics(L_pp: np.ndarray, times: np.ndarray, N: int = EXACT_N) -> np.ndarray:
    """Exact R(t) by integrating vec(rho) under the full Liouvillian.

    No MPS, no truncation, no Trotter error: builds the dense generator,
    exponentiates it once per time step via an eigendecomposition, and
    evaluates the Renyi-2 correlator on the resulting density matrix. Only
    tractable because N=4 makes the generator 256x256.

    Input: L_pp, the sample's perturbation; times, the times to evaluate at; N.
    Output: array of R(t), same length as times.
    """
    L2_terms = ex.build_L2_terms(L_pp)
    jump_ops = exact.build_jump_operators(L2_terms, [], N)
    gen = vectorize.liouvillian_generator([], jump_ops, d=2 ** N)

    kets = [np.array([1, 0], dtype=complex) if k % 2 == 0
            else np.array([0, 1], dtype=complex) for k in range(N)]
    rho0 = np.array([[1.0]], dtype=complex)
    for k in kets:
        rho0 = np.kron(rho0, np.outer(k, k.conj()))
    v0 = vectorize.vec(rho0)

    # Diagonalize once, then every time point is a cheap re-exponentiation.
    w, V = np.linalg.eig(gen)
    Vinv = np.linalg.inv(V)
    c = Vinv @ v0

    i, j = N // 4, (3 * N) // 4
    out = np.empty(len(times))
    for idx, t in enumerate(times):
        v = V @ (np.exp(w * t) * c)
        rho = vectorize.unvec(v, 2 ** N)
        tr = np.trace(rho).real
        if abs(tr) > 1e-300:
            rho = rho / tr
        out[idx] = observables.renyi2_correlator_dense(rho, models.X, i, j, N)
    return out


def rolling(t: np.ndarray, v: np.ndarray, window: float) -> np.ndarray:
    """Time-windowed rolling mean (NaN-robust), matching the earlier figures."""
    half = window / 2.0
    out = np.empty_like(v)
    lo = hi = 0
    n = len(t)
    for k in range(n):
        while lo < n and t[lo] < t[k] - half:
            lo += 1
        while hi < n and t[hi] <= t[k] + half:
            hi += 1
        out[k] = np.nanmean(v[lo:hi])
    return out


def main() -> None:
    runs = load_runs()
    samples = sorted(runs)
    print(f"loaded {len(samples)} samples: {samples}", flush=True)

    plt = ex._mpl()
    fig, axes = plt.subplots(len(samples), 2,
                             figsize=(13, 2.6 * len(samples)), squeeze=False)

    sample_objs = ex.draw_samples()
    for row, s in enumerate(samples):
        traj = runs[s]["trajectory"]
        t = np.array([p["t"] for p in traj])
        v = np.array([p["R"][R_PLOT] for p in traj])
        roll = rolling(t, v, ROLLING_WINDOW)
        ref = finite_reference(s)

        t_ex = np.logspace(np.log10(max(t.min(), 0.05)), np.log10(t.max()), N_EXACT_POINTS)
        try:
            r_ex = exact_dynamics(sample_objs[s]["L_pp"], t_ex)
        except Exception as err:  # never let one sample kill the figure
            print(f"  sample {s}: exact dynamics failed ({err})", flush=True)
            r_ex = np.full_like(t_ex, np.nan)

        ax = axes[row][0]
        ax.plot(t, v, "-", color="tab:red", lw=0.5, alpha=0.25, label=f"iTEBD r={R_PLOT} (raw)")
        ax.plot(t, roll, "-", color="tab:red", lw=1.8,
                label=f"iTEBD rolling mean ({ROLLING_WINDOW:.0f} t.u.)")
        ax.plot(t_ex, r_ex, "--", color="tab:blue", lw=1.5,
                label=f"exact ED, N={EXACT_N} (r=2)")
        if ref:
            ax.axhline(ref, color="black", lw=1.4, label="finite N=20 (r=10)")
        ax.set_xscale("log")
        ax.set_xlabel("evolved time"); ax.set_ylabel(rf"$R(0,{R_PLOT})$")
        ax.set_title(f"sample {s}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=6, loc="best")

        ax = axes[row][1]
        if ref:
            ax.plot(t, v / ref, "-", color="tab:red", lw=0.5, alpha=0.25)
            ax.plot(t, roll / ref, "-", color="tab:red", lw=1.8, label="iTEBD / finite")
            ax.plot(t_ex, r_ex / ref, "--", color="tab:blue", lw=1.5, label="exact N=4 / finite")
            ax.axhline(2.0, color="grey", ls=":", lw=1.4, label="2.0")
            ax.axhline(1.0, color="black", ls="--", lw=1.0, label="1.0")
            ax.set_ylim(0, 3.2)
        ax.set_xscale("log")
        ax.set_xlabel("evolved time"); ax.set_ylabel("ratio to finite N=20")
        ax.set_title(f"sample {s}: the factor", fontsize=10)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=6, loc="best")

    fig.suptitle("Corrected iTEBD dynamics vs exact ED and the finite chain "
                 rf"($\epsilon={ex.EPSILON}$, |neel>, $\chi=128$)", y=0.999)
    fig.tight_layout()
    path = os.path.join(ex.RESULTS_DIR, "imps_corrected_dynamics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}", flush=True)

    # Numbers behind the figure, for the record.
    print(f"\n{'s':>2} {'iTEBD plateau':>14} {'exact N=4 plateau':>18} "
          f"{'finite N=20':>12} {'iTEBD/finite':>13} {'exactN4/finite':>15}")
    for s in samples:
        traj = runs[s]["trajectory"]
        t = np.array([p["t"] for p in traj])
        v = np.array([p["R"][R_PLOT] for p in traj])
        late = float(np.nanmedian(v[t > 0.7 * t.max()]))
        ref = finite_reference(s)
        try:
            ex_late = float(exact_dynamics(sample_objs[s]["L_pp"], np.array([t.max()]))[0])
        except Exception:
            ex_late = float("nan")
        print(f"{s:>2} {late:>14.4e} {ex_late:>18.4e} "
              f"{ref if ref else float('nan'):>12.4e} "
              f"{late/ref if ref else float('nan'):>13.3f} "
              f"{ex_late/ref if ref else float('nan'):>15.3f}")


if __name__ == "__main__":
    main()
