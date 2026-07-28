# Lindbladian steady states — project notes

Finite-chain MPS/TEBD code for finding Lindblad steady states, used to study
**strong-to-weak spontaneous symmetry breaking (SWSSB)** in a parity-symmetric
dissipative spin chain.

## The physics question

The chain has a **strong** Z₂ symmetry P = Z₁…Z_N: every jump operator
individually commutes with P (not merely the superoperator), so the Liouvillian
block-decomposes by two charges (q_L, q_R) and each block evolves independently.

SWSSB is the phase where the steady state stops being strongly symmetric while
remaining weakly symmetric. No linear observable detects it — `Tr[ρ X_i X_j]`
misses it entirely — so the diagnostic is quadratic in ρ:

    R(i,j) = Tr[ρ A ρ† A†] / Tr[ρ† ρ],    A = X_i X_j†

**Long-range order — R(i,j) not decaying with |i−j| — is the SWSSB signal.**

Model (purely dissipative, H = 0), bond jumps on every nearest-neighbour bond
at rate 1:

    L   = X_a X_{a+1} (1 − Z_a Z_{a+1})          Hermitian, dark on |0…0⟩
    L'  = X_a X_{a+1} (1 − Z_a)(1 − Z_{a+1})     non-Hermitian, dark on |0…0⟩
    L'' = random on the 8-dim commutant of Z⊗Z, ‖L''‖ = ε (currently 0.2)

A two-site operator commutes with the *global* P iff it commutes with Z⊗Z on
its own bond — a purely local condition, which is what makes the random L''
sampling in `models.random_zz_commuting_operator` legitimate.

Two initial states, both pure, both in the **(+,+) sector** (asserted at every
size — Néel has an even number of ones for N ∈ {4,8,12,16,20}):

- `zero` = |0…0⟩⟨0…0| — a **dark state** of the baseline, so baseline R ≡ 0 and
  all signal comes from L''. This is the clean arm.
- `neel` = |0101…⟩⟨0101…| — not dark. Under the baseline it flows *to* the dark
  state (R = 3.7e-21 at N=8), confirming the baseline (+,+) steady state is
  unique.

Running both probes steady-state uniqueness within the sector.

## ⚠️ The one trap that matters: χ-limited vs cutoff-limited

**Before believing any correlator from this pipeline, compare
`result['final_bond_dims']` against `chi_max`.**

If `max(final_bond_dims) == chi_max`, the run is **χ-limited**: the bond
dimension cap is binding, the state is not converged, and the number is a
*lower bound*, not a result. If the max is comfortably below `chi_max`, the
1e-10 cutoff bound first and the state is genuinely represented.

This is not a subtlety — it inverted the conclusion of this study. At N=20,
sample 0, `zero` start:

| χ | R(N=16) | R(N=20) | ratio | limited by |
|---|---------|---------|-------|------------|
| 32 | 3.86e-4 | 1.82e-4 | 0.471 | χ |
| 64 | 5.90e-4 | 3.49e-4 | 0.591 | χ |
| **128** | **7.12e-4** | **7.04e-4** | **0.989** | **cutoff** (max bond 55 / 71) |

At χ ≤ 64 the correlator shows a convincing exponential decay in separation and
a convincing decay with system size. **Both are entirely fabricated by
truncation.** At χ=128 the profile R(i,r) is flat to ~0.1% across every
separation (N=20: 7.14e-4 at r=6 → 7.09e-4 at r=19), and R is independent of N.

Do not quote χ=32 or χ=64 numbers as converged. Do not conclude "no SWSSB" from
them.

## Current status of the results

**Established** (ε=0.2, sample 0 only, `zero` start): R is flat in both
separation and system size at converged bond dimension — consistent with SWSSB.

**Not yet established**: genericity. Only one of the ten L'' samples has been
run at χ=128. The 10-sample × {N=12,16,20} × χ=128 run (~5.8 h at 6 workers)
is what would turn this into a result. Also untested: any ε other than 0.2, and
whether `neel` agrees at converged χ.

The `zero`-vs-`neel` differences seen at χ=32 (up to 8e-2 at N=20) are
**numerical, not physical** — they are ~1e-14 at N=4 where truncation is exact,
and grow in lockstep with the discarded weight.

## Layout

    lindblad_mps/
      vectorize.py    vec conventions, Liouvillian generator, bond gates.
                      Local-vec = each site's (bra,ket) pair kept together and
                      interleaved, so factorized operators act as Kronecker chains.
      mps.py          MPS over vectorized ρ; site dim = local_dim² = 4.
      tebd.py         Trotter TEBD driver + find_steady_state (annealed dt).
      observables.py  Renyi-2 correlator: dense / local-vec / MPS-native.
      exact.py        dense reference for small N.
      models.py       the jump operators and random parity-commuting sampling.
      diagnostics.py  convergence diagnostics.
      blas.py         BLAS thread control — see below.

    experiments/
      renyi2_swssb.py            main study (ε=0.2, N ∈ {4,8,12,16,20}, χ=32)
      renyi2_swssb_chi64.py      χ=64 grid + χ ∈ {96,128} sweep extension
      renyi2_swssb_probe_n16.py  single N=16, χ=128 probe
      results/                   pickles + PNGs, and _cache/ (see below)

## Running things

    uv run pytest                                  # 59 tests, ~4 s
    uv run python experiments/renyi2_swssb.py      # ~43 min at 6 workers

Experiments parallelise across `N_WORKERS` processes (6). Jobs are queued
longest-first. **Every completed run is pickled into `experiments/results/_cache/`**
keyed by `kind_label_init_N_chi`, and reused on any later run — so re-running a
study, extending it, or resuming after a crash costs only the new work. The
cache is committed, so it survives a fresh clone.

`run_job` catches exceptions per job rather than letting them propagate: one
failure must not abort the pool. (It did once, losing 43 min of completed runs.)

## Performance notes

**BLAS threading is pinned to 1 thread inside TEBD** (`blas.limit_threads`,
via `blas_threads=1` on `evolve`/`find_steady_state`). This is a ~4×
end-to-end speedup, not a micro-optimisation: threaded OpenBLAS is
*pathologically slow* on the small LAPACK factorisations TEBD produces — a
128×128 complex SVD takes 38 ms threaded vs 6.1 ms serial on this hardware.
Environment variables can't fix it from inside the package (OpenBLAS reads them
at library load, i.e. at `import numpy`), hence threadpoolctl.

~95% of TEBD flops are the SVD in `apply_two_site_gate`. It uses
`_robust_svd`, which falls back from LAPACK `gesdd` to the slower but more
robust `gesvd` on non-convergence — this fires rarely but does fire (once in a
1500-step N=20 run) and used to crash the study.

Use `np.tensordot`, not `np.einsum`, in hot paths: einsum without
`optimize=True` falls through to a scalar C loop instead of dispatching to
BLAS, at identical flop count.

## Conventions

Docstrings use an Input/Output format — match it. Physics constants and
experiment configuration live at module top level in the experiment scripts,
not in argparse.
