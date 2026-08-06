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
  all signal comes from L''. It approaches the perturbed steady state from
  R = 0, i.e. **from below**.
- `neel` = |0101…⟩⟨0101…| — not dark. Starts far away carrying real
  correlations, so it approaches **from the other side**.

Running both is not redundancy. A unique steady state in the sector forces the
same R from either, so where they agree the value is converged from two
directions, and where they split the pair brackets the truth. Both are now run
on the full grid — see below.

## ⚠️ Three traps, and each has inverted a conclusion here

### Trap 1: χ-limited vs cutoff-limited

**Compare `result['final_bond_dims']` against `chi_max` before believing any
correlator.** If `max(final_bond_dims)` is at (or within ~10% of) `chi_max`, the
run is **χ-limited**: the cap is binding, the state is not converged, and the
number is a *lower bound*. If the max is comfortably below, the 1e-10 cutoff
bound first and the state is genuinely represented.

At N=20, sample 0, `zero` start:

| χ | R(N=16) | R(N=20) | ratio | limited by |
|---|---------|---------|-------|------------|
| 32 | 3.86e-4 | 1.82e-4 | 0.471 | χ |
| 64 | 5.90e-4 | 3.49e-4 | 0.591 | χ |
| **128** | **7.12e-4** | **7.04e-4** | **0.989** | **cutoff** (max bond 55 / 71) |

At χ ≤ 64 the correlator shows a convincing exponential decay in separation and
with system size. **Both are entirely fabricated by truncation.** Do not quote
χ=32 or χ=64 numbers as converged, and do not conclude "no SWSSB" from them.

**χ=128 is not automatically enough.** The χ sweep was run on sample 0, which
settles at bond dimension 72 of 128 — the easy end of the ensemble. Samples 1
and 9 reach 128 and 124 at N=20, so their correlators there are lower bounds.
Note also that the sweep has exactly one point that is not cap-bound (χ=128
itself), so it brackets convergence from below without ever demonstrating a
plateau; showing saturation needs χ=192 or 256 agreeing with χ=128.

### Trap 2: converged-looking but not relaxed

**Use `residual` for this. It is the only honest convergence test here** —
r = ‖ℒ|ρ⟩‖ / ‖|ρ⟩‖, which is exactly zero at a steady state of any sector and
refers to nothing else: not dt, not the initial state, not the schedule. Every
run computes it (`lindblad_mps/residual.py`, a Liouvillian MPO of bond
dimension 18 contracted in four layers, a few seconds against runs of hours).

Read it against the **dt² Trotter floor**, not against zero. TEBD converges to
the fixed point of the *Trotterized* propagator, so a fully relaxed run stops
at r = C·dt², measured at N=4 as C = 0.0202 across dt ∈ [0.005, 0.1] to three
digits. A run sitting on its floor has relaxed and is limited only by dt; a run
orders of magnitude above it has not, however still its observables look. To
tell them apart, re-run one case at half the final dt: the floor drops 4×, a
genuinely unrelaxed residual barely moves. Below ~1e-7 the number is arithmetic
noise (r is a square root of a near-total cancellation between terms of order
‖ℒ‖²).

**Cached runs from before this existed have `residual = None`, and re-running
will not fill it in** — the diagnostic changes neither the cache key nor
`run_config`, so a cached run comes back as it stands. Delete the specific
cache file to force a recompute.

The rest of this section is why that was needed.

The `converged` field is **not a convergence test** and is retained only for
continuity with old pickles. It checks `1 − |⟨ρ(t)|ρ(t+dt)⟩| < 1e-6` over *one*
Trotter step, which shrinks with dt whether or not the state is near the fixed
point, so a slowly-drifting state passes trivially.

Demonstrated by the L''=0 `neel` control at χ=128, whose exact steady state is
the dark state at **bond dimension 1**:

| N | R | final bond dims (peak) | `converged` | 1 − overlap |
|---|---|---|---|---|
| 12 | 4.94e-09 | 3 | True | −2e-16 |
| 16 | 6.29e-06 | 52 | True | 2.7e-11 |
| 20 | 5.41e-07 | 56 | True | 6.7e-12 |

N=12 genuinely relaxed. N=16 and N=20 carry an entangled bulk — bond dims small
at the chain ends, peaking mid-chain, a relaxation front that has not finished
crossing — yet report converged with 1 − overlap five orders below tolerance.
Relaxation time grows with N faster than the fixed 5 × 300-step schedule allows.

Two corollaries that run against instinct:

- **A small bond dimension can mean the opposite of "well converged."** A state
  that has not finished building its correlations has not built its entanglement
  either. Sample 5 grows 59→76→111 across N=12,16,20; sample 8 sits at
  45→48→48 and is the *least* converged run in the grid.
- **Agreeing with a product-state answer at low χ is not evidence of
  correctness.** The L''=0 `neel` artifact is *larger* at χ=128 (6.3e-06) than
  at χ=32 (9.0e-07) at N=16, because the true answer is a bond-dimension-1
  product state and hard truncation shoves the state toward it for the wrong
  reason.

### Trap 3: the infinite-system steady state is not always a density matrix

Nothing in the vectorized iTEBD constrains ρ to be positive. Measured on the
true 8-site reduced density matrix of the infinite chain
(`experiments/imps_positivity_hermiticity.py`, ε=0.2, χ=128, |neel⟩):

| | samples 0–6 | **samples 7, 8, 9** |
|---|---|---|
| λ_min | +1e-14 … +3e-12 | **−1.80e-2, −1.15e-2, −3.79e-2** |
| purity | 0.564 … 0.753 | **1.293, 1.186, 1.626** |
| negative weight | 0 | **11.2%, 7.8%, 18.8%** |

Purity above 1 is impossible for a density matrix. For samples 7, 8, 9 the
converged state is genuinely not one — reproducibly, and the defect grows with
window size. Hermiticity is fine everywhere (worst 8e-06, most at 1e-14).

**R is unaffected.** Tracked through runs that landed on either branch, R is
identical to seven digits — sample 4 gave 1.304313e-03 with purity 0.808264
*and* with purity 1.246957, drift −0.09% in both. So the factor
R_iTEBD = 1.9875 × the finite-N law is not caused by non-positivity: samples
0–6 are genuine density matrices and show the same factor (1.9842–1.9891) as
7, 8, 9 (1.9919, 1.9913, 1.9854).

#### The bug this hid behind — do not reintroduce it

Until 2026-07-31 this was *not* reproducible: successive runs disagreed about
which samples were physical (one pass said "sample 0 only", the next swapped
it to sample 6, a 30-run pass reported 20/30). That was **not** physics.

`leading_eigenpairs` called `scipy.sparse.linalg.eigs()` with no `v0`, so
ARPACK generated its own start vector from an internal Fortran RNG that
`np.random.seed()` cannot reach and whose state persists within a process
across calls. Under multiprocessing the outcome therefore tracked **pool
scheduling** — how many jobs a worker had already handled. `canonicalize()`
truncates, so a different eigenvector on a near-degenerate transfer spectrum
gives a different truncation and the evolution forks.

The signature that identified it: purity did not *drift*, it **jumped**
(sample 4, by thirds: 0.807812 → 1.246957 → 1.246957) and was stationary on
both sides of the jump, while R stayed put. Only tracking an absolute quantity
*through* a run rather than at the end exposed this.

Fixed by a deterministic `v0` (`imps.ARPACK_V0_SEED`), with regression tests in
`tests/test_imps.py::TestArpackDeterminism` — one on `leading_eigenpairs`
under a churned global RNG, one comparing gauge-invariant Schmidt spectra
across a full evolution. Repeats now agree to 0.00e+00 in both R and purity.

**Retracted with it:** an earlier claim that the stationary manifold contains
*physical* members with different R, based on sample 3 returning 5.279e-03
against 1.976e-03. That does not reproduce — three checked runs give
1.975851/1.975898/1.975851e-03 at ~0.15% drift, matching production. It came
from a run whose convergence had never been checked, because this script
records no drift diagnostics. **Do not quote an absolute quantity from a run
without one.**

## Diagnostics every run reports

`run_steady_state_correlator` records, at negligible cost (~50 ms against runs
of many minutes):

- `stage_correlators` — R at the end of each dt stage. **This is the real
  convergence test.** Read the *shape*: the L''=0 control fell monotonically by
  an order of magnitude per stage, whereas converged runs oscillate as dt
  anneals (e.g. 7.375e-04 → 7.056e-04 → 7.172e-04).
- `stage_drift`, `time_converged` — relative change over the final stage against
  `STAGE_DRIFT_TOL` (1e-2). **This threshold is mis-calibrated**: it flags 14 of
  100 runs, mostly Trotter oscillation rather than genuine drift. The boolean
  should test monotonicity, not magnitude. Until then, treat `!` as "go look at
  `stage_correlators`", not as a verdict.
- `residual`, `residual_per_bond` — ‖ℒ|ρ⟩‖/‖|ρ⟩‖ and the same divided by the
  N−1 bond terms. **The convergence test**; see Trap 2 for how to read it.
- `min_profile`, `positivity_violation` — R = Tr[(AρA)ρ] with A = X_i X_j†
  Hermitian and unitary is a trace of two positive semidefinite matrices, so
  **R ≥ 0 for any physical ρ**. A negative value measures how much positivity
  truncation has destroyed. Recorded, never raised: truncation-limited runs are
  still worth keeping, and `run_job` would otherwise bank them as failures.

The L''=0 `neel` control at χ=128 reaches −3.75e-06, so **~4e-06 is the
pipeline's false-positive floor**. All 100 sample runs are clean (0 violations).

## Current status of the results

**Established.** Genericity, both initial states, at converged bond dimension:
10 L'' samples × N ∈ {4,8,12,16,20} × χ=128 × {`zero`,`neel`} = 100 runs.

- R is **flat in system size**: R(16)/R(12) = 0.998 (`zero`), 0.996 (`neel`).
- R is **flat in separation**: nine of ten samples vary ≤3% across the outer
  half of the N=20 profile.
- The smallest correlator anywhere is 1.92e-04, **51× the control floor**.
- **N=4 is exact** — a physical-dimension-4 chain cannot exceed bond dimension
  4^min(k,N−k), so the middle-bond bound is 16 and χ cannot bind even in
  principle (drift ~1e-7). R there agrees with N=20 to ~1%, so the plateau is
  anchored on a point carrying no truncation caveat at all.
- **Steady-state uniqueness holds where the runs are converged**: the two starts
  agree to 0.00% at N=4 and 8, 0.21% at N=12, 3.75% at N=16 (worst sample).

**The reliable range is N ≤ 16.** At N=20 the two starts disagree by up to 26%,
and the disagreement is one-sided — `neel` reads higher than `zero` for 8 of 10
samples, exactly the under-relaxation signature. Neither start is converged
there at the current schedule; use the zero/neel spread as the error bar rather
than either value alone.

**Worked example — sample 8.** From `zero` it ran 2.2153, 2.2156, 2.2093,
2.0603, 1.9165 (×1e-4) across N = 4…20: flat to 0.3% up to N=12, then a 13%
fall that looked like the one sample without long-range order. It is not χ-
limited (bond 48/128). From `neel` the same sample reads **2.5976e-04** at N=20,
17% *above* the plateau, and its profile *rises* with separation where the
`zero` profile falls. Two starts straddling a flat value is a run stopped
mid-relaxation; a sample genuinely losing order would decay from both. The
converged value is the N≤12 plateau at 2.21e-04. Sample 8 has the smallest
correlator in the ensemble (13× below sample 6) because its L'' direction
couples most weakly to the SWSSB channel — which gives both a small R and a
small Liouvillian gap, hence the slowest relaxation.

**What sets R, and what gaps the Liouvillian.** In particle language (|1⟩ =
particle) the baseline is diffusion-limited pair annihilation: L takes
|01⟩ ↔ |10⟩ at amplitude 2 (symmetric hopping, D = 4) and L' = 4|00⟩⟨11| takes
|11⟩ → |00⟩. That is **A + A → ∅**, which is critical — gap ~ D(π/N)², z = 2 —
so the *unperturbed* model is gapless in the thermodynamic limit.

A generic L'' on the ZZ commutant has a ⟨11|L''|00⟩ entry: **pair creation**,
which turns it into A + A ⇌ ∅, a relevant perturbation with a finite-density
active steady state and a gap of order the creation rate. Measured across the
ten samples at N=12:

    R = q/8 exactly,   q ≡ |<11|L''|00>|²

**This is now an analytic result** (user's derivation, 2026-08-04), not a fit.
Measured R/q at χ=128, ε=0.2, over the ten samples — `zero` and `neel` agree to
every digit shown up to N=12:

| N | R/q | vs 1/8 | spread |
|---|-----|--------|--------|
| **4** (exact) | **0.125149** | **+0.12%** | 0.23% |
| 8 | 0.125431 | +0.35% | 0.32% |
| 12 | 0.125927 | +0.74% | 0.48% |
| 16 | 0.126303 | +1.04% | 1.6% |
| 20 | 0.123282 | −1.37% | 5.4% |

The old fitted constant 0.1256 was calibrated at N=12 and carries that row's
+0.7% finite-size drift; **quote 1/8, not 0.1256**. N=4 is truncation-free and
sits 0.12% above 1/8 at ε=0.2, the same size as the O(ε²) correction seen in
the infinite system (below), so the residual is a higher-order term in ε rather
than an error in the law. The N=20 row is under-relaxation, not physics.

The rank correlation with q is perfect. **The SWSSB signal is driven entirely by
the pair-creation matrix element**, and the under-relaxation follows the same
quantity (corr(log creation rate, log zero/neel gap) = −0.68): small creation ⇒
small gap ⇒ slow relaxation ⇒ wide zero/neel straddle. Sample 8 is not a sample
losing order — it has the smallest creation amplitude in the ensemble
(0.00177, 12.5× below sample 6) and its R is exactly where that law puts it.

Two consequences. The perturbed Liouvillian is **gapped**, N-independently, so
an infinite-system method is not facing a gapless generator — but the gap is
O(ε²) and the ε → 0 limit is singular, recovering the critical baseline. And R
∝ ε² through |c|² means the signal is leading-order perturbative around the
dark state, which makes the ε dependence a sharp prediction rather than an open
grid (see below). The 1/8 was suspiciously round because it is exact.

**Not yet established.** Saturation in χ (see Trap 1).
N=20 at a schedule long enough to converge — cheap in principle, since bond
dims sit at 48–72 for most samples, so relaxation and not truncation is the
bottleneck there (samples 1 and 9 excepted, which need χ > 128).

The `zero`-vs-`neel` differences seen at χ=32 (up to 16% at N=20) are
**numerical** — exactly 0.0% at N=4 where truncation is exact, growing in
lockstep with the discarded weight.

## The infinite system (iTEBD)

`experiments/imps_eps_init_grids.py`, χ=128, |neel⟩ unless stated. Bond
dimension never binds: it peaks at 29 at ε=0.2 and falls to **7–12 at
ε=0.05**, because small ε sits closer to the dark state. Small-ε runs are
therefore *cheaper per unit time*, not more expensive — the ε=0.05 grid
(47040 units × 10 samples) cost about the same wall clock as the ε=0.2 one.

**Established in the thermodynamic limit:**

- **R is flat in separation.** Sample 8 at 15015 units: R varies by 0.0001%
  over r = 20…100, R(100)/R(1) = 1.000462. This is the SWSSB long-range-order
  signature, with no finite-size caveat.
- **R is independent of the initial state.** All 10 samples agree between
  |0…0⟩ and |neel⟩ to ≤0.2%, eight of ten to 0.0%. Sample 8 reads
  R = 4.4348e-04 from *both* — five digits, from opposite directions.
- **R ∝ ε², over a 4× range in ε (16× in R), to ~0.1%:**

  | ε | predicted R(ε)/R(0.2) | measured | spread |
  |---|---|---|---|
  | 0.15 | 0.5625 | 0.56290 ± 0.00040 | 0.07% |
  | 0.10 | 0.2500 | 0.25027 ± 0.00023 | 0.09% |
  | 0.05 | 0.0625 | 0.06259 ± 0.00007 | 0.12% |

  The systematic **+0.11–0.14% excess** at all three ε is **explained**: it is
  a bias in the ε=0.2 *reference*, which is itself 0.146% below the exact law
  (see "R = q/4, settled" below). Divide out and the excess is gone.

**Schedule length scales as 1/ε², and this is load-bearing.** τ ~ 1/gap and
gap ~ pair-creation rate ~ ε², so a fixed schedule gives *less* relaxation at
smaller ε. Schedules are (0.2/ε)² × the ε=0.2 ones. Verified by control runs
at 5× extra length: sample 8 at ε=0.05 gives 2.7717e-05 at 240240 units vs
2.7714e-05 at 47040, and at ε=0.1 the two agree exactly — so the rule is
conservative, not marginal.

**Per-sample relaxation times cannot be extracted from these trajectories.**
Two estimators were tried and both failed: "last excursion beyond 1% of the
final value" gave τ(0.15)/τ(0.2) ∈ [0.33, 12.5], and a single-exponential fit
to the smoothed approach gave [0.14, 20] against a predicted 1.78; τ·rate is
not constant either. The fitting window dominates. Scale the *schedules known
to have converged* (drift ≈ 0% by thirds) instead of any fitted τ.

**R = q/4, settled.** ✅ The infinite-system correlator is

    R_iTEBD = q/4,   q ≡ |<11|L''|00>|²

analytically (user's derivation, 2026-08-04), which also **predicts the factor
of 2 against the finite-N law R = q/8**. The factor is no longer an anomaly and
the search for a numerical cause is closed — do not reopen it.

Measured R(r=100)/q from the cached grids, ten samples each:

| ε | init | R/q | vs 1/4 | spread |
|---|------|-----|--------|--------|
| 0.20 | neel | 0.249635 | −0.146% | 0.12% |
| 0.20 | zero | 0.24966 | −0.136% | 0.13% |
| 0.15 | neel | 0.249810 | −0.076% | 0.07% |
| 0.10 | neel | 0.249907 | −0.037% | 0.03% |
| 0.05 | neel | 0.249977 | −0.009% | 0.01% |

The residual is a clean **O(ε²) correction, not an error in the law**: the
relative deviation divided by ε² is −0.036, −0.034, −0.037, −0.036 — constant
across a 4× range in ε — so

    R/q = (1/4)(1 − 0.036 ε² + …),   extrapolating to 0.2500015 at ε → 0,

i.e. 1/4 to six parts in a million. That same correction is what biases the
ε=0.2 reference and produced the +0.11–0.14% "excess" in the ε-sweep table
above.

**On dimensions.** R is dimensionless and q is a rate, so the exact statement
is R = q/(4γ) with γ the baseline bond rate, which is 1 in every run here. A
prediction of the form q/(4ε²) — with ε the L'' operator norm — is excluded by
the ε sweep *independently of its prefactor*: q ∝ ε² by construction, so
q/(4ε²) is ε-independent, whereas R varies by the full 16× across the sweep,
tracking q. (At ε=0.2 it also overshoots by exactly 1/ε² = 25.)

Superseded investigation, kept only so it is not redone: the factor was
localized to the **denominator** (numerator agrees with exact dynamics to 4.4%,
purity ratio 0.531, exact N=8 purity 0.926622), and ruled out as measurement
(tiling + finite routine gives ratio 1.000), bond dimension (identical across
χ ∈ {16,32,64}), positivity (Trap 3), convergence (sample 8 joined the pack at
1.9913 once run 5× longer), an equal mixture of two orthogonal states (A has
two X's, cannot change charge sector, so numerator and purity halve together),
and transfer-matrix degeneracy (leading eigenvalue is simple in every sample —
though samples 0 and 4 carry a cluster of three near-unit eigenvalues 0.4% and
1.8% below η₁, while sample 6 has a clean gap η₂ = 0.223 and the *same*
factor).

⚠️ `iMPS.canonicalize()` does **not** truncate by default, so calling it on a
χ=32 state can regauge it up to bond dimension ~84 (exact, but the transfer
operator becomes 7056×7056). Pass `chi_max` if size matters.

## Layout

    lindblad_mps/
      vectorize.py    vec conventions, Liouvillian generator, bond gates.
                      Local-vec = each site's (bra,ket) pair kept together and
                      interleaved, so factorized operators act as Kronecker chains.
      mps.py          MPS over vectorized ρ; site dim = local_dim² = 4.
      tebd.py         Trotter TEBD driver + find_steady_state (annealed dt,
                      optional per-stage callback for convergence diagnostics).
      observables.py  Renyi-2 correlator: dense / local-vec / MPS-native.
      exact.py        dense reference for small N.
      models.py       the jump operators and random parity-commuting sampling.
      diagnostics.py  convergence diagnostics.
      residual.py     Liouvillian MPO + ||L|rho>||/|||rho>||, the absolute
                      steady-state test (see Trap 2). The MPO has bond
                      dimension r+2 = 18 for this model, so L†L would be 324 —
                      finite and small, contrary to the usual assumption that
                      squaring it is non-local. What rules out variational
                      (L†L) steady-state search here is conditioning, not
                      locality: gap(L†L) ~ gap(L)² ~ 4e-6 for sample 8.
      blas.py         BLAS thread control — see below.
      imps.py         infinite MPS (2-site unit cell, Vidal Γ-Λ form),
                      Orús-Vidal canonicalization via transfer-matrix fixed
                      points. Read its module docstring before touching
                      canonicalize(): four separate gauge bugs were found and
                      fixed there, and product-state tests cannot catch any of
                      them (at bond dimension 1 every Λ-weighting is equivalent).
      itebd.py        infinite-system Trotter driver.
      iobservables.py Renyi-2 profile + correlation length for an iMPS. The
                      forward sweep MUST use LEFT-weighted tensors and a
                      Λ_right² closure; getting it wrong silently fabricates an
                      exponential decay out of a flat correlator. Environments
                      are rescaled by a shared factor each step — exactly
                      ratio-preserving, and required: 100 unrenormalized
                      transfer steps overflow float64 when the state is far
                      from canonical mid-anneal, poisoning precisely the
                      large-r values the SWSSB claim rests on.
      ilpdo.py        locally-purified (ρ = XX†) ansatz, positive by
                      construction. Built to test the positivity hypothesis for
                      the factor of 2; that hypothesis is now dead, so this is
                      unused by the production path.
      itebd_lpdo.py   Trotter driver for the LPDO ansatz.

    experiments/
      renyi2_swssb.py            main study (ε=0.2, N ∈ {4..20}, χ=32) and the
                                 shared job/cache plumbing every other script uses
      renyi2_swssb_chi64.py      χ=64 grid + χ ∈ {96,128} sweep extension
      renyi2_swssb_probe_n16.py  single N=16, χ=128 probe
      renyi2_swssb_chi128.py     **the headline run** — 10 samples × 5 sizes ×
                                 both initial states at χ=128
      renyi2_swssb_profiles.py   re-plots from the pickles, runs no TEBD
      imps_eps_init_grids.py     **the infinite-system study** — the ε sweep
                                 {0.2, 0.15, 0.1, 0.05}, both initial states,
                                 and the long-schedule convergence controls.
                                 Grids are declared in one GRIDS dict; ε lives
                                 in run_config because the cache key cannot
                                 express it, and SUPERSEDES routes a sample to
                                 its longer re-run where one exists.
      results/                   pickles + PNGs, and _cache/ (see below)

Figures worth knowing — finite: `renyi2_swssb_size_scaling[_neel].png` (R vs N,
all samples, with the χ=32 curves behind for contrast), `renyi2_swssb_n20_
profiles[_neel].png` (full R(i,r)), `renyi2_swssb_init_comparison.png` (the
uniqueness test), `renyi2_swssb_chi_extended.png` (χ sweep + per-sample bond
dims). Infinite: `imps_finite_vs_infinite.png` (finite N vs the thermodynamic
limit, infinite point on its own broken axis), `imps_correlator_vs_epsilon.png`
(R vs ε with a FREE-exponent power-law fit — p = 1.99909 ± 0.00077),
`imps_trajectories_all.png` (every sample × every ε, raw), and
`imps_positivity_hermiticity.png`.

**Deleted 07-31, do not resurrect from old pickles.** Everything produced by
the pre-fix iTEBD — before the correlator Λ-weighting and canonicalize gauge
bugs were found — was removed, because those figures show a spurious
exponential decay and violent oscillation that is entirely measurement
artifact: `imps_swssb_infinite[_longrun]`, `imps_sample8_timescale[_long]`,
`imps_sample8_rolling_average`, plus `imps_all10_samples_neel_summary` and
`imps_all10_r50_r100_shared_axis`, which were re-plots made *before* their own
source pickles were re-run. Also removed: `imps_projected_measurement`, which
benchmarked the positivity-projection/LPDO route that the factor-of-2
investigation has since refuted. The surviving `allsamplests` cache is
post-fix, independently confirmed by its |neel⟩ values matching the
separately-computed |zero⟩ runs to five digits.

## Running things

    uv run pytest                                     # 59 tests, ~2 s
    uv run python experiments/renyi2_swssb_chi128.py  # ~2.7 h per initial state

Experiments parallelise across `N_WORKERS` processes. **Every completed run is
pickled into `experiments/results/_cache/`** keyed by `kind_label_init_N_chi`,
and reused on any later run — so re-running a study, extending it, or resuming
after a crash costs only the new work. The cache is committed, so it survives a
fresh clone.

Each entry also stores a `run_config` (dt schedule, steps, recanonicalize
interval, cutoff) which the **key does not cover**, and a mismatch forces a
recompute. Without this, lengthening the schedule — precisely the fix when a run
turns out not to be converged in time — would be silently defeated by the cache
handing back the old, shorter run under the same name.

`run_job` catches exceptions per job rather than letting them propagate: one
failure must not abort the pool. (It did once, losing 43 min of completed runs.)

## Performance notes

**BLAS threading is pinned to 1 thread inside TEBD** (`blas.limit_threads`,
via `blas_threads=1` on `evolve`/`find_steady_state`). This is a ~4×
end-to-end speedup, not a micro-optimisation: threaded OpenBLAS is
*pathologically slow* on the small LAPACK factorisations TEBD produces — a
128×128 complex SVD takes 38 ms threaded vs 6.1 ms serial. Environment
variables can't fix it from inside the package (OpenBLAS reads them at library
load, i.e. at `import numpy`), hence threadpoolctl.

~95% of TEBD flops are the SVD in `apply_two_site_gate`, and it scales viciously
in χ — measured single-threaded on the (4χ, 4χ) complex matrix: 3.2 ms at χ=32,
21.6 ms at χ=64, 51.1 ms at χ=96, **132.5 ms at χ=128**. Everything else is
noise: the per-step `state.copy()` + `overlap()` are 1.3% of a step and
`canonicalize` 0.7%. Bond dimension is the only lever that matters.

It uses `_robust_svd`, which falls back from LAPACK `gesdd` to the slower but
more robust `gesvd` on non-convergence — rare, but it fires (once in a 1500-step
N=20 run) and used to crash the study.

Use `np.tensordot`, not `np.einsum`, in hot paths: einsum without
`optimize=True` falls through to a scalar C loop instead of dispatching to
BLAS, at identical flop count.

**`N_WORKERS = 12` on a 6-core/12-thread machine.** SMT measured at 1.59×
throughput at χ=32 (12 jobs in 183.7 s vs 6 in 145.7 s) and ~1.46× at χ=128
where the larger SVDs press harder on cache. Memory is not a constraint
(~20 MB/worker). Drop to 6 on a machine without SMT.

**Cached `seconds` are not comparable across machines.** The older entries were
written on a 6-core laptop; the current desktop is **1.86×** faster per core
(measured: six N=20/χ=32 jobs, 144-145 s vs 260-289 s cached). Divide
cache-derived estimates accordingly.

**Correlators are not comparable across machines either.** At χ=32 the same run
differs 5–12% between machines — far more than the ~1% quoted for run-to-run
scatter on one machine — because truncated non-unitary evolution amplifies
different BLAS summation orders over 1500 steps. Keep a study on one machine;
don't alias foreign cache entries into a grid to save time.

## An untested optimisation

The converged N=20 state needs bond dimension ~71, but the run operates near
128 throughout, because `apply_two_site_gate` truncates locally and
non-variationally and `canonicalize` only runs every 10 steps. If more frequent
canonicalization holds bonds near their true value the SVD cost drops by
(71/128)³ ≈ 6×, and truncation gets *more* accurate, not less. Untested; note
that changing `RECANON_EVERY` correctly invalidates the whole cache.

## Conventions

Docstrings use an Input/Output format — match it. Physics constants and
experiment configuration live at module top level in the experiment scripts,
not in argparse.
