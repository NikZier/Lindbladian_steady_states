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

## ⚠️ The traps, and each has inverted a conclusion here

Traps 1–3 are general; 4 and 5 were found in the second model's section below
and are documented there. **Trap 5 is the one that has cost the most** — it
invalidated an entire published-in-this-file conclusion — so read it even if you
are only working on the first model.

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

⚠️ **And read the residual against the gap of the mode still moving, not only
against the dt² floor.** r ≈ gap × δ, so a slow mode makes a tiny residual
compatible with a state far from the fixed point: in the second model, sample 8
at N=20 sat at r = 5.1e-5 with dt² = 2.5e-5 — apparently on the floor — while
its slow mode had rate q = 1.8e-3, putting it a few percent away in norm and
**19× off in R**. Dividing r by the gap is one line and it is the difference
between the two readings.

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

## Second model: the driven classical circuit (2026-08-11, retracted 08-13)

`experiments/renyi2_drift_annihilation.py`, `models.classical_drift_annihilation_
jump_operators`. A purely classical two-site circuit — biased hopping plus pair
annihilation — in Lindblad form, one jump operator per classical transition:

    L_R = sqrt(p) |01><10|    L_L = sqrt(1-p) |10><01|    L_A = |00><11|

at p = 0.8, rates 1, same ε = 0.2 and the **same ten L'' seeds** as above, so
sample-by-sample comparison is model-vs-model rather than draw-vs-draw.

**The first model IS this one at p = 1/2, hop_rate = 8, annihilation_rate = 16**
— for every purpose this study measures. The two Lindbladians are genuinely
different: L = XX(1−ZZ) = 2(|01><10| + |10><01|) is *one* jump operator, so
L ρ L† carries interference terms the two-separate-operator embedding replaces
with dephasing, and the bond generators differ in exactly two entries (the
|01><10| ↔ |10><01| coherence, magnitude 4). But the effect on the sector
steady state is ~1e-6 in norm and on R is **1e-6 relative, sample by sample**.
Do not treat the models as incomparable on that basis — an earlier note here
did, and it was wrong.

Shared, and what makes the comparison legitimate: strong Z₂ parity (particle
number mod 2, since annihilation removes pairs), the dark vacuum |0…0>, the
bond-local ZZ-commuting form, hence the same random-L'' ensemble.

### ⚠️ RETRACTED 2026-08-13: "R decays exponentially, no SWSSB here"

**This model has SWSSB.** The finite-N grid below measures a transient, not a
steady state. The infinite-system run (`imps_drift_eps_grids.py`, next section)
gives R flat to r = 100 for **all 10 samples at all four ε ∈ {0.2, 0.15, 0.1,
0.05}** — 40 of 40 runs, R(100)/R(1) ∈ [0.9999, 1.033].

What was wrong: **evolved time**. The finite schedule is
300 × (0.1+0.05+0.02+0.01+0.005) = **55.5 time units**, and this model needs
~12/q ≈ 540–6800 units at ε=0.2. The first model tolerated the same schedule
because its baseline rates are 4–16 rather than 1, buying ~16× more relaxation
per unit time — the rate mismatch noted above for ε has the same consequence
for the *schedule*, and that consequence was missed.

The retracted claim, kept so it is not re-derived: the N=20 profile falls with
step ratio 0.822, 0.806, 0.799, 0.794, 0.790, 0.788, 0.786, 0.785, 0.787,
0.794, 0.835 across sep 1→12 — constant over twelve sites and a decade in R,
R² = 0.985–0.9996 for exponential fits at N=16 and N=20, giving ξ = 4.11…7.38
with corr(log q, log ξ) = +0.887. **All of it is a partially-relaxed state.** A
clean constant step ratio over a decade is not evidence of a steady-state
exponential; a relaxation front crossing the chain produces one too.

The one diagnostic that pointed at it was ignored: ξ correlated with q, and q
is the *relaxation rate*. That is the signature of a time-dependent profile,
not of a length scale.

Two numbers show the size of the error. Infinite-vs-finite at ε=0.2, per
sample, ordered by q:

    s          q  R_inf (flat)  R_fin (N=20)  inf/fin
    8  1.774e-03    7.0381e-03    3.6978e-04    19.0x
    7  3.988e-03    1.6031e-02    8.6075e-04    18.6x
    4  5.222e-03    2.0550e-02    1.2087e-03    17.0x
    0  5.680e-03    2.1997e-02    1.3366e-03    16.5x
    2  5.619e-03    2.2015e-02    1.4685e-03    15.0x
    3  7.923e-03    3.0615e-02    2.1536e-03    14.2x
    1  1.039e-02    4.0665e-02    3.1570e-03    12.9x
    9  1.393e-02    5.4316e-02    5.0793e-03    10.7x
    5  1.422e-02    5.4685e-02    5.3329e-03    10.3x
    6  2.214e-02    8.4266e-02    1.3604e-02     6.2x

**The discrepancy is monotone in q**, i.e. in 1/τ. A finite-size effect would
not order itself by the pair-creation rate; under-relaxation must.

And the apparent ξ was never a decay length of R. The converged profile is
R(r) = R_∞ + A e^{−r/ξ_tm} with ξ_tm *small*: sample 9 at ε=0.2 has ξ_tm = 1.34
and rises 0.053909 → 0.054310 over r = 1…10, then sits flat to r = 100. The
transfer matrix's ξ is the **approach length to the plateau**, not a decay of
it — so a finite ξ from `iobservables.correlation_length` is fully compatible
with long-range order, and reading it as a decay length inverts the physics.

### The R = q/8 law generalizes — it is R = 2q/γ_A

⚠️ Corrects an earlier claim here that "there is no such law". There is; the
evidence against it (4.3% sample scatter at N=4) was an O(ε²) correction, not a
breakdown. At ε = 0.01 the scatter is **0.008%**, and 0.008% × (0.2/0.01)² =
3.2% ≈ the 4.3% seen at ε = 0.2. R ∝ q holds essentially exactly; ε = 0.2 is
simply a *relatively* stronger perturbation here (baseline rates ~1, against
4–16 in the first model).

Measured on the exact N=4 sector state, F ≡ R·γ_A/q as ε → 0, at p = 1/2:

    hop   ann   hop/ann    eps=0.2    eps=0.05    eps=0.01
    1     1       1.000    2.010014   2.000846    2.000034
    8    16       0.500    2.000926   2.000059    2.000002
    1    16       0.062    2.002542   2.000160    2.000006
    16    1      16.000    2.012297   2.000840    2.000034

    R = 2q/γ_A   exactly at p = 1/2, γ_A the PAIR-ANNIHILATION rate,
                 independent of the hop rate over a 256x range in hop/γ_A.

So **the first model's 1/8 is 2/16 — a statement about its annihilation rate
alone.** The hopping rate, hence the diffusion constant, drops out entirely.
The first model reproduces F = 2.000002 at ε = 0.01 directly, which also
validates `sector_steady_state` against the analytic result.

The bias breaks both properties: F is no longer 2, and it acquires a hop-rate
dependence (a bias velocity gives a second dimensionless ratio v/γ_A).

    p         0.50     0.60     0.70     0.80     0.90     1.00
    F (1,1)   2.0000   2.3226   2.4385   2.3415   2.0889   1.7501
    F (8,16)  2.0000   2.2995   2.4757   2.5070   2.4108   2.2222

R/q at p=0.8, rates 1 is 2.3615 against the first model's 0.12515 — 18.9×,
which decomposes as 16.2× rate normalization and 1.17× bias, nothing anomalous.

⚠️ The R/q fall with N below is **relaxation, not a finite ξ** (see the
retraction above). Reproduced only so the numbers are not re-measured and
re-misread:

    N          4       8      12      16      20
    mean R/q   2.361   0.839   0.513   0.387   0.308
    spread     4.3%   29%     59%     93%    132%

Both the fall and the growing spread are the fixed 55.5-unit schedule buying
less and less relaxation as N grows, with the per-sample rate q setting how
much each one loses. N=4 is the only row that means anything, because it is the
only one that relaxes inside the schedule.

### The infinite system settles it: R = 4q/γ_A (2026-08-13)

`experiments/imps_drift_eps_grids.py`, χ=128, p=0.8, |0…0⟩, 10 samples ×
ε ∈ {0.20, 0.15, 0.10, 0.05}, 46 jobs / 576 240 time units / ~3.9 h on 12
workers. **R is flat in separation at every ε** — this is the SWSSB
long-range-order signature with no finite-size caveat:

    ε      n    R(100)/R(1) range     peak bond
    0.20   10   0.99954 – 1.00754     29–50 / 128
    0.15   10   0.99998 – 1.00394     27–40 / 128
    0.10   10   1.00003 – 1.01723     19–37 / 128
    0.05   10   0.99989 – 1.03316     10–25 / 128

Truncation never binds (worst case bond 50 of 128), so unlike the finite grid
this is not a lower bound.

**The amplitude law, restricting to runs with |drift| < 1%:**

    ε      n    mean R/q    spread
    0.20   10   3.887955    1.48%
    0.15   10   3.937483    0.84%
    0.10    8   3.973188    0.42%
    0.05    7   3.994284    0.10%

    fit  R/q = 4.0014 (1 − 0.709 ε²)     ε→0 intercept 4.0014, i.e. 4 to 0.04%

    R_iTEBD = 4q/γ_A

which **reproduces the first model's R = q/4 exactly** (γ_A = 16 there:
4/16 = 1/4), so the two models share one infinite-system law as well as one
finite-N law. The O(ε²) coefficient is −0.709 against the first model's −0.036,
a ratio of 19.7 ≈ the 16× rate normalization — the same correction measured in a
model whose baseline rates are 16× smaller.

Note this makes the infinite law 4q/γ_A against the finite N=4 exact
F = 2.3415 at p = 0.8. At p = 1/2 the finite law is 2q/γ_A and the ratio is
exactly 2, matching the first model's q/8 → q/4. At p = 0.8 it is 1.71, so
**the factor of 2 is a p = 1/2 statement**; do not assume it away from there.

**Convergence, four independent ways** (this is the claim that overturns a
documented conclusion, so it is not resting on one test):

- **Two starts, opposite directions.** |neel⟩ vs |0…0⟩ at ε=0.2 on samples
  8 / 3 / 6 agree to **0.00–0.11%** at r = 1, 20 and 100 — including sample 8,
  the slowest relaxer and the sample the retracted conclusion rested on.
- **3×-length control.** Sample 4 at 8820 units reproduces its 2940-unit value
  to **six digits** (−0.00% at r = 1, 20, 100), so TARGET_GAP_TIMES = 12 is
  enough and every factor-1 job in the grid is sound.
- **L''=0 from |0…0⟩ returns R = 0.000e+00 at bond 1** — the exact dark vacuum.
  All 40 production runs start from |0…0⟩, so that is their floor.
- Peak bond dimension 10–50 against a cap of 128.

⚠️ **The L''=0 |neel⟩ control does NOT converge** (R = 2.9e-4 … 5.1e-4,
comparable to the grid's smallest signal 4.4e-4) and this is expected, not a
pipeline fault: at L''=0 the model is diffusion-limited pair annihilation,
which is *critical*, so no fixed schedule relaxes it from a non-dark state.
A |neel⟩-started run therefore has no usable control floor here — a second
reason the grid runs from |0…0⟩.

**Cost note, opposite to intuition.** |0…0⟩ is 116× cheaper than |neel⟩ at
ε=0.05 (bond 20 and 34 ms per time unit, against bond 128 — capped — and
3623 ms), because the vacuum is dark so the state never leaves its
low-entanglement neighbourhood. The |neel⟩ transient at small ε is also violent
enough to drive R *negative* (truncation destroying positivity). Per-ε cost is
then roughly flat: 16× more time units at ε=0.05 against ~17× cheaper steps.

Schedules are sized **per sample** as 12/q rounded up to whole 2940-unit copies,
not uniformly per ε: q spans 12.5× across the ensemble, so a uniform schedule
over-runs sample 6 tenfold while leaving sample 8 unrelaxed. Since q ∝ ε² this
reproduces the 1/ε² scaling automatically rather than hard-coding it.

### Quality of this grid — it is *worse* than the first model's, not better

⚠️ This section previously read "better than the first model's, on every axis".
Every item below is still measured correctly; the *inference* from them was
wrong, and it is instructive exactly because each one looks like a convergence
test and none of them is one.

- **N=4 is exact and matches**: `sector_steady_state` solves the (+,+) block
  null space densely (`exact.steady_state` refuses — the strong symmetry makes
  the global zero eigenvalue degenerate). TEBD agrees to **0.0002%** worst case,
  and the L''=0 steady state comes back as the pure dark vacuum (purity
  1.000000, R = −3e−16). **But N=4 relaxes inside 55.5 units and N=20 does
  not**, so agreement at N=4 licenses the *pipeline*, never the schedule at
  larger N. This is Trap 2's structure exactly: relaxation time grows with N.
- **The zero-vs-neel spread (≤0.08% to N=16, ≤0.68% at N=20) is not evidence
  of relaxation** — see Trap 5. Both starts equilibrate locally within a few
  time units and then crawl along the *same* slow manifold at rate ~q, agreeing
  with each other the whole way while both sit ~14× below the fixed point.
- **The residual does not rescue it either.** Sample 8's 5.1e-5 at N=20 was
  read as "≈ the dt² = 2.5e-5 Trotter floor". But the slow mode has rate
  q = 1.8e-3, so r ≈ q·δ puts the state a few percent from the fixed point in
  norm — and R, being quadratic in ρ, is off by 19×. **Never read a residual as
  a floor without dividing by the gap of the mode that is still moving.**
- **Control floor is 7.09e-10**, ~5600× below the first model's 4e-06. Real,
  and irrelevant to the failure: a floor bounds the *smallest believable
  signal*, not the distance to the steady state.
- **Truncation is the binding constraint instead**, and much harder than
  before: N=20 needs χ > 128 for 9 of 10 samples (the first model settled at
  45–72). Cost ~101 min per N=20 run because the bond sits at the cap
  throughout. Peak bond dimension is **not** a function of q — sample 6 has the
  largest q and sits at 85 while mid-q samples 1 and 5 cap out — so it tracks
  the *direction* of L'' in the commutant, and which samples are usable cannot
  be predicted.

χ sweep, sample 8, N=20: 3.004e-4 (χ=32, capped), 3.294e-4 (χ=64, capped),
3.698e-4 (χ=128, **bond 64, cutoff-limited**). Only the last is a real point;
truncation suppresses R by 19% at χ=32, i.e. in the decay-fabricating direction.

### Trap 4: don't fit R(N) when you can read R(i,r)

⚠️ Both items below stand as stated, but note the larger lesson they hide:
reading the profile instead of fitting R(N) was the right *move* and still gave
the wrong *answer*, because both were measured on an unrelaxed state. A better
diagnostic applied to a state that has not converged is still wrong. Preferring
the profile does not substitute for asking whether the run finished.

Two things went wrong before the profiles were looked at, both worth avoiding:

- **The boundary upturn inverts a ξ fit.** R rises over the last ~2 sites of an
  open chain (sample 8, N=20: 2.45e-4, 2.60e-4, 5.27e-4 at sep 12,13,14).
  Fitting the "outer half" of the profile — the natural choice — lands squarely
  on it and returns ξ = ∞ or nonsense (910, 13862). Drop the last two sites.
- **AICc silently refuses a 3-parameter fit on 4 points.** The plateau model
  R_∞ + A e^{−N/2ξ} has k=3, and the correction term needs n > k+1, so at four
  surviving sizes it returns ∞ and the plateau loses *by construction*, not on
  evidence. `drift_annihilation_scaling.py` reports "power_law 7 / exponential
  3" and that verdict is worthless for the nine samples with four points.
  The profile needs no model selection at all — prefer it.

### ⚠️ Trap 5: two initial states agreeing is not evidence of relaxation

**The single most expensive mistake in this project.** The zero/neel test is
used everywhere here as *the* two-sided convergence bracket, on the argument
that one start approaches from above and the other from below, so agreement
pins the answer between them. That argument has a hole, and this model fell
straight through it.

The two starts differ only in their *local* configuration. The fast rates (~1
here, 4–16 in the first model) equilibrate that within a few time units. After
that both states sit on the same slow manifold — the one direction whose decay
rate is the small pair-creation rate q — and they crawl along it **together**,
in the same direction, at the same speed. They agree with each other at every
moment while both are far from the fixed point.

Measured: at N=20 the two starts agree to **0.13%** (sample 8) and ≤0.68%
(worst in the ensemble), and both are **19× below** the true value.

    what "agreement" requires   what it actually got
    ------------------------------------------------------------------------
    approach from opposite      both descend onto one slow manifold, then
    sides of the answer         co-drift along it in the same direction

So the test is only a bracket once *both* trajectories have flattened. Read
`stage_correlators` / the trajectory shape first; if both are still moving
monotonically, their agreement means they are correlated, not converged. What
would have caught it: any absolute reference (the infinite-system value, an
exact sector solve at a size that does relax, or 12/q as a required schedule
length computed *before* the run rather than a spread checked after it).

Corollary for the first model's section above: "steady-state uniqueness holds
where the runs are converged, 0.00% at N=4 and 8, 0.21% at N=12" is evidence of
uniqueness *given* convergence, and evidence of nothing at all without it.

### What is open now

**The p-sweep is dead as posed, and so is the question it was going to answer.**
It was the highest-value follow-up only while ξ was believed finite at p = 0.8
and infinite at p = 1/2, so that some p_c had to separate them. There is no such
boundary to find: the order is present at p = 0.8, and the bias destroys
nothing. Do not run it to locate p_c. A p-sweep is still mildly interesting for
the *amplitude* — whether R = 4q/γ_A picks up a bias-dependent prefactor in the
thermodynamic limit as F does at finite N (the F table above: 2.0 → 2.34 →
1.75 across p) — but that is a quantitative curiosity, not an open phase
question.

⚠️ Do not compare a p-sweep at fixed ε against the p = 0.8 grid without
renormalizing: the *relative* perturbation strength depends on the baseline
rates, which is what inflated the N=4 sample scatter from 0.78% to 4.3% here.

**The genuinely open item: the finite grid is now known-wrong and unfixed.**
Every N ≥ 8 number in `renyi2_drift_annihilation.pkl` measures a transient. It
should be re-run at a 12/q-sized schedule (~50–120× the current 55.5 units) and
the finite values should then rise toward 4q/γ_A. That is expensive at N=20,
where truncation binds independently (χ > 128 for 9 of 10 samples), so the
tractable version is N ≤ 12 with the long schedule — enough to confirm the
finite numbers move in the predicted direction, without fighting truncation at
the same time. Until that is done, **quote the infinite-system grid, not the
finite one, for this model.**

**Then check the first model's finite grid for the same disease.** Its rates are
16× larger, so 55.5 units buys ~16× more relaxation and its N ≤ 16 plateau at
R = q/8 is probably genuine — its N=4-exact anchor and its two-sided agreement
would both be consistent either way, which is precisely the trap. The cheap
discriminator: it already reports the *right* answer (R/q = 0.1256 vs the
analytic 1/8), which an unrelaxed grid would not. Worth one long-schedule run at
N=12 to confirm rather than assume.

## The infinite system (iTEBD) — FIRST model

⚠️ Everything in this section is the **first** model (`baseline_jump_operators`,
L = XX(1−ZZ)). The second model's infinite-system run lives in its own section
above ("The infinite system settles it: R = 4q/γ_A"). Keeping this straight is
not pedantry: reading this section's flat R against the second model's finite-N
decay, as if the two were the same Lindbladian, is what delayed the retraction
above by two days.

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
      renyi2_drift_annihilation.py  **the second model** — driven classical
                                 circuit, 10 samples x 5 sizes x both starts at
                                 chi=128, plus L''=0 controls, a chi sweep and
                                 an exact (+,+)-sector reference at N=4
      drift_annihilation_scaling.py  R(N) decay-shape fits for the above. Read
                                 its AICc caveat (Trap 4) before quoting it;
                                 the profile is the better diagnostic
      imps_drift_eps_grids.py    **the second model in the thermodynamic
                                 limit** — p=0.8, ε ∈ {0.2,0.15,0.1,0.05},
                                 10 samples from |0…0⟩, plus |neel⟩ cross-checks
                                 at ε=0.2, L''=0 controls and a 3×-length
                                 control. Schedules are sized per sample as
                                 12/q, not uniformly per ε. This is the run
                                 that retracted "no SWSSB in the second model"
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
`imps_positivity_hermiticity.png`. Second model, infinite:
`imps_drift_flatness.png` (R(r)/R(1) per sample, one panel per ε — the flat
lines are the retraction), `imps_drift_R_vs_epsilon.png` (R vs q, and R/q vs ε),
`imps_drift_finite_vs_infinite.png` (the finite grid's decay against the
thermodynamic limit, same sample, same ε).

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

    uv run pytest                                     # 133 tests, ~5 s
    uv run python experiments/renyi2_swssb_chi128.py  # ~2.7 h per initial state
    uv run python experiments/imps_drift_eps_grids.py  # ~3.9 h, 46 jobs

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
