"""Infinite TEBD driver for the manifestly-positive LPDO ansatz.

Mirrors itebd.py one-for-one -- same second-order Trotter split over the
2-bond unit cell, same stage_callback contract, same BLAS thread pinning --
but evolves rho = X X^dagger via Kraus operators (ilpdo.apply_bond_gate_cp)
instead of applying a superoperator to a vectorized rho. See ilpdo.py for why:
the vectorized path drifts off the physical manifold and lands on a stationary
but non-positive operator, which inflated R by a factor of ~2 relative to the
ED-validated finite chain.

Kept as a separate module rather than folded into itebd.py so the vectorized
path stays untouched and the two can be run against each other -- the
factor-of-2 comparison is the acceptance test for this whole ansatz.

Trace normalization comes free here, which is worth noting because the
vectorized ansatz cannot have it. Canonicalizing X drives its transfer
operator's leading eigenvalue to 1, and that eigenvalue *is* the per-unit-cell
value of <X|X> = Tr[X X^dagger] = Tr[rho]. So periodic canonicalization
trace-normalizes rho automatically; there is no separate normalize() step and
no competition between unit trace and unit 2-norm (for an infinite chain the
vectorized code has to choose the 2-norm, which is how Tr[rho] ended up
negative there).
"""

import numpy as np

from . import blas, ilpdo


def unit_cell_kraus(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt: float,
    d_site: int = 2,
) -> list[np.ndarray]:
    """Kraus operators for the one gate applied to BOTH bonds of the unit cell.

    Thin wrapper around ilpdo.kraus_operators. As in itebd.unit_cell_gate, a
    translation-invariant infinite chain needs exactly one bond gate (uniform
    0.5/0.5 single-site weighting, no boundary special-casing), unlike
    tebd.build_bond_gates' N-1 distinct gates.

    Input: model terms; dt; d_site.
    Output: list of (d_site**2, d_site**2) Kraus operators.
    """
    return ilpdo.kraus_operators(H2_terms, H1_terms, L2_terms, L1_terms, dt, d_site)


def evolve_infinite_lpdo(
    state: ilpdo.iLPDO,
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt: float,
    n_steps: int,
    chi_max: int | None = None,
    kappa_max: int | None = None,
    cutoff: float | None = None,
    d_site: int = 2,
    canonicalize_every: int = 10,
    blas_threads: int | None = 1,
) -> dict:
    """Run n_steps of second-order-Trotter imaginary-time iTEBD on the LPDO, in place.

    Each step is exp(dt*L) ~= U_A(dt/2) U_B(dt) U_A(dt/2), the same split
    itebd.evolve_infinite uses. Every canonicalize_every steps the state is
    regauged with ilpdo.canonicalize (which reuses imps.canonicalize on the
    fused view of X) -- this both fixes the gauge and, as noted in the module
    docstring, trace-normalizes rho.

    Between canonicalizations a cheap Frobenius rescale of X runs every step,
    purely to stop floats over/underflowing across many non-unitary steps. It
    is plumbing, not physics: rho = X X^dagger is only defined up to this
    scale, and canonicalize sets the physical normalization.

    Input:
        state: iLPDO to evolve in place.
        H2_terms, H1_terms, L2_terms, L1_terms: model terms.
        dt, n_steps: step size and count.
        chi_max, cutoff: bond truncation.
        kappa_max: cap on each site's Kraus dimension. With Kraus rank 8 the
            leg grows x8 per gate, so leaving this None will blow up within a
            few steps -- set it.
        d_site: physical dimension of one site.
        canonicalize_every: regauge (and trace-normalize) this often.
        blas_threads: BLAS/LAPACK cap for the loop, see blas.limit_threads.
    Output: dict of history lists: 'discarded_weight' and
        'kraus_discarded_weight' (max over the step's three gate applications),
        'trace_per_cell' recorded at each canonicalization, and
        'eigenvalue_drift' (|leading transfer eigenvalue - 1| before that
        call's rescale), the analogue of itebd.evolve_infinite's diagnostic.
    """
    kraus_half = unit_cell_kraus(H2_terms, H1_terms, L2_terms, L1_terms, dt / 2, d_site)
    kraus_full = unit_cell_kraus(H2_terms, H1_terms, L2_terms, L1_terms, dt, d_site)

    history: dict = {
        "discarded_weight": [], "kraus_discarded_weight": [],
        "trace_per_cell": [], "eigenvalue_drift": [],
    }
    with blas.limit_threads(blas_threads):
        for step in range(n_steps):
            d1 = ilpdo.apply_bond_gate_cp(state, "A", kraus_half, chi_max, kappa_max, cutoff)
            d2 = ilpdo.apply_bond_gate_cp(state, "B", kraus_full, chi_max, kappa_max, cutoff)
            d3 = ilpdo.apply_bond_gate_cp(state, "A", kraus_half, chi_max, kappa_max, cutoff)
            history["discarded_weight"].append(
                max(d["discarded_weight"] for d in (d1, d2, d3))
            )
            history["kraus_discarded_weight"].append(
                max(d["kraus_discarded_weight"] for d in (d1, d2, d3))
            )

            for key in ("A", "B"):
                nrm = np.linalg.norm(state.X[key])
                if nrm > 0 and np.isfinite(nrm):
                    state.X[key] = state.X[key] / nrm

            if (step + 1) % canonicalize_every == 0:
                diag = ilpdo.canonicalize(state, chi_max, cutoff)
                history["eigenvalue_drift"].append(abs(diag["eigenvalue_left_env"] - 1.0))
                history["trace_per_cell"].append(state.trace_per_cell())

    return history


def find_steady_state_infinite_lpdo(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt_schedule: list[float],
    steps_per_dt: int,
    chi_max: int | None = None,
    kappa_max: int | None = None,
    cutoff: float | None = None,
    d_site: int = 2,
    canonicalize_every: int = 10,
    initial_state: ilpdo.iLPDO | None = None,
    blas_threads: int | None = 1,
    stage_callback=None,
) -> tuple[ilpdo.iLPDO, dict]:
    """Find the infinite-chain steady state on the LPDO ansatz via annealed iTEBD.

    Mirrors itebd.find_steady_state_infinite: one evolve_infinite_lpdo call per
    dt in the schedule, reusing the evolving state, with the same
    stage_callback(stage, dt, state) contract (state is canonicalized at that
    point provided steps_per_dt is a multiple of canonicalize_every).

    Note the callback receives an iLPDO, not an iMPS -- call
    ilpdo.to_vectorized_imps() plus imps.canonicalize() to measure it with
    iobservables.

    Input: as evolve_infinite_lpdo, plus dt_schedule/steps_per_dt and
        initial_state (defaults to ilpdo.iLPDO.maximally_mixed(d_site)).
    Output: (state, history) with concatenated per-step diagnostics plus
        history['dt'], the dt active at each recorded step.
    """
    state = (initial_state.copy() if initial_state is not None
             else ilpdo.iLPDO.maximally_mixed(d_site))

    history: dict = {
        "discarded_weight": [], "kraus_discarded_weight": [],
        "trace_per_cell": [], "eigenvalue_drift": [], "dt": [],
    }
    with blas.limit_threads(blas_threads):
        for stage, dt in enumerate(dt_schedule):
            stage_history = evolve_infinite_lpdo(
                state, H2_terms, H1_terms, L2_terms, L1_terms, dt, steps_per_dt,
                chi_max, kappa_max, cutoff, d_site, canonicalize_every, blas_threads,
            )
            for key in ("discarded_weight", "kraus_discarded_weight",
                        "trace_per_cell", "eigenvalue_drift"):
                history[key].extend(stage_history[key])
            history["dt"].extend([dt] * steps_per_dt)
            if stage_callback is not None:
                stage_callback(stage, dt, state)

        ilpdo.canonicalize(state, chi_max, cutoff)
    return state, history
