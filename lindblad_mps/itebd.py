"""Infinite-chain TEBD driver: 2nd-order Trotter over a 2-site unit cell.

Mirrors tebd.py's driver-loop role one-for-one (see tebd.py's module
docstring for what "imaginary-time TEBD" means here). No code is shared with
the finite driver: its odd/even-bond bookkeeping across N-1 bonds has no
analogue when there are always exactly 2 bonds, so cost per step is O(1)
SVDs, independent of any "system size".
"""

import numpy as np

from . import blas
from . import imps
from . import vectorize


def unit_cell_gate(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt: float,
    d_site: int = 2,
) -> np.ndarray:
    """Build the single gate exp(dt * L_bond) applied to BOTH bonds of the unit cell.

    weight_left = weight_right = 0.5 uniformly and no boundary special-casing
    (there is no boundary): unlike tebd.build_bond_gates, which builds N-1
    distinct gates for a finite chain, a translation-invariant infinite chain
    needs exactly one. Thin wrapper around vectorize.bond_gate.

    Input: model terms, see tebd.build_bond_gates(); dt: (imaginary) time step.
    Output: (d_site**4, d_site**4) ndarray, the vectorized bond propagator.
    """
    return vectorize.bond_gate(
        H2_terms, L2_terms, dt,
        H1_left_terms=H1_terms, L1_left_terms=L1_terms,
        H1_right_terms=H1_terms, L1_right_terms=L1_terms,
        weight_left=0.5, weight_right=0.5, d_site=d_site,
    )


def evolve_infinite(
    state: imps.iMPS,
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt: float,
    n_steps: int,
    chi_max: int | None = None,
    cutoff: float | None = None,
    d_site: int = 2,
    canonicalize_every: int = 10,
    blas_threads: int | None = 1,
) -> dict:
    """Run n_steps of second-order-Trotter imaginary-time iTEBD in place.

    Each step is exp(dt*L) ~= U_A(dt/2) U_B(dt) U_A(dt/2), the exact analogue
    of tebd.evolve's half-odd/full-even/half-odd split with the unit cell's 2
    bonds standing in for the finite chain's even/odd bond sets. Each
    apply_bond_gate call already renormalizes its own bond (see
    imps.apply_bond_gate); every canonicalize_every steps, state.canonicalize
    is called, which both regauges the state to true Vidal canonical form
    (see imps.py) and corrects Gamma's overall-magnitude drift (its
    eta**0.25 rescale). Between canonicalize() calls, a cheap defensive
    Frobenius-norm rescale of Gamma is applied every step purely to keep
    floats from over/underflowing -- not a physical correction (see
    imps.py's module docstring).

    Input:
        state: iMPS to evolve in place.
        H2_terms, H1_terms, L2_terms, L1_terms: model terms, see
            unit_cell_gate().
        dt: (imaginary) time step. n_steps: number of Trotter steps.
        chi_max, cutoff: bond-dimension truncation, applied at every gate and
            every canonicalize() call.
        d_site: physical dimension of one site.
        canonicalize_every: call state.canonicalize() every this many steps.
        blas_threads: BLAS/LAPACK thread cap for the whole loop (see
            blas.limit_threads -- matters even more here than in the finite
            case, since ARPACK's eigs issues many small BLAS calls per
            canonicalize()). None leaves threading untouched.
    Output: dict with diagnostic history lists:
        'discarded_weight': max per-step discarded SVD weight (max over the
            3 gate applications that make up one Trotter step),
        'eigenvalue_drift': |eigenvalue_A - 1| at each canonicalize() call,
            PRE-rescale -- recorded only at canonicalize_every boundaries
            (a per-step version would cost as much as canonicalize() itself
            every step); the infinite analogue of tebd.evolve's per-step
            'overlap' history, at coarser granularity for cost reasons.
    """
    gate_half = unit_cell_gate(H2_terms, H1_terms, L2_terms, L1_terms, dt / 2, d_site)
    gate_full = unit_cell_gate(H2_terms, H1_terms, L2_terms, L1_terms, dt, d_site)

    history: dict = {"discarded_weight": [], "eigenvalue_drift": []}
    with blas.limit_threads(blas_threads):
        for step in range(n_steps):
            d1 = imps.apply_bond_gate(state, "A", gate_half, chi_max, cutoff)
            d2 = imps.apply_bond_gate(state, "B", gate_full, chi_max, cutoff)
            d3 = imps.apply_bond_gate(state, "A", gate_half, chi_max, cutoff)
            history["discarded_weight"].append(max(d1, d2, d3))

            for name in ("A", "B"):
                nrm = np.linalg.norm(state.Gamma[name])
                if nrm > 0 and np.isfinite(nrm):
                    state.Gamma[name] = state.Gamma[name] / nrm

            if (step + 1) % canonicalize_every == 0:
                diag = state.canonicalize(chi_max, cutoff)
                history["eigenvalue_drift"].append(abs(diag["eigenvalue_A"] - 1.0))

    return history


def find_steady_state_infinite(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt_schedule: list[float],
    steps_per_dt: int,
    chi_max: int | None = None,
    cutoff: float | None = None,
    d_site: int = 2,
    canonicalize_every: int = 10,
    initial_state: imps.iMPS | None = None,
    blas_threads: int | None = 1,
    stage_callback=None,
) -> tuple[imps.iMPS, dict]:
    """Find the infinite-chain Lindbladian steady state via annealed iTEBD.

    Mirrors tebd.find_steady_state exactly (no N argument -- the chain is
    translation-invariant by construction): runs evolve_infinite() once per
    entry in dt_schedule, reusing the evolving state between stages.

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: model terms.
        dt_schedule, steps_per_dt: as tebd.find_steady_state.
        chi_max, cutoff: bond-dimension truncation.
        d_site: physical dimension of one site.
        canonicalize_every: passed to evolve_infinite() at every stage.
        initial_state: starting iMPS; defaults to imps.iMPS.maximally_mixed(d_site).
        blas_threads: see evolve_infinite().
        stage_callback: optional callable(stage_index, dt, state), invoked
            after each dt stage -- the state is canonicalized at that point
            provided steps_per_dt is a multiple of canonicalize_every, exactly
            as in tebd.find_steady_state.
    Output: (state, history) -- the final iMPS (canonicalized) and a dict of
        concatenated per-step diagnostics (see evolve_infinite()), plus
        history['dt'], the dt active at each recorded step.
    """
    state = initial_state.copy() if initial_state is not None else imps.iMPS.maximally_mixed(d_site)

    history: dict = {"discarded_weight": [], "eigenvalue_drift": [], "dt": []}
    with blas.limit_threads(blas_threads):
        for stage, dt in enumerate(dt_schedule):
            stage_history = evolve_infinite(
                state, H2_terms, H1_terms, L2_terms, L1_terms, dt, steps_per_dt,
                chi_max, cutoff, d_site, canonicalize_every, blas_threads,
            )
            history["discarded_weight"].extend(stage_history["discarded_weight"])
            history["eigenvalue_drift"].extend(stage_history["eigenvalue_drift"])
            history["dt"].extend([dt] * steps_per_dt)
            if stage_callback is not None:
                stage_callback(stage, dt, state)

        state.canonicalize(chi_max, cutoff)
    return state, history
