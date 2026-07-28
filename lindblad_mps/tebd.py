"""Finite-chain TEBD driver for finding the Lindbladian steady state.

The steady state is the fixed point of imaginary-time evolution under the
vectorized Liouvillian, so this is structurally imaginary-time TEBD: the
"time step" dt has no physical meaning beyond a convergence knob, and the
state must be renormalized (in 2-norm, see mps.MPS.normalize) after every
step since the gates exp(dt * L_bond) are not unitary.

The model (nearest-neighbour Hamiltonian/jump terms, plus single-site terms)
is specified exactly as in vectorize.py / exact.py: lists of
(operator, coefficient) pairs, applied uniformly to every bond/site
(translation-invariant chain).
"""

import numpy as np

from . import blas
from . import mps as mps_module
from . import vectorize


def bond_weights(bond_index: int, n_bonds: int) -> tuple[float, float]:
    """Fraction of a single-site term's coefficient assigned to one bond.

    Interior sites split their single-site term 0.5/0.5 across their two
    adjacent bonds; chain-end sites get full weight 1.0 on their only bond
    (see vectorize.build_bond_generator).

    Input: bond_index, index of the bond (0 to n_bonds-1); n_bonds, total
        number of bonds (N-1 for an N-site chain).
    Output: (weight_left, weight_right) tuple of floats.
    """
    weight_left = 1.0 if bond_index == 0 else 0.5
    weight_right = 1.0 if bond_index == n_bonds - 1 else 0.5
    return weight_left, weight_right


def build_bond_gates(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    dt: float,
    d_site: int = 2,
) -> list[np.ndarray]:
    """Build the N-1 vectorized bond gates exp(dt * L_bond) for a chain.

    Each bond's single-site contributions are weighted via bond_weights() so
    that summing all bonds gives every site its single-site term exactly
    once (interior sites split across two bonds, boundary sites get full
    weight on their one bond).

    Input:
        H2_terms, L2_terms: two-site (Hamiltonian, jump) terms, applied
            uniformly to every bond.
        H1_terms, L1_terms: single-site (Hamiltonian, jump) terms, applied
            uniformly to every site.
        N: number of sites.
        dt: (imaginary) time step for this gate set.
        d_site: physical dimension of one site.
    Output: list of N-1 (d_site^4, d_site^4) ndarrays, one gate per bond,
        in local-vec order (ready for mps.MPS.apply_two_site_gate).
    """
    n_bonds = N - 1
    gates = []
    for b in range(n_bonds):
        wl, wr = bond_weights(b, n_bonds)
        gates.append(
            vectorize.bond_gate(
                H2_terms,
                L2_terms,
                dt,
                H1_left_terms=H1_terms,
                L1_left_terms=L1_terms,
                H1_right_terms=H1_terms,
                L1_right_terms=L1_terms,
                weight_left=wl,
                weight_right=wr,
                d_site=d_site,
            )
        )
    return gates


def apply_gate_layer(
    state: mps_module.MPS,
    gates: list[np.ndarray],
    bonds: list[int],
    chi_max: int | None,
    cutoff: float | None,
) -> list[float]:
    """Apply a list of gates to a list of non-overlapping bonds, in place.

    Bonds in one layer (all-even or all-odd bond indices) never share a
    site, so applying them one at a time in any order is equivalent to
    applying them "simultaneously".

    Input:
        state: MPS to update in place.
        gates: list of (phys_dim^2, phys_dim^2) gates, one per bond in `bonds`.
        bonds: list of bond indices (left-site index of each bond).
        chi_max, cutoff: truncation passed to MPS.apply_two_site_gate.
    Output: list of per-bond discarded-weight fractions (see
        MPS.apply_two_site_gate), same length as `bonds`.
    """
    discarded = []
    for bond, gate in zip(bonds, gates):
        discarded.append(state.apply_two_site_gate(bond, gate, chi_max, cutoff))
    return discarded


def evolve(
    state: mps_module.MPS,
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt: float,
    n_steps: int,
    chi_max: int | None = None,
    cutoff: float | None = None,
    d_site: int = 2,
    recanonicalize_every: int = 10,
    blas_threads: int | None = 1,
) -> dict:
    """Run n_steps of second-order-Trotter imaginary-time TEBD in place.

    Each step is the symmetric Trotter decomposition
    exp(dt*L) ~= U_odd(dt/2) U_even(dt) U_odd(dt/2), where "even"/"odd"
    refers to the parity of the bond index (bonds (0,1),(2,3),... vs
    (1,2),(3,4),...). The state is renormalized to unit 2-norm after every
    step (gates are non-unitary), and periodically re-canonicalized (see
    mps.MPS.canonicalize) to correct the truncation drift that non-unitary
    gates introduce.

    Input:
        state: MPS to evolve in place (e.g. from MPS.maximally_mixed()).
        H2_terms, H1_terms, L2_terms, L1_terms: model terms, see
            build_bond_gates().
        dt: (imaginary) time step.
        n_steps: number of Trotter steps to take.
        chi_max, cutoff: bond-dimension truncation applied at every gate and
            every re-canonicalization.
        d_site: physical dimension of one site.
        recanonicalize_every: call state.canonicalize() every this many steps.
        blas_threads: BLAS/LAPACK thread cap held for the whole loop (see
            blas.limit_threads -- the default of 1 is ~4x faster end to end
            at chi=32, because threaded LAPACK loses badly on the small
            (chi*d, chi*d) SVDs). None leaves threading untouched.
    Output: dict with diagnostic history lists:
        'norm': pre-renormalization norm at each step (drifts away from 1
            as a measure of how fast the state is still evolving),
        'overlap': |<rho(t)|rho(t+dt))>| at each step (-> 1 at convergence,
            since both states are unit-normalized before comparing),
        'discarded_weight': largest per-bond discarded SVD weight at each step.
    """
    N = state.N
    n_bonds = N - 1
    even_bonds = list(range(0, n_bonds, 2))
    odd_bonds = list(range(1, n_bonds, 2))

    gates_half = build_bond_gates(H2_terms, H1_terms, L2_terms, L1_terms, N, dt / 2, d_site)
    gates_full = build_bond_gates(H2_terms, H1_terms, L2_terms, L1_terms, N, dt, d_site)
    gates_half_even = [gates_half[b] for b in even_bonds]
    gates_half_odd = [gates_half[b] for b in odd_bonds]
    gates_full_even = [gates_full[b] for b in even_bonds]

    history = {"norm": [], "overlap": [], "discarded_weight": []}
    state.normalize()

    # Entered once per run, not per gate: the thread-limit switch itself costs
    # ~1 ms, far more than a single bond SVD.
    with blas.limit_threads(blas_threads):
        for step in range(n_steps):
            prev = state.copy()

            d1 = apply_gate_layer(state, gates_half_odd, odd_bonds, chi_max, cutoff)
            d2 = apply_gate_layer(state, gates_full_even, even_bonds, chi_max, cutoff)
            d3 = apply_gate_layer(state, gates_half_odd, odd_bonds, chi_max, cutoff)

            norm = state.normalize()
            if (step + 1) % recanonicalize_every == 0:
                state.canonicalize(chi_max, cutoff)
                state.normalize()

            overlap = abs(state.overlap(prev))
            history["norm"].append(norm)
            history["overlap"].append(overlap)
            history["discarded_weight"].append(max(d1 + d2 + d3, default=0.0))

    return history


def find_steady_state(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    dt_schedule: list[float],
    steps_per_dt: int,
    chi_max: int | None = None,
    cutoff: float | None = None,
    d_site: int = 2,
    recanonicalize_every: int = 10,
    initial_state: mps_module.MPS | None = None,
    blas_threads: int | None = 1,
    stage_callback=None,
) -> tuple[mps_module.MPS, dict]:
    """Find the Lindbladian steady state of a finite chain via annealed TEBD.

    Runs evolve() once per entry in dt_schedule (largest dt first is the
    usual choice, to reach the basin of attraction quickly, then smaller dt
    to reduce the O(dt^2) Trotter error of the converged state), reusing the
    evolving state between stages.

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: model terms, see
            build_bond_gates().
        N: number of sites.
        dt_schedule: list of (imaginary) time steps, one TEBD stage per entry.
        steps_per_dt: number of Trotter steps to run at each dt.
        chi_max, cutoff: bond-dimension truncation.
        d_site: physical dimension of one site.
        recanonicalize_every: passed to evolve() at every stage.
        initial_state: starting MPS; defaults to MPS.maximally_mixed(N, d_site).
        blas_threads: BLAS/LAPACK thread cap for the whole run, see evolve().
        stage_callback: optional callable(stage_index, dt, state) invoked after
            each dt stage, for measuring an observable's drift across stages.
            An observable still moving between the last two stages means the
            run has not converged in time -- which the per-step 'overlap'
            history cannot detect, since that shrinks with dt no matter how
            far the state still is from the fixed point. The state passed in
            is live, not a copy: read it, do not mutate it. Provided that
            steps_per_dt is a multiple of recanonicalize_every, it is already
            canonicalized and unit-normalized at this point.
    Output:
        (state, history): the final MPS (canonicalized and unit-normalized)
        and a dict of concatenated per-step diagnostic lists (see evolve()),
        plus history['dt'], the dt active at each recorded step.
    """
    state = initial_state.copy() if initial_state is not None else mps_module.MPS.maximally_mixed(
        N, d_site
    )

    history = {"norm": [], "overlap": [], "discarded_weight": [], "dt": []}
    with blas.limit_threads(blas_threads):
        for stage, dt in enumerate(dt_schedule):
            stage_history = evolve(
                state,
                H2_terms,
                H1_terms,
                L2_terms,
                L1_terms,
                dt,
                steps_per_dt,
                chi_max,
                cutoff,
                d_site,
                recanonicalize_every,
                blas_threads,
            )
            for key in ("norm", "overlap", "discarded_weight"):
                history[key].extend(stage_history[key])
            history["dt"].extend([dt] * steps_per_dt)
            if stage_callback is not None:
                stage_callback(stage, dt, state)

        state.canonicalize(chi_max, cutoff)
        state.normalize()
    return state, history
