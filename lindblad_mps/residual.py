"""Steady-state residual ||L|rho>|| / |||rho>|| for an MPS, via a Liouvillian MPO.

Why this exists
---------------
TEBD reports two convergence numbers and neither is a convergence test. The
per-step overlap 1 - |<rho(t)|rho(t+dt)>| shrinks with dt whether or not the
state is near the fixed point, so a slowly-drifting state passes it trivially.
The correlator's drift between dt stages is an improvement but is still a
*relative* statement about one observable, and it oscillates as dt anneals.

The residual is the honest test, because it is a statement about the state
rather than about the trajectory that produced it:

    r = ||L |rho>|| / || |rho> ||

is exactly zero at any steady state and nonzero otherwise, with no reference
to dt, to the initial state, or to how long the run took. It is what
renyi2_swssb.validate_against_exact already computes densely at N=4; this
module computes the same quantity for an MPS at any N.

Reading it
----------
r has units of rate: since d rho/dt = L rho, it is the fractional rate of
change of the state per unit time. If the residual is dominated by the slowest
mode, the remaining relative error in a converged observable is roughly r *
tau, with tau the relaxation time -- so r must be compared against the
Liouvillian gap, not against zero. `residual_per_bond` divides by N-1, since L
is a sum of N-1 bond terms and an evenly-spread error is extensive.

It does not go to zero, and what it stops at is informative. TEBD converges to
the fixed point of the *Trotterized* propagator, which differs from the true
steady state at second order in dt, so a fully relaxed run sits on a floor
r = C dt^2. Measured on this model at N=4 (test_residual.py pins it):

    dt      0.1        0.05       0.02       0.01       0.005
    r       1.954e-04  5.004e-05  8.062e-06  2.018e-06  5.045e-07
    r/dt^2  1.954e-02  2.002e-02  2.015e-02  2.018e-02  2.018e-02

which is what makes the diagnostic a convergence *test* rather than just a
number: a run at its Trotter floor has relaxed and is limited only by dt,
while a run orders of magnitude above the floor has not, no matter how still
its observables look. Calibrate the floor by re-running one converged case at
half the final dt -- the floor drops 4x, a genuinely unrelaxed residual barely
moves.

The other floor is arithmetic: r is extracted as sqrt of <rho|L^dagger L|rho>,
a near-total cancellation between terms of order ||L||^2, so r cannot be
resolved below roughly ||L|| * sqrt(machine epsilon) ~ 1e-7 here. Far below
any Trotter floor of interest, but do not read six digits off a residual of
1e-7.

Cost
----
The contraction is a four-layer transfer sweep (bra MPS, bra MPO, ket MPO, ket
MPS) with an environment of shape (chi, D, D, chi), costing O(chi^3 D^2 d +
chi^2 D^3 d^2) per site. It never forms L|rho>, which would have bond
dimension D*chi. For the SWSSB model D = 18 and the sweep takes seconds at
chi = 48, minutes at chi = 128 -- negligible against runs of hours, and it is
measured once at the end of a run rather than per step.

The bond generator's operator-Schmidt spectrum falls off a cliff (29, 17, 4.0,
4.0, 4.0, 1.1, 0.95, then 0.037 and below), so `mpo_cutoff=1e-3` drops D from
18 to 9 and the D^3 term by 8x, at the price of putting a floor of the same
relative size under the measured residual. Left off by default: the point of
the diagnostic is to be trusted near zero.
"""

import numpy as np

from . import mps as mps_module
from . import tebd, vectorize


def bond_generators(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    d_site: int = 2,
) -> list[np.ndarray]:
    """Build the N-1 vectorized bond generators whose sum is the chain Liouvillian.

    The exact analogue of tebd.build_bond_gates, without the exponentiation:
    same terms, same single-site weighting via tebd.bond_weights (interior
    sites split 0.5/0.5 across their two bonds, chain ends take full weight on
    their only bond), so that summing the bonds gives every site its
    single-site term exactly once.

    Input:
        H2_terms, L2_terms: two-site (Hamiltonian, jump) terms, applied to
            every bond.
        H1_terms, L1_terms: single-site terms, applied to every site.
        N: number of sites.
        d_site: physical dimension of one site.
    Output: list of N-1 (d_site^4, d_site^4) ndarrays in local-vec order.
    """
    n_bonds = N - 1
    generators = []
    for b in range(n_bonds):
        wl, wr = tebd.bond_weights(b, n_bonds)
        generators.append(
            vectorize.build_bond_generator(
                H2_terms,
                L2_terms,
                H1_left_terms=H1_terms,
                L1_left_terms=L1_terms,
                H1_right_terms=H1_terms,
                L1_right_terms=L1_terms,
                weight_left=wl,
                weight_right=wr,
                d_site=d_site,
            )
        )
    return generators


def split_bond_generator(
    generator: np.ndarray, phys_dim: int, cutoff: float = 0.0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Operator-Schmidt decompose a bond generator into single-site factors.

    Writes G[(i1 i2), (j1 j2)] = sum_k A_k[i1, j1] B_k[i2, j2] by regrouping
    the indices so that each site's (out, in) pair sits together, then taking
    an SVD and splitting sqrt(S) into both factors.

    Input:
        generator: (phys_dim^2, phys_dim^2) two-site operator in local-vec order.
        phys_dim: dimension of one local-vec site (local_dim^2, i.e. 4 for spins).
        cutoff: drop Schmidt values below cutoff * (largest); 0 keeps all.
    Output: (A_list, B_list), each a list of r (phys_dim, phys_dim) operators
        for the left and right site of the bond.
    """
    d = phys_dim
    regrouped = (
        generator.reshape(d, d, d, d)  # (i1, i2, j1, j2)
        .transpose(0, 2, 1, 3)         # (i1, j1, i2, j2)
        .reshape(d * d, d * d)
    )
    U, S, Vh = np.linalg.svd(regrouped)
    keep = len(S)
    if cutoff > 0 and S[0] > 0:
        keep = max(1, int(np.sum(S >= cutoff * S[0])))
    root = np.sqrt(S[:keep])
    A = [(U[:, k] * root[k]).reshape(d, d) for k in range(keep)]
    B = [(root[k] * Vh[k, :]).reshape(d, d) for k in range(keep)]
    return A, B


def liouvillian_mpo(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    d_site: int = 2,
    mpo_cutoff: float = 0.0,
) -> list[np.ndarray]:
    """Build an MPO for the full chain Liouvillian L = sum_b G_b.

    Standard three-channel finite-state machine over the bond index: channel 0
    has started no term yet, channels 1..r carry "the left factor A_k was
    applied on the previous site, the matching B_k is due now", and the last
    channel has completed its term and passes identity onwards. A sum of
    nearest-neighbour terms of operator-Schmidt rank r therefore needs bond
    dimension exactly r + 2 -- squaring it for L^dagger L would give (r+2)^2,
    which is finite and small, not the non-local object it is sometimes taken
    for. Boundary tensors are the corresponding row/column slices.

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: model terms, see
            bond_generators().
        N: number of sites (>= 2).
        d_site: physical dimension of one site.
        mpo_cutoff: relative Schmidt cutoff per bond (see
            split_bond_generator); 0 is exact.
    Output: list of N tensors of shape (D_left, phys_dim, phys_dim, D_right),
        indexed [bond_in, physical_out, physical_in, bond_out], with
        D_left = 1 on the first site and D_right = 1 on the last.
    """
    if N < 2:
        raise ValueError(f"need at least 2 sites for a bond Liouvillian, got N={N}")

    d = d_site * d_site
    generators = bond_generators(H2_terms, H1_terms, L2_terms, L1_terms, N, d_site)
    splits = [split_bond_generator(G, d, mpo_cutoff) for G in generators]
    r = max(len(A) for A, _ in splits)
    D = r + 2
    identity = np.eye(d, dtype=complex)

    tensors = []
    for n in range(N):
        W = np.zeros((D, d, d, D), dtype=complex)
        W[0, :, :, 0] = identity          # no term started yet
        W[D - 1, :, :, D - 1] = identity  # term already completed
        if n < N - 1:  # site n is the left site of bond n: a term may start here
            for k, A in enumerate(splits[n][0]):
                W[0, :, :, 1 + k] = A
        if n > 0:  # site n is the right site of bond n-1: a term may end here
            for k, B in enumerate(splits[n - 1][1]):
                W[1 + k, :, :, D - 1] = B
        tensors.append(W)

    # The first site can only be in the "nothing started" channel, the last
    # only in the "completed" one.
    tensors[0] = tensors[0][0:1]
    tensors[-1] = tensors[-1][:, :, :, D - 1 : D]
    return tensors


def mpo_norm2(state: mps_module.MPS, mpo: list[np.ndarray]) -> float:
    """Compute ||W |state>||^2 = <state| W^dagger W |state> by transfer sweep.

    Four layers (conjugated MPS, conjugated MPO, MPO, MPS) contracted site by
    site, carrying an environment of shape (chi_bra, D_bra, D_ket, chi_ket).
    W |state> is never formed: it would have bond dimension D * chi, and its
    norm would then cost O((D chi)^3) per site instead of this sweep's
    O(chi^3 D^2 d + chi^2 D^3 d^2).

    Input: state, an MPS; mpo, a list of N tensors from liouvillian_mpo().
    Output: real float, clipped at 0 (it is a squared norm; only rounding
        noise can make the contraction negative).
    """
    if len(mpo) != state.N:
        raise ValueError(f"MPO has {len(mpo)} sites, state has {state.N}")

    # Legs: (bra MPS bond, bra MPO bond, ket MPO bond, ket MPS bond).
    E = np.ones((1, 1, 1, 1), dtype=complex)
    for n in range(state.N):
        A = state.tensors[n]
        W = mpo[n]
        # tensordot, not einsum: see mps.expectation_product_operator.
        T = np.tensordot(E, A.conj(), axes=([0], [0]))       # (w1,w2,a2, j1,a1')
        T = np.tensordot(T, W.conj(), axes=([0, 3], [0, 2]))  # (w2,a2,a1', i,w1')
        T = np.tensordot(T, W, axes=([0, 3], [0, 1]))         # (a2,a1',w1', j2,w2')
        T = np.tensordot(T, A, axes=([0, 3], [0, 1]))         # (a1',w1',w2', a2')
        E = T
    return max(float(E[0, 0, 0, 0].real), 0.0)


def steady_state_residual(
    state: mps_module.MPS,
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    d_site: int = 2,
    mpo_cutoff: float = 0.0,
) -> dict:
    """Measure how far an MPS is from being a steady state of the Liouvillian.

    Computes r = ||L |rho>|| / || |rho> ||, which vanishes exactly at a steady
    state of any symmetry sector and is independent of dt, of the initial
    state and of the schedule that produced the MPS. See the module docstring
    for how to read the number: it is a rate, to be compared against the
    Liouvillian gap rather than against zero.

    Input:
        state: the MPS to test (need not be normalized or canonical).
        H2_terms, H1_terms, L2_terms, L1_terms: the same model terms passed to
            tebd.find_steady_state.
        d_site: physical dimension of one site.
        mpo_cutoff: relative Schmidt cutoff on the Liouvillian MPO; 0 is exact.
    Output: dict with 'residual', 'residual_per_bond' (divided by N-1),
        'residual_norm' (the unnormalized ||L |rho>||), 'state_norm' and
        'mpo_bond_dim'.
    """
    mpo = liouvillian_mpo(
        H2_terms, H1_terms, L2_terms, L1_terms, state.N, d_site, mpo_cutoff
    )
    numerator = np.sqrt(mpo_norm2(state, mpo))
    denominator = np.sqrt(max(state.norm2(), 0.0))
    residual = numerator / denominator if denominator > 0 else float("inf")
    return {
        "residual": float(residual),
        "residual_per_bond": float(residual / max(state.N - 1, 1)),
        "residual_norm": float(numerator),
        "state_norm": float(denominator),
        "mpo_bond_dim": int(max(W.shape[0] for W in mpo)),
    }
