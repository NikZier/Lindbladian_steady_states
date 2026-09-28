"""Dense (exact-diagonalization) reference implementation, for validating
the local-vec/TEBD-gate machinery in vectorize.py on small chains (N <~ 6).

Everything here works with ordinary dense physical operators/density
matrices ((2^N, 2^N) arrays) and the global (non site-interleaved) vec
convention from vectorize.liouvillian_generator, which is dimension-agnostic
and therefore reusable at d = 2^N.
"""

import numpy as np
import scipy.linalg as sla

from . import vectorize


def build_hamiltonian(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    N: int,
    local_dim: int = 2,
) -> np.ndarray:
    """Build the dense total Hamiltonian for an N-site chain.

    Sums each two-site term over every nearest-neighbour bond (i, i+1) and
    each single-site term over every site.

    Input:
        H2_terms: list of (op, coeff) pairs, op of shape (local_dim^2, local_dim^2),
            applied uniformly to every bond (translation-invariant coupling).
        H1_terms: list of (op, coeff) pairs, op of shape (local_dim, local_dim),
            applied uniformly to every site.
        N: number of sites.
        local_dim: physical dimension of one site.
    Output:
        (local_dim^N, local_dim^N) dense Hamiltonian.
    """
    dim = local_dim**N
    H = np.zeros((dim, dim), dtype=complex)
    for op, c in H2_terms:
        for i in range(N - 1):
            H += c * vectorize.embed_bond_operator(op, i, N, local_dim)
    for op, c in H1_terms:
        for i in range(N):
            H += c * vectorize.embed_site_operator(op, i, N, local_dim)
    return H


def build_jump_operators(
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    local_dim: int = 2,
) -> list[tuple[np.ndarray, float]]:
    """Build the dense list of (jump_operator, rate) pairs for an N-site chain.

    Each two-site jump term is instantiated once per bond, each single-site
    jump term once per site, all embedded into the full N-site Hilbert space.

    Input: same term/N/local_dim conventions as build_hamiltonian().
    Output: list of (op, rate) pairs, op of shape (local_dim^N, local_dim^N).
    """
    jump_ops = []
    for op, gamma in L2_terms:
        for i in range(N - 1):
            jump_ops.append((vectorize.embed_bond_operator(op, i, N, local_dim), gamma))
    for op, gamma in L1_terms:
        for i in range(N):
            jump_ops.append((vectorize.embed_site_operator(op, i, N, local_dim), gamma))
    return jump_ops


def steady_state(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    local_dim: int = 2,
    degeneracy_tol: float = 1e-6,
) -> np.ndarray:
    """Solve for the steady-state density matrix of an N-site chain by dense ED.

    Builds the full physical Liouvillian generator, finds its eigenvector
    with eigenvalue closest to zero (the steady state, since generator maps
    trace-preserving so 0 is always an eigenvalue for a valid Lindbladian),
    reshapes it into a density matrix, Hermitizes, and normalizes the trace
    to 1. Raises if the zero eigenvalue looks degenerate (ambiguous steady
    state), since picking a single eigenvector would then be arbitrary.

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: interaction terms in the same
            (operator, coefficient) format as build_hamiltonian/build_jump_operators.
        N: number of sites.
        local_dim: physical dimension of one site.
        degeneracy_tol: minimum gap required between the smallest and second
            smallest |eigenvalue| to trust a unique steady state.
    Output:
        (local_dim^N, local_dim^N) dense steady-state density matrix.
    """
    dim = local_dim**N
    H = build_hamiltonian(H2_terms, H1_terms, N, local_dim)
    jump_ops = build_jump_operators(L2_terms, L1_terms, N, local_dim)

    generator = vectorize.liouvillian_generator(
        [(H, 1.0)], jump_ops, d=dim
    )

    eigvals, eigvecs = sla.eig(generator)
    order = np.argsort(np.abs(eigvals))
    smallest, second_smallest = np.abs(eigvals[order[0]]), np.abs(eigvals[order[1]])
    if second_smallest - smallest < degeneracy_tol:
        raise RuntimeError(
            f"Steady state may be degenerate: smallest |eigenvalues| are "
            f"{smallest:.2e} and {second_smallest:.2e} (gap < {degeneracy_tol:.1e})."
        )

    rho = vectorize.unvec(eigvecs[:, order[0]], dim)
    rho = (rho + rho.conj().T) / 2  # enforce Hermiticity (removes numerical noise)
    rho = rho / np.trace(rho)
    return rho


def sector_indices(N: int, charge: int = 1, local_dim: int = 2) -> np.ndarray:
    """Vec indices of the (charge, charge) strong-symmetry block.

    A strong Z_2 symmetry P = Z_1...Z_N assigns every computational basis state
    |a> the charge (-1)^(number of ones in a). The Liouvillian block-decomposes
    by the PAIR (q_L, q_R) of bra and ket charges; this returns the indices of
    the diagonal block q_L = q_R = charge, in the row-major vec convention
    vec(|a><b|) = dim * a + b (see vectorize).

    Input: N, number of sites; charge, the sector's parity eigenvalue (+1/-1);
        local_dim, physical dimension (only 2 is meaningful for Z_2 parity).
    Output: 1-D int array of vec indices belonging to the sector.
    """
    if local_dim != 2:
        raise ValueError(f"Z_2 parity sectors need local_dim=2, got {local_dim}")
    dim = local_dim**N
    popcount = np.array([bin(k).count("1") for k in range(dim)])
    keep = np.flatnonzero(np.where(popcount % 2, -1, 1) == charge)
    return (keep[:, None] * dim + keep[None, :]).reshape(-1)


def sector_steady_state(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    charge: int = 1,
    local_dim: int = 2,
    leakage_tol: float = 1e-12,
) -> dict:
    """Exact steady state inside one strong-symmetry sector, by dense null space.

    steady_state() cannot be used when the model has a strong symmetry: the zero
    eigenvalue of the full Liouvillian is then degenerate (one steady state per
    sector) and it refuses on exactly that condition. Restricting to the
    (charge, charge) block makes the null vector unique again, and it is the
    sector every TEBD run in this repo actually lives in.

    The sector-leakage assertion is not a formality. It checks numerically that
    the generator does not connect the block to its complement, which is the
    premise of the whole study -- and for a HAMILTONIAN perturbation it is the
    live test of [H, P] = 0, since a term violating the strong symmetry would
    show up here and nowhere else in the pipeline.

    Two gap numbers are returned and they are not interchangeable:

      * 'gap_rate' -- min |Re lambda| over the non-stationary eigenvalues of the
        restricted generator: the relaxation rate of the slowest mode OF THIS
        N-SITE CHAIN.
        WARNING, measured 2026-09-19: for the SWSSB baseline this is NOT the
        rate a thermodynamic-limit schedule should be sized from. At small N it
        returns the BASELINE diffusive mode -- 3.0626 at N=4 and 1.2785 at N=6,
        both independent of epsilon to 4 digits over epsilon in
        {0.2, 0.1, 0.05} -- whereas the mode that actually sets the infinite
        system's relaxation is the perturbation-induced one at rate ~ epsilon^2.
        The two are unrelated because A + A -> 0 is diffusion-limited and
        critical: a 4- or 6-site chain has no room for the diffusive slowing
        that makes the infinite chain slow. Size infinite-system schedules from
        the pair-creation rate instead (see CLAUDE.md Traps 2 and 5).
      * 'gap_singular' -- the smallest nonzero singular value, which is what the
        earlier experiment-local version of this function returned. For a
        NON-NORMAL generator (every Liouvillian here) it is not the relaxation
        rate and must not be used as one; it is kept for continuity.

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: (operator, coefficient) pairs in
            the same convention as build_hamiltonian/build_jump_operators. Rates
            are honoured as given (unlike the earlier version, which forced 1.0).
        N: number of sites -- dense, so keep to N <= 6.
        charge: the sector's parity eigenvalue (+1 for both starts used here).
        local_dim: physical dimension of one site.
        leakage_tol: tolerance on the strong-symmetry block structure.
    Output: dict with
        'rho' -- the normalized (local_dim^N, local_dim^N) sector steady state,
        'gap_rate', 'gap_singular' -- see above,
        'eigenvalues' -- the restricted generator's spectrum, sorted by |lambda|
            (the first entry is the stationary mode, numerically ~0),
        'indices' -- the sector's vec indices.
    """
    dim = local_dim**N
    H = build_hamiltonian(H2_terms, H1_terms, N, local_dim)
    jump_ops = build_jump_operators(L2_terms, L1_terms, N, local_dim)
    generator = vectorize.liouvillian_generator([(H, 1.0)], jump_ops, d=dim)

    idx = sector_indices(N, charge, local_dim)
    outside = np.setdiff1d(np.arange(dim * dim), idx)
    leakage = float(np.abs(generator[np.ix_(outside, idx)]).max())
    assert leakage < leakage_tol, (
        f"sector leakage {leakage:.2e} > {leakage_tol:.1e}: the generator "
        f"connects the (charge={charge}) block to its complement, i.e. some "
        f"term does not commute with the strong symmetry P = Z_1...Z_N"
    )

    block = generator[np.ix_(idx, idx)]

    # Null vector by SVD: better conditioned than picking an eigenvector of a
    # near-defective generator, and it reproduces the earlier implementation.
    _, S, Vh = np.linalg.svd(block)
    v = np.zeros(dim * dim, dtype=complex)
    v[idx] = Vh[-1].conj()
    rho = vectorize.unvec(v, dim)
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)

    eigvals = sla.eigvals(block)
    order = np.argsort(np.abs(eigvals))
    eigvals = eigvals[order]
    # eigvals[0] is the stationary mode; the gap is the slowest decay among the rest.
    gap_rate = float(np.min(np.abs(eigvals[1:].real))) if eigvals.size > 1 else float("nan")

    return {
        "rho": rho,
        "gap_rate": gap_rate,
        "gap_singular": float(S[-2]) if S.size > 1 else float("nan"),
        "eigenvalues": eigvals,
        "indices": idx,
    }
