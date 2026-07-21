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
