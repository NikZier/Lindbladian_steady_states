"""Renyi-2 correlator for a local order parameter O.

Definition used here (the "SWSSB"-type Renyi-2 correlator):

    R(i, j) = Tr[ rho A rho^dagger A^dagger ] / Tr[ rho^dagger rho ],   A = O_i O_j^dagger

This is quadratic in rho (not linear like a normal expectation value), so
it is computed via two-replica-style contractions. Two implementations are
provided:

- renyi2_correlator_dense: a direct trace-formula evaluation on a dense
  physical density matrix. This is the reference/definition.
- renyi2_correlator_localvec: an evaluation that only touches rho through
  its local-vec representation (vectorize.physical_to_local_vec) and a
  Kronecker chain of local d^2 x d^2 blocks, one per site. This is the
  form that generalizes to an MPS contraction (each block becomes a local
  MPO tensor acting on the vectorized-rho MPS), so it's the one that
  matters going forward; renyi2_correlator_dense exists to validate it.

Both must agree exactly (up to floating point) for any rho, since they
compute the same quantity in two different index orderings -- that
agreement is the correctness test for the local-vec convention in
vectorize.py.
"""

import numpy as np

from . import mps as mps_module
from . import vectorize


def renyi2_correlator_dense(
    rho: np.ndarray, O: np.ndarray, i: int, j: int, N: int, local_dim: int = 2
) -> float:
    """Reference (dense, trace-formula) evaluation of the Renyi-2 correlator R(i,j).

    Computes Tr[rho A rho^dagger A^dagger] / Tr[rho^dagger rho] with
    A = O_i O_j^dagger, using ordinary dense operators embedded in the full
    N-site physical Hilbert space.

    Input:
        rho: (local_dim^N, local_dim^N) dense density matrix.
        O: (local_dim, local_dim) local order-parameter operator.
        i, j: site indices (may be equal).
        N: number of sites.
        local_dim: physical dimension of one site.
    Output:
        Real float, the correlator value.
    """
    if i == j:
        site_ops = {i: O @ O.conj().T}
    else:
        site_ops = {i: O, j: O.conj().T}
    A = vectorize.embed_product_operator(site_ops, N, local_dim)

    numerator = np.trace(rho @ A @ rho.conj().T @ A.conj().T)
    denominator = np.trace(rho.conj().T @ rho)

    value = numerator / denominator
    assert abs(value.imag) < 1e-8 * max(abs(value.real), 1.0), (
        f"Renyi-2 correlator has non-negligible imaginary part: {value}"
    )
    return value.real


def renyi2_correlator_mps(state: "mps_module.MPS", O: np.ndarray, i: int, j: int) -> float:
    """MPS-native evaluation of the Renyi-2 correlator R(i,j).

    Same quantity and per-site blocks as renyi2_correlator_localvec, but
    contracted directly against an mps.MPS via
    MPS.expectation_product_operator (a polynomial-cost transfer-matrix
    sweep), instead of building the global (local_dim^2)^N operator. This is
    the form TEBD output (an MPS) should be measured with.

    Input:
        state: mps.MPS representing rho (any normalization; the
            normalization cancels between numerator and denominator).
        O: (local_dim, local_dim) local order-parameter operator.
        i, j: site indices (may be equal).
    Output: real float, the correlator value.
    """
    if i == j:
        A_i = O @ O.conj().T
        blocks = {i: np.kron(A_i, A_i.conj())}
    else:
        A_i, A_j = O, O.conj().T
        blocks = {i: np.kron(A_i, A_i.conj()), j: np.kron(A_j, A_j.conj())}

    numerator = state.expectation_product_operator(blocks)
    denominator = state.norm2()

    value = numerator / denominator
    assert abs(value.imag) < 1e-8 * max(abs(value.real), 1.0), (
        f"Renyi-2 correlator has non-negligible imaginary part: {value}"
    )
    return value.real


def renyi2_correlator_localvec(
    rho_lv: np.ndarray, O: np.ndarray, i: int, j: int, N: int, local_dim: int = 2
) -> float:
    """Local-vec evaluation of the Renyi-2 correlator R(i,j).

    Same quantity as renyi2_correlator_dense, but computed entirely from the
    local-vec representation of rho (see vectorize.physical_to_local_vec) via
    a Kronecker chain of local blocks. For sigma = A rho A^dagger with
    A = O_i O_j^dagger (a tensor product across sites), vec(sigma) in the
    local-vec ordering equals a Kronecker chain of per-site blocks
    M_n = A_n (x) conj(A_n) applied to vec(rho); the numerator is then
    vec(sigma)^dagger . vec(rho). This is the form that extends to MPS/MPO
    contraction once TEBD is implemented.

    Input:
        rho_lv: (local_dim^(2N),) local-vec vector (see
            vectorize.physical_to_local_vec).
        O: (local_dim, local_dim) local order-parameter operator.
        i, j: site indices (may be equal).
        N: number of sites.
        local_dim: physical dimension of one site.
    Output:
        Real float, the correlator value.
    """
    vec_dim = local_dim * local_dim  # dimension of one site's local-vec block

    if i == j:
        A_i = O @ O.conj().T
        blocks = {i: np.kron(A_i, A_i.conj())}
    else:
        A_i, A_j = O, O.conj().T
        blocks = {i: np.kron(A_i, A_i.conj()), j: np.kron(A_j, A_j.conj())}

    M = vectorize.embed_product_operator(blocks, N, local_dim=vec_dim)

    numerator = np.vdot(M @ rho_lv, rho_lv)
    denominator = np.vdot(rho_lv, rho_lv)

    value = numerator / denominator
    assert abs(value.imag) < 1e-8 * max(abs(value.real), 1.0), (
        f"Renyi-2 correlator has non-negligible imaginary part: {value}"
    )
    return value.real
