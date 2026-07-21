"""Vectorization conventions and Lindblad generator/gate construction.

Vec convention used throughout this package: for a d x d matrix X,
    vec(X) = X.flatten()   (row-major / numpy default, i.e. vec(X)[d*a+b] = X[a,b])
which satisfies the identity  vec(A X B) = (A (x) B^T) vec(X).

For an N-site chain, the *local-vec* representation of the full density
matrix rho (a 2^N x 2^N matrix) is a 4^N vector obtained by vectorizing
the local 2x2 block at each site and interleaving those blocks site by
site (site_1's 4 components, then site_2's 4 components, ...), rather
than flattening rho globally. This ordering is what lets any operator
that factorizes across sites (nearest-neighbour gates, or products of
single-site operators like O_i O_j) act as a plain Kronecker chain of
local 4x4 (or d^2 x d^2) blocks on the vectorized state -- this is the
representation TEBD gates and MPS tensors will use.
"""

import numpy as np
from scipy.linalg import expm


def vec(rho: np.ndarray) -> np.ndarray:
    """Vectorize a square matrix via row-major flattening: vec(X)[d*a+b] = X[a,b].

    Input: rho, a (d, d) ndarray.
    Output: a (d^2,) ndarray.
    """
    return rho.reshape(-1)


def unvec(v: np.ndarray, d: int) -> np.ndarray:
    """Inverse of vec(): reshape a length-d^2 vector back into a (d, d) matrix.

    Input: v, a (d^2,) ndarray; d, the matrix dimension.
    Output: a (d, d) ndarray.
    """
    return v.reshape(d, d)


def liouvillian_generator(
    H_terms: list[tuple[np.ndarray, float]],
    L_terms: list[tuple[np.ndarray, float]],
    d: int,
) -> np.ndarray:
    """Build the vectorized Lindblad generator L such that d(vec rho)/dt = L @ vec(rho).

    Uses the row-major vec convention (see module docstring):
        L = -i(H (x) I - I (x) H^T)
            + sum_k gamma_k [ L_k (x) conj(L_k)
                               - 1/2 (L_k^dag L_k) (x) I
                               - 1/2 I (x) (L_k^dag L_k)^T ]
    where H = sum_k c_k * H_k over H_terms.

    Input:
        H_terms: list of (operator, coefficient) pairs, each operator (d, d),
                  summed with their coefficients to form the total Hamiltonian.
        L_terms: list of (jump_operator, rate) pairs, each jump_operator (d, d).
                  The dissipator for each term is weighted linearly by `rate`.
        d: local Hilbert space dimension the operators act on.
    Output:
        (d^2, d^2) ndarray, the vectorized generator.
    """
    I_d = np.eye(d, dtype=complex)

    H = np.zeros((d, d), dtype=complex)
    for op, c in H_terms:
        H += c * op

    generator = -1j * (np.kron(H, I_d) - np.kron(I_d, H.T))

    for L, gamma in L_terms:
        Ld_L = L.conj().T @ L
        generator += gamma * (
            np.kron(L, L.conj())
            - 0.5 * np.kron(Ld_L, I_d)
            - 0.5 * np.kron(I_d, Ld_L.T)
        )

    return generator


def reorder_operator_to_local_vec(M: np.ndarray, n_sites: int, local_dim: int = 2) -> np.ndarray:
    """Reorder a vectorized multi-site operator from global-vec to local-vec index order.

    liouvillian_generator(..., d=local_dim**n_sites) built directly from
    physically-embedded n_sites operators produces a matrix whose row and
    column indices are each ordered globally as (bra_1..bra_n, ket_1..ket_n)
    flattened. The local-vec chain representation instead needs each site's
    (bra_n, ket_n) pair kept together and interleaved site by site (see
    physical_to_local_vec). This applies that same index permutation to both
    the row and column legs of an operator (a similarity transform by the
    permutation matrix), so the result acts correctly on local-vec vectors.

    Input:
        M: (local_dim^(2*n_sites), local_dim^(2*n_sites)) ndarray in
            global-vec order.
        n_sites: number of physical sites the operator jointly acts on
            (2 for a bond generator/gate).
        local_dim: physical dimension of one site.
    Output:
        Same-shape ndarray, reordered to local-vec (site-interleaved) order.
    """
    d = local_dim
    tensor = M.reshape((d,) * (2 * n_sites) + (d,) * (2 * n_sites))
    out_perm = [idx for n in range(n_sites) for idx in (n, n_sites + n)]
    in_perm = [2 * n_sites + idx for idx in out_perm]
    tensor = tensor.transpose(out_perm + in_perm)
    dim = d ** (2 * n_sites)
    return tensor.reshape(dim, dim)


def build_bond_generator(
    H2_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    H1_left_terms: list[tuple[np.ndarray, float]] | None = None,
    L1_left_terms: list[tuple[np.ndarray, float]] | None = None,
    H1_right_terms: list[tuple[np.ndarray, float]] | None = None,
    L1_right_terms: list[tuple[np.ndarray, float]] | None = None,
    weight_left: float = 0.5,
    weight_right: float = 0.5,
    d_site: int = 2,
) -> np.ndarray:
    """Assemble the vectorized two-site (bond) Lindblad generator for one bond.

    Combines genuine two-site Hamiltonian/jump terms (already (d_site^2, d_site^2)
    operators on the bond) with single-site terms embedded onto the left/right
    site of the bond and weighted by `weight_left`/`weight_right`. Interior bonds
    should use weight 0.5 on both sides so that summing all bonds of a chain gives
    each interior site its single-site term exactly once; boundary bonds (chain
    ends) should use weight 1.0 on the terminal site since it has only one bond.

    Input:
        H2_terms, L2_terms: two-site (Hamiltonian, jump) terms as (op, coeff) pairs,
            op of shape (d_site^2, d_site^2).
        H1_left_terms, L1_left_terms: single-site terms for the left site of the
            bond, op of shape (d_site, d_site). May be None/empty.
        H1_right_terms, L1_right_terms: single-site terms for the right site.
        weight_left, weight_right: fraction of the single-site term's coefficient
            assigned to this bond.
        d_site: physical dimension of one site (2 for spins).
    Output:
        (d_site^4, d_site^4) ndarray: the vectorized generator, in local-vec
        (site-interleaved) index order, ready to act on two adjacent
        site-blocks of a local-vec chain state (see physical_to_local_vec).
    """
    I_site = np.eye(d_site, dtype=complex)
    H1_left_terms = H1_left_terms or []
    L1_left_terms = L1_left_terms or []
    H1_right_terms = H1_right_terms or []
    L1_right_terms = L1_right_terms or []

    H_terms = list(H2_terms)
    H_terms += [(np.kron(op, I_site), weight_left * c) for op, c in H1_left_terms]
    H_terms += [(np.kron(I_site, op), weight_right * c) for op, c in H1_right_terms]

    L_terms = list(L2_terms)
    L_terms += [(np.kron(op, I_site), weight_left * g) for op, g in L1_left_terms]
    L_terms += [(np.kron(I_site, op), weight_right * g) for op, g in L1_right_terms]

    generator = liouvillian_generator(H_terms, L_terms, d=d_site * d_site)
    # generator so far is in "global two-site vec" order (bra_i,bra_{i+1},ket_i,ket_{i+1});
    # convert to local-vec (site-interleaved) order for use in the chain representation.
    return reorder_operator_to_local_vec(generator, n_sites=2, local_dim=d_site)


def bond_gate(
    H2_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    dt: float,
    H1_left_terms: list[tuple[np.ndarray, float]] | None = None,
    L1_left_terms: list[tuple[np.ndarray, float]] | None = None,
    H1_right_terms: list[tuple[np.ndarray, float]] | None = None,
    L1_right_terms: list[tuple[np.ndarray, float]] | None = None,
    weight_left: float = 0.5,
    weight_right: float = 0.5,
    d_site: int = 2,
) -> np.ndarray:
    """Build the exponentiated two-site (bond) TEBD gate exp(dt * L_bond).

    Thin wrapper around build_bond_generator() + scipy.linalg.expm(). See
    build_bond_generator() for the meaning of all arguments.

    Input: same as build_bond_generator(), plus dt, the (imaginary) time step.
    Output: (d_site^4, d_site^4) ndarray, the vectorized bond propagator.
    """
    generator = build_bond_generator(
        H2_terms,
        L2_terms,
        H1_left_terms,
        L1_left_terms,
        H1_right_terms,
        L1_right_terms,
        weight_left,
        weight_right,
        d_site,
    )
    return expm(dt * generator)


def embed_bond_operator(op: np.ndarray, i: int, N: int, local_dim: int = 2) -> np.ndarray:
    """Embed an operator acting on sites (i, i+1) into the full N-site space.

    Input:
        op: (local_dim^2, local_dim^2) ndarray acting on the bond (i, i+1).
        i: index of the left site of the bond (0-indexed, 0 <= i <= N-2).
        N: total number of sites.
        local_dim: dimension of one site's local space.
    Output:
        (local_dim^N, local_dim^N) ndarray, op tensored with identities on all
        other sites.
    """
    left_dim = local_dim ** i
    right_dim = local_dim ** (N - i - 2)
    return np.kron(np.kron(np.eye(left_dim), op), np.eye(right_dim))


def embed_site_operator(op: np.ndarray, i: int, N: int, local_dim: int = 2) -> np.ndarray:
    """Embed a single-site operator acting on site i into the full N-site space.

    Input:
        op: (local_dim, local_dim) ndarray.
        i: site index (0-indexed).
        N: total number of sites.
        local_dim: dimension of one site's local space.
    Output:
        (local_dim^N, local_dim^N) ndarray, op tensored with identities elsewhere.
    """
    left_dim = local_dim ** i
    right_dim = local_dim ** (N - i - 1)
    return np.kron(np.kron(np.eye(left_dim), op), np.eye(right_dim))


def embed_product_operator(
    site_ops: dict[int, np.ndarray], N: int, local_dim: int = 2
) -> np.ndarray:
    """Embed a tensor-product operator specified by non-overlapping single-site factors.

    Used e.g. to build A = O_i (x) O_j^dagger for i != j as a single global
    operator: pass site_ops = {i: O, j: O.conj().T}.

    Input:
        site_ops: dict mapping site index -> (local_dim, local_dim) local factor.
                  Sites not present are assigned the identity.
        N: total number of sites.
        local_dim: dimension of one site's local space.
    Output:
        (local_dim^N, local_dim^N) ndarray, the Kronecker chain of factors.
    """
    I_site = np.eye(local_dim, dtype=complex)
    factors = [site_ops.get(n, I_site) for n in range(N)]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def physical_to_local_vec(rho: np.ndarray, N: int, local_dim: int = 2) -> np.ndarray:
    """Convert a dense physical density matrix into the local-vec chain representation.

    rho is a (local_dim^N, local_dim^N) matrix indexed by (bra_1..bra_N, ket_1..ket_N).
    This reshapes and transposes it so that each site's own (bra_n, ket_n) pair is
    vectorized and kept together, giving a (local_dim^2)^N vector ordered site by
    site: [site_1's local_dim^2 components, site_2's, ...]. This is the ordering
    under which nearest-neighbour gates and product observables act as a simple
    Kronecker chain (see module docstring).

    Input: rho, (local_dim^N, local_dim^N) ndarray; N, number of sites.
    Output: (local_dim^(2N),) ndarray.
    """
    # index order: (bra_1, ..., bra_N, ket_1, ..., ket_N)
    tensor = rho.reshape((local_dim,) * (2 * N))
    # interleave to (bra_1, ket_1, bra_2, ket_2, ..., bra_N, ket_N)
    perm = [idx for n in range(N) for idx in (n, N + n)]
    tensor = tensor.transpose(perm)
    return tensor.reshape(-1)


def local_vec_to_physical(v: np.ndarray, N: int, local_dim: int = 2) -> np.ndarray:
    """Inverse of physical_to_local_vec(): recover the dense physical density matrix.

    Input: v, (local_dim^(2N),) ndarray in site-interleaved local-vec order;
           N, number of sites.
    Output: (local_dim^N, local_dim^N) ndarray.
    """
    tensor = v.reshape((local_dim,) * (2 * N))
    # tensor axes are currently (bra_1, ket_1, ..., bra_N, ket_N); undo the interleave
    inv_perm = [None] * (2 * N)
    for n in range(N):
        inv_perm[n] = 2 * n
        inv_perm[N + n] = 2 * n + 1
    tensor = tensor.transpose(inv_perm)
    return tensor.reshape(local_dim**N, local_dim**N)


def assemble_chain_generator(
    bond_generators: list[np.ndarray], N: int, local_dim: int = 2
) -> np.ndarray:
    """Sum embedded bond generators into the full chain's local-vec generator.

    Used only for small-N dense validation (not for TEBD, which applies gates
    bond by bond instead of building this explicitly). Each bond_generators[i]
    already includes its share of the single-site terms via build_bond_generator's
    weight_left/weight_right, so summing the embedded bonds gives every site's
    single-site term exactly once.

    Input:
        bond_generators: list of N-1 generators, each (local_dim^2, local_dim^2)
            acting on the vectorized (local-vec) space of one bond, i.e. the
            output of build_bond_generator with d_site = local_dim (physical).
            NOTE: these act on the *vectorized* bond space, so when embedding
            into the chain's local-vec representation the "local_dim" for
            embedding is local_dim**2 (one vectorized site = local_dim^2 numbers).
        N: total number of sites.
        local_dim: physical dimension of one site (2 for spins).
    Output:
        (local_dim^(2N), local_dim^(2N)) ndarray, the full chain generator
        acting on the local-vec representation of vec(rho).
    """
    vec_dim = local_dim * local_dim
    total_dim = vec_dim**N
    generator = np.zeros((total_dim, total_dim), dtype=complex)
    for i, bond_gen in enumerate(bond_generators):
        generator += embed_bond_operator(bond_gen, i, N, local_dim=vec_dim)
    return generator
