"""Finite matrix product state (MPS) representation of a vectorized density
matrix, in the local-vec (site-interleaved) convention from vectorize.py.

Each site's tensor has shape (chi_left, phys_dim, chi_right), where
phys_dim = local_dim**2 is the dimension of that site's own vectorized
Hilbert-Schmidt space (4 for spin-1/2). Open boundary conditions: the first
tensor has chi_left = 1, the last has chi_right = 1.

Note there is no positivity or unitarity constraint anywhere here -- this is
an ordinary (complex, generally non-Hermitian-after-truncation) tensor
network representing a vector in the vectorized Liouville space. "Norm"
below always means the plain Euclidean/Frobenius inner product
<rho|rho> = Tr[rho^dagger rho], computed via transfer-matrix contraction.
"""

import numpy as np
import scipy.linalg

from . import vectorize


def _robust_svd(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD with a fallback driver for non-convergence.

    numpy uses LAPACK's divide-and-conquer gesdd, which is the fast choice but
    occasionally fails to converge on the near-degenerate spectra that
    truncated TEBD produces (a two-site tensor whose singular values span many
    orders of magnitude with clusters of near-ties). gesvd, the older QR
    iteration, is several times slower but much more robust; since the
    fallback only fires on the rare failure it costs nothing on average.

    Input: mat, the (m, n) matrix to decompose.
    Output: (U, S, Vh) as from np.linalg.svd(..., full_matrices=False).
    """
    if not np.all(np.isfinite(mat)):
        raise FloatingPointError(
            "non-finite entries in the two-site tensor before SVD: the state "
            "has diverged (check dt, the jump-operator rates, and whether the "
            "norm is being renormalized each step)"
        )
    try:
        return np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(
            mat, full_matrices=False, lapack_driver="gesvd", check_finite=False
        )


def _truncated_svd(
    mat: np.ndarray, chi_max: int | None, cutoff: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition with optional rank/threshold truncation.

    Input:
        mat: (m, n) ndarray to decompose.
        chi_max: maximum number of singular values to keep (None = no cap).
        cutoff: drop singular values smaller than cutoff * (largest singular
            value); at least one singular value is always kept (None = no cutoff).
    Output:
        (U, S, Vh, S_full): truncated U (m, chi), S (chi,), Vh (chi, n), and
        the full untruncated singular value spectrum S_full (for diagnostics,
        e.g. computing discarded weight).
    """
    U, S, Vh = _robust_svd(mat)
    keep = len(S)
    if chi_max is not None:
        keep = min(keep, chi_max)
    if cutoff is not None and S[0] > 0:
        n_above = int(np.sum(S[:keep] >= cutoff * S[0]))
        keep = min(keep, max(1, n_above))
    return U[:, :keep], S[:keep], Vh[:keep, :], S


class MPS:
    """Finite, open-boundary matrix product state over N sites.

    Attributes:
        tensors: list of N ndarrays, tensors[n] of shape
            (chi_left, phys_dim, chi_right).
        local_dim: physical (spin) dimension of one site (2 for spin-1/2).
        phys_dim: local_dim**2, the dimension of one site's vectorized block.
    """

    def __init__(self, tensors: list[np.ndarray], local_dim: int = 2):
        """Construct an MPS directly from a list of site tensors.

        Input:
            tensors: list of N ndarrays, each (chi_left, phys_dim, chi_right),
                with matching bond dimensions between neighbours and
                chi_left = chi_right = 1 at the two chain ends.
            local_dim: physical dimension of one site (2 for spins); each
                tensor's physical leg must have dimension local_dim**2.
        """
        self.tensors = tensors
        self.local_dim = local_dim
        self.phys_dim = local_dim * local_dim

    @property
    def N(self) -> int:
        """Number of sites in the chain."""
        return len(self.tensors)

    @property
    def bond_dims(self) -> list[int]:
        """List of the N-1 internal bond dimensions, left to right."""
        return [t.shape[2] for t in self.tensors[:-1]]

    def copy(self) -> "MPS":
        """Return a new MPS with independently-copied tensor arrays.

        Output: a new MPS object; mutating it does not affect self.
        """
        return MPS([t.copy() for t in self.tensors], self.local_dim)

    @classmethod
    def product_state(cls, site_vectors: list[np.ndarray], local_dim: int = 2) -> "MPS":
        """Build a bond-dimension-1 MPS from a per-site local-vec vector.

        Input:
            site_vectors: list of N vectors, each of shape (local_dim**2,),
                e.g. the local-vec representation of a single site's 2x2
                density matrix block (see vectorize.vec).
            local_dim: physical dimension of one site.
        Output: an MPS with all bond dimensions equal to 1.
        """
        tensors = [v.reshape(1, len(v), 1).astype(complex) for v in site_vectors]
        return cls(tensors, local_dim)

    @classmethod
    def pure_product_state(cls, kets: list[np.ndarray], local_dim: int = 2) -> "MPS":
        """Build the bond-dimension-1 MPS for a product pure-state density matrix.

        Given per-site kets |psi_n>, forms rho = (x)_n |psi_n><psi_n| and
        represents it in the local-vec chain convention. Used to start TEBD
        from a definite strong-symmetry eigenstate, e.g. all kets = |0> gives
        rho = |0...0><0...0|, which stays in its strong-symmetry sector under
        any parity-commuting Lindbladian.

        Input:
            kets: list of N vectors, each (local_dim,), the single-site pure
                states (need not be normalized; each block is |psi><psi|).
            local_dim: physical dimension of one site.
        Output: an MPS with all bond dimensions equal to 1.
        """
        site_vectors = [vectorize.vec(np.outer(k, np.conj(k))) for k in kets]
        return cls.product_state(site_vectors, local_dim)

    @classmethod
    def maximally_mixed(cls, N: int, local_dim: int = 2) -> "MPS":
        """Build the bond-dimension-1 MPS representing rho = I (unnormalized
        maximally mixed state), a convenient generic initial guess for TEBD.

        Input: N, number of sites; local_dim, physical dimension of one site.
        Output: an MPS with all bond dimensions equal to 1.
        """
        site_vec = vectorize.vec(np.eye(local_dim, dtype=complex))
        return cls.product_state([site_vec] * N, local_dim)

    @classmethod
    def from_dense(
        cls,
        rho: np.ndarray,
        N: int,
        local_dim: int = 2,
        chi_max: int | None = None,
        cutoff: float | None = None,
    ) -> "MPS":
        """Decompose a dense physical density matrix into an MPS via sequential SVD.

        Exponentially expensive in N (builds the full local-vec vector first);
        intended only for validating the MPS/TEBD machinery against exact.py
        on small chains, not for production use.

        Input:
            rho: (local_dim^N, local_dim^N) dense density matrix.
            N: number of sites.
            local_dim: physical dimension of one site.
            chi_max, cutoff: optional truncation applied at each SVD cut.
        Output: an MPS representing rho (exactly, up to any truncation applied).
        """
        d = local_dim * local_dim
        v = vectorize.physical_to_local_vec(rho, N, local_dim)

        tensors = []
        mat = v.reshape(1, d**N)
        chi_l = 1
        for n in range(N - 1):
            mat = mat.reshape(chi_l * d, d ** (N - n - 1))
            U, S, Vh, _ = _truncated_svd(mat, chi_max, cutoff)
            chi_new = U.shape[1]
            tensors.append(U.reshape(chi_l, d, chi_new))
            mat = S[:, None] * Vh
            chi_l = chi_new
        tensors.append(mat.reshape(chi_l, d, 1))
        return cls(tensors, local_dim)

    def to_local_vec(self) -> np.ndarray:
        """Contract the full MPS into its flat local-vec representation.

        Exponentially expensive in N; intended only for small-N validation
        against exact.py, not for production use.

        Output: (phys_dim^N,) ndarray, in the same site-interleaved order as
        vectorize.physical_to_local_vec.
        """
        full = self.tensors[0]
        for n in range(1, self.N):
            full = np.tensordot(full, self.tensors[n], axes=(-1, 0))
        return full.reshape(-1)

    def to_dense(self) -> np.ndarray:
        """Contract the full MPS and convert back to a dense physical density matrix.

        Exponentially expensive in N; intended only for small-N validation
        against exact.py, not for production use.

        Output: (local_dim^N, local_dim^N) dense density matrix.
        """
        v = self.to_local_vec()
        return vectorize.local_vec_to_physical(v, self.N, self.local_dim)

    def apply_two_site_gate(
        self,
        bond: int,
        gate: np.ndarray,
        chi_max: int | None = None,
        cutoff: float | None = None,
    ) -> float:
        """Apply a two-site (bond) gate in place and SVD-truncate the updated bond.

        Contracts gate (phys_dim^2, phys_dim^2) into the tensors at sites
        (bond, bond+1), then splits the result back into two tensors via
        truncated SVD, moving the orthogonality center to site bond+1
        (tensors[bond] becomes left-orthonormal).

        Input:
            bond: index of the left site of the bond (0 <= bond <= N-2).
            gate: (phys_dim^2, phys_dim^2) ndarray in local-vec (site-
                interleaved) order, e.g. from vectorize.bond_gate.
            chi_max, cutoff: optional truncation of the updated bond.
        Output:
            discarded_weight: float in [0, 1), the fraction of squared
            singular-value weight discarded by truncation (0 if untruncated).
        """
        d = self.phys_dim
        A_l = self.tensors[bond]
        A_r = self.tensors[bond + 1]
        chi_l, _, chi_m = A_l.shape
        _, _, chi_r = A_r.shape

        # tensordot (not einsum) throughout: einsum without optimize=True falls
        # through to its scalar C loop instead of dispatching to BLAS, which
        # costs ~10x wall time here at identical flop count.
        theta = np.tensordot(A_l, A_r, axes=(2, 0))  # (l, i, j, r)
        gate_tensor = gate.reshape(d, d, d, d)  # (I, J, i, j)
        theta = np.tensordot(gate_tensor, theta, axes=([2, 3], [1, 2]))  # (I, J, l, r)
        theta = theta.transpose(2, 0, 1, 3)  # (l, I, J, r)

        mat = theta.reshape(chi_l * d, d * chi_r)
        U, S, Vh, S_full = _truncated_svd(mat, chi_max, cutoff)
        chi_new = len(S)

        self.tensors[bond] = U.reshape(chi_l, d, chi_new)
        self.tensors[bond + 1] = (S[:, None] * Vh).reshape(chi_new, d, chi_r)

        total_weight = np.sum(S_full**2)
        kept_weight = np.sum(S**2)
        return 0.0 if total_weight == 0 else 1.0 - kept_weight / total_weight

    def left_canonicalize(self) -> None:
        """Sweep left to right with exact QR decompositions (no truncation).

        After this call, tensors[0..N-2] are left-orthonormal (each
        satisfies sum_{l,i} conj(A[l,i,r]) A[l,i,r'] = delta_{r,r'}); all
        remaining weight/normalization is carried by tensors[N-1]. This is
        the lossless first half of canonicalize()'s compression sweep.
        """
        d = self.phys_dim
        for n in range(self.N - 1):
            A = self.tensors[n]
            chi_l, _, chi_r = A.shape
            mat = A.reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(mat)
            chi_new = Q.shape[1]
            self.tensors[n] = Q.reshape(chi_l, d, chi_new)
            self.tensors[n + 1] = np.tensordot(R, self.tensors[n + 1], axes=(1, 0))

    def canonicalize(self, chi_max: int | None = None, cutoff: float | None = None) -> float:
        """Restore canonical form and (optionally) compress the MPS.

        TEBD gates are generally non-unitary, so repeated local (bond-only)
        SVD truncation in apply_two_site_gate() is not variationally optimal
        once the flanking tensors drift out of canonical form. This restores
        exactness by doing a lossless left-to-right QR sweep
        (left_canonicalize) followed by a right-to-left SVD sweep that both
        right-canonicalizes and truncates optimally to chi_max/cutoff -- the
        standard two-sweep MPS compression algorithm. Should be called
        periodically during TEBD (see tebd.evolve) and before extracting
        observables.

        Input: chi_max, cutoff -- truncation applied during the right sweep.
        Output: max_discarded_weight, the largest per-bond discarded weight
            fraction encountered during the truncating sweep (0 if untruncated).
        """
        d = self.phys_dim
        self.left_canonicalize()

        max_discarded = 0.0
        for n in range(self.N - 1, 0, -1):
            A = self.tensors[n]
            chi_l, _, chi_r = A.shape
            mat = A.reshape(chi_l, d * chi_r)
            U, S, Vh, S_full = _truncated_svd(mat, chi_max, cutoff)
            chi_new = len(S)

            total_weight = np.sum(S_full**2)
            kept_weight = np.sum(S**2)
            if total_weight > 0:
                max_discarded = max(max_discarded, 1.0 - kept_weight / total_weight)

            self.tensors[n] = Vh.reshape(chi_new, d, chi_r)
            US = U * S[None, :]
            self.tensors[n - 1] = np.tensordot(self.tensors[n - 1], US, axes=(2, 0))
        return max_discarded

    def norm2(self) -> float:
        """Compute <rho|rho> = Tr[rho^dagger rho] via transfer-matrix contraction.

        Cost is polynomial in N and the bond dimension (no dense state is
        ever built), unlike to_local_vec(). Used both as a normalization
        target and as a diagnostic (drift here signals truncation error).

        Output: real float (the imaginary part is discarded; it is zero up
        to floating-point noise for any valid MPS).
        """
        return self.overlap(self).real

    def overlap(self, other: "MPS") -> complex:
        """Compute <self|other> = sum of conj(self_entries) * other_entries via
        transfer-matrix contraction, without building either dense vector.

        Input: other, an MPS over the same N sites and phys_dim as self
            (bond dimensions may differ).
        Output: complex overlap value.
        """
        return self.expectation_product_operator({}, other)

    def expectation_product_operator(
        self, site_ops: dict[int, np.ndarray], other: "MPS" = None
    ) -> complex:
        """Contract <self| (x)_n op_n |other> for a tensor-product operator.

        Sites not present in site_ops are treated as identity. With
        site_ops={} this reduces to overlap()/norm2(); this is the
        MPS-native analogue of observables.renyi2_correlator_localvec's
        Kronecker-chain contraction, done as an efficient transfer-matrix
        sweep instead of building the global operator.

        Input:
            site_ops: dict mapping site index -> (phys_dim, phys_dim) local
                operator (e.g. a block kron(A, conj(A)) from observables.py).
            other: MPS to use as the ket; defaults to self (giving
                <self| (x) op_n |self>).
        Output: complex contraction value.
        """
        if other is None:
            other = self
        E = np.ones((1, 1), dtype=complex)
        for n in range(self.N):
            # Explicit pairwise tensordots rather than one optimize=True einsum:
            # einsum would re-solve its contraction path on every site of every
            # sweep, and would contract a dense identity at the (typically all)
            # sites carrying no operator. Cost per site is 2 * chi^3 * d.
            T = np.tensordot(E, self.tensors[n].conj(), axes=(0, 0))  # (b, i, c)
            op = site_ops.get(n)
            if op is None:
                # identity site: contract the bra's physical leg straight
                # through against the ket's.
                E = np.tensordot(T, other.tensors[n], axes=([0, 1], [0, 1]))  # (c, d)
            else:
                T = np.tensordot(T, op, axes=(1, 0))  # (b, c, j)
                E = np.tensordot(T, other.tensors[n], axes=([0, 2], [0, 1]))  # (c, d)
        return E[0, 0]

    def trace(self) -> complex:
        """Compute Tr[rho] via a single (non-conjugated) contraction sweep.

        Uses Tr[X] = vec(I)^T vec(X) (row-major vec convention, see
        vectorize.py), contracting each site's physical leg against
        vec(I_local_dim) directly -- unlike norm2()/overlap(), this does NOT
        conjugate the tensors, since it is a linear functional of rho, not
        an inner product. TEBD only normalizes the state's 2-norm (see
        normalize()), so Tr[rho] is not pinned to 1 during evolution; track
        it as a diagnostic of how physical the (truncated) state still is.

        Output: complex scalar, ideally close to real and nonzero for a
            physical steady state (up to the state's overall 2-norm scale).
        """
        trace_vec = vectorize.vec(np.eye(self.local_dim, dtype=complex))
        v = np.ones((1,), dtype=complex)
        for n in range(self.N):
            v = v @ np.tensordot(self.tensors[n], trace_vec, axes=(1, 0))
        return v[0]

    def normalize(self) -> float:
        """Rescale the state in place so that norm2() == 1.

        Divides the first tensor by sqrt(norm2()). Necessary because TEBD
        gates are non-unitary and will otherwise make the state grow or
        shrink without bound over many steps.

        Output: the norm (sqrt(norm2())) before rescaling, useful as a
            diagnostic (e.g. tracking Tr-like drift over the TEBD run).
        """
        norm = np.sqrt(max(self.norm2(), 0.0))
        if norm > 0:
            self.tensors[0] = self.tensors[0] / norm
        return norm
