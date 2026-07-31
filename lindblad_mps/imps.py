"""Infinite MPS (2-site unit cell, Vidal Gamma-Lambda canonical form) over the
vectorized-density-matrix local-vec space, for TEBD in the thermodynamic
limit.

Why a 2-site unit cell
----------------------
The model applies its jump operators uniformly to every bond (fully
translation-invariant), but the 2nd-order Trotter split needs even/odd bonds,
and the study's two initial states need at least a 2-site cell to be
representable at all: 'zero' (|00...>) fits a 1-site cell, but 'neel'
(|0101...>) only respects 2-site translation invariance. A 2-site cell is
therefore both necessary and standard, not an arbitrary choice.

Why canonicalization needs real fixed points, not the naive local update
--------------------------------------------------------------------------
TEBD gates here are imaginary-time evolution under a Lindbladian, hence
non-unitary. The cheap "simple update" (apply_bond_gate below) truncates a
bond using only its two immediately-adjacent Lambda as a stand-in for the
true environment -- accurate for the bond itself, but not variationally
correct once the state has drifted from the exactly-generated case, exactly
as mps.MPS.apply_two_site_gate is not optimal without periodic
mps.MPS.canonicalize() in the finite case. The infinite analogue
(canonicalize() below) is the Orus-Vidal gauge-fixing algorithm: regauge each
bond using the dominant left/right eigenvectors ("fixed points") of the
actual transfer matrix, found via Arnoldi iteration on a LinearOperator (never
a dense chi^2 x chi^2 matrix -- infeasible past chi ~ 100, and the entire
reason iMPS methods scale at all).

Normalization
-------------
Every apply_bond_gate() call already renormalizes its own bond
(Lambda_new = S / norm(S)) -- this is the complete per-step substitute for
mps.MPS.normalize(); there is no single global scalar norm for an infinite
system to track. Gamma's *overall* magnitude is a separate gauge freedom that
per-step Lambda renormalization does not fix (Gamma -> c*Gamma leaves every
Lambda update unchanged); only canonicalize()'s eta**0.25 rescale corrects it,
by construction of the dominant eigenvalue itself (see canonicalize()'s
docstring for the exponent). Between canonicalize() calls this drift is
bounded but nonzero, which is why canonicalize_every should stay modest and
why itebd.evolve_infinite additionally applies a cheap defensive per-step
Frobenius-norm rescale of Gamma -- plumbing against float overflow, not a
physical correction.
"""

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

from . import mps as mps_module
from . import vectorize


class iMPS:
    """Infinite MPS over a 2-site unit cell (A, B) in Vidal canonical form.

    Attributes:
        Gamma: {'A': ndarray(chi_prev, phys_dim, chi_next), 'B': ndarray(...)}
        Lambda: {'A': ndarray(chi_A,) real >=0, 'B': ndarray(chi_B,) real >=0}
            Lambda['A'] is the bond immediately to the RIGHT of Gamma['A']
            (between A and B); Lambda['B'] is immediately to the right of
            Gamma['B'] (between B and the next cell's A). The chain reads
            ... Lambda_B Gamma_A Lambda_A Gamma_B Lambda_B Gamma_A ...
            so Gamma['A']'s left bond dimension equals len(Lambda['B']) and
            its right bond dimension equals len(Lambda['A']) (and
            symmetrically for Gamma['B']).
        local_dim, phys_dim: as in mps.MPS (phys_dim = local_dim**2 by
            default, but see phys_dim below).
    """

    def __init__(self, Gamma_A, Gamma_B, Lambda_A, Lambda_B, local_dim: int = 2,
                 phys_dim: int | None = None):
        """Construct an iMPS from unit-cell tensors.

        Input:
            Gamma_A, Gamma_B: (chi_left, phys_dim, chi_right) site tensors.
            Lambda_A, Lambda_B: real bond spectra, see the class docstring.
            local_dim: physical dimension of one spin (2 here).
            phys_dim: dimension of the site index, defaulting to local_dim**2
                (one site's vectorized density-matrix block). Passing it
                explicitly lets this class -- and hence canonicalize(),
                transfer_step() and build_transfer_operator(), which all read
                only phys_dim and never local_dim -- be reused for a tensor
                network whose site index is NOT a vectorized 2x2 block. In
                particular ilpdo.iLPDO's X-tensors carry a combined
                (physical, Kraus) index of dimension local_dim*kappa, which
                is generally not a perfect square; reusing this machinery
                avoids duplicating the gauge-fixing, which took three
                separate bug fixes to get right.
        """
        self.Gamma = {"A": Gamma_A, "B": Gamma_B}
        self.Lambda = {
            "A": np.asarray(Lambda_A, dtype=float),
            "B": np.asarray(Lambda_B, dtype=float),
        }
        self.local_dim = local_dim
        self.phys_dim = local_dim * local_dim if phys_dim is None else phys_dim

    @property
    def bond_dims(self) -> dict:
        """{'A': len(Lambda['A']), 'B': len(Lambda['B'])}."""
        return {"A": len(self.Lambda["A"]), "B": len(self.Lambda["B"])}

    def copy(self) -> "iMPS":
        """Return a new iMPS with independently-copied tensor arrays."""
        return iMPS(
            self.Gamma["A"].copy(), self.Gamma["B"].copy(),
            self.Lambda["A"].copy(), self.Lambda["B"].copy(), self.local_dim,
            phys_dim=self.phys_dim,
        )

    @classmethod
    def product_state(cls, vector_A: np.ndarray, vector_B: np.ndarray, local_dim: int = 2) -> "iMPS":
        """Build a bond-dim-1 iMPS from two per-site local-vec vectors.

        Input: vector_A, vector_B, each (local_dim**2,) local-vec vectors
            (see vectorize.vec); local_dim: physical dimension of one site.
        Output: an iMPS with both bond dimensions equal to 1.
        """
        Gamma_A = np.asarray(vector_A, dtype=complex).reshape(1, -1, 1)
        Gamma_B = np.asarray(vector_B, dtype=complex).reshape(1, -1, 1)
        return cls(Gamma_A, Gamma_B, np.array([1.0]), np.array([1.0]), local_dim)

    @classmethod
    def pure_product_state(cls, ket_A: np.ndarray, ket_B: np.ndarray, local_dim: int = 2) -> "iMPS":
        """Build the bond-dim-1 iMPS for a 2-periodic pure-product-state density matrix.

        Given per-site kets |psi_A>, |psi_B>, forms the 2-periodic product
        rho = ... |psi_A><psi_A| (x) |psi_B><psi_B| (x) ... Passing
        ket_A = ket_B = |0> gives the 'zero' start; ket_A=|0>, ket_B=|1>
        gives the 'neel' start -- this is exactly why a 2-site unit cell is
        required (Neel only respects 2-site translation invariance).

        Input: ket_A, ket_B, each (local_dim,) (need not be normalized).
        Output: an iMPS with both bond dimensions equal to 1.
        """
        vA = vectorize.vec(np.outer(ket_A, np.conj(ket_A)))
        vB = vectorize.vec(np.outer(ket_B, np.conj(ket_B)))
        return cls.product_state(vA, vB, local_dim)

    @classmethod
    def maximally_mixed(cls, local_dim: int = 2) -> "iMPS":
        """Build the bond-dim-1 iMPS representing rho = I (unnormalized maximally mixed state)."""
        v = vectorize.vec(np.eye(local_dim, dtype=complex))
        return cls.product_state(v, v, local_dim)

    def canonicalize(
        self,
        chi_max: int | None = None,
        cutoff: float | None = None,
        dense_threshold: int = 100,
        eig_reg_rtol: float = 1e-10,
        pinv_rtol: float = 1e-10,
        lambda_reg: float = 1e-12,
    ) -> dict:
        """Bound-method form of the module-level canonicalize() -- see its docstring."""
        return canonicalize(self, chi_max, cutoff, dense_threshold, eig_reg_rtol,
                            pinv_rtol, lambda_reg)


def right_weighted(Gamma: np.ndarray, Lambda_right: np.ndarray) -> np.ndarray:
    """Theta = Gamma with its trailing bond weighted: Gamma * diag(Lambda_right).

    Used throughout as the tensor to feed into transfer_step /
    transfer_step_transpose: since the chain reads
    ...Lambda Gamma_1 Lambda_1 Gamma_2 Lambda_2..., a tensor already carrying
    its own trailing Lambda ("Theta") is what correctly reproduces the chain
    when composed repeatedly, without separately tracking Lambda insertions
    between transfer steps.

    This is also the correct weighting for the Vidal RIGHT-canonical target
    condition sum_s Gamma_s Lambda^2 Gamma_s^dagger = I: with theta =
    right_weighted(Gamma, Lambda), transfer_step_transpose(theta, I) equals
    exactly that sum (see canonicalize()'s docstring for why the LEFT
    condition needs left_weighted below instead, not this one -- using
    right_weighted for both was this implementation's second bug).
    """
    return Gamma * Lambda_right[None, None, :]


def left_weighted(Gamma: np.ndarray, Lambda_left: np.ndarray) -> np.ndarray:
    """Theta = Gamma with its leading bond weighted: diag(Lambda_left) * Gamma.

    The correct weighting for the Vidal LEFT-canonical target condition
    sum_s Gamma_s^dagger Lambda^2 Gamma_s = I: with theta =
    left_weighted(Gamma, Lambda), transfer_step(theta, I) equals exactly that
    sum. NOT interchangeable with right_weighted -- canonicalize() needs both,
    one per direction (see its docstring); this is easy to get wrong because
    right_weighted alone happens to be internally consistent for one
    direction (transfer_step_transpose) while silently wrong for the other
    (transfer_step), which is exactly the bug this function's introduction
    fixed.
    """
    return Lambda_left[:, None, None] * Gamma


def transfer_step(theta: np.ndarray, X: np.ndarray, op: np.ndarray | None = None) -> np.ndarray:
    """One right-moving transfer step: (chi_l, chi_l) -> (chi_r, chi_r).

        Y[r,r'] = sum_{l,l',s,s'} conj(theta)[l,s,r] op[s,s'] X[l,l'] theta[l',s',r']

    (op=None: identity on s,s', i.e. Y[r,r'] = sum conj(theta)[l,s,r] X[l,l'] theta[l',s,r']).
    Same contraction pattern as mps.MPS.expectation_product_operator's
    per-site loop body, generalized from a (1,1)-seeded environment to an
    arbitrary (chi,chi) one. tensordot throughout, not einsum -- see
    mps.MPS.apply_two_site_gate for why (einsum without optimize=True falls
    through to a scalar C loop at identical flop count).

    Input: theta, (chi_l, phys_dim, chi_r); X, (chi_l, chi_l); op, optional
        (phys_dim, phys_dim) local operator (bra index first, ket index second).
    Output: (chi_r, chi_r) ndarray.
    """
    XT = np.tensordot(X, theta, axes=(1, 0))  # (l, s', r')
    if op is None:
        return np.tensordot(theta.conj(), XT, axes=([0, 1], [0, 1]))  # (r, r')
    OXT = np.tensordot(op, XT, axes=(1, 1))  # (s, l, r')
    return np.tensordot(theta.conj(), OXT, axes=([0, 1], [1, 0]))  # (r, r')


def transfer_step_transpose(theta: np.ndarray, X: np.ndarray, op: np.ndarray | None = None) -> np.ndarray:
    """One left-moving transfer step: (chi_r, chi_r) -> (chi_l, chi_l).

        Y[l,l'] = sum_{r,r',s,s'} theta[l,s,r] op[s,s'] X[r,r'] conj(theta)[l',s',r']

    The genuine index-transpose of transfer_step (not its Hermitian adjoint,
    i.e. NOT derived via .conj().T) -- needed to find a LEFT fixed point as
    the dominant right eigenvector of this transposed map. This transfer
    operator (built from sum_s theta_s (x) conj(theta_s)) is not Hermitian in
    general, so deriving this from transfer_step via an adjoint would
    silently compute the wrong left fixed point; this is implemented as its
    own direct contraction and cross-checked against an independently-coded
    dense reference in tests/test_imps.py.

    Input/Output: as transfer_step, with left/right roles swapped.
    """
    XT = np.tensordot(X, theta.conj(), axes=(1, 2))  # (r, l', s')
    if op is None:
        return np.tensordot(theta, XT, axes=([2, 1], [0, 2]))  # (l, l')
    OXT = np.tensordot(op, XT, axes=(1, 2))  # (s, r, l')
    return np.tensordot(theta, OXT, axes=([1, 2], [0, 1]))  # (l, l')


def build_transfer_operator(
    thetas: tuple[np.ndarray, np.ndarray],
    ops: tuple[np.ndarray | None, np.ndarray | None] | None = None,
    transpose: bool = False,
) -> LinearOperator:
    """Compose two single-site transfer steps into one 2-site unit-cell LinearOperator.

    Input:
        thetas: (theta_first, theta_second), the two sites' Theta tensors in
            FORWARD chain order relative to the bond being solved for (i.e.
            theta_first is the site encountered first when advancing
            rightward away from that bond). For bond 'A' (between Gamma_A and
            Gamma_B) this is (Theta_B, Theta_A) -- advancing right from bond A
            you reach the next Theta_B first, then Theta_A brings you back to
            an instance of bond A one unit cell over. For bond 'B' it is
            (Theta_A, Theta_B).
        ops: optional (op_first, op_second) local-operator dressings, each
            None or (phys_dim, phys_dim); None or omitted for the plain
            (undressed) transfer operator.
        transpose: False builds the RIGHT-moving composed operator (applies
            theta_first's step then theta_second's step) -- its dominant
            eigenvector is the bond's RIGHT fixed point. True builds the
            LEFT-moving composed operator, applying theta_second's TRANSPOSE
            step then theta_first's (the reverse order, since walking
            backward from the bond crosses the sites in the opposite
            sequence) -- its dominant eigenvector is the bond's LEFT fixed
            point. Both act on / return a chi*chi vector representing a
            (chi, chi) matrix, chi = theta_first.shape[0] (== theta_second's
            right bond dimension, by the unit cell's periodic closure).
    Output: a complex LinearOperator of shape (chi**2, chi**2).
    """
    theta_first, theta_second = thetas
    op_first, op_second = ops if ops is not None else (None, None)
    chi = theta_first.shape[0]

    if transpose:
        def matvec(v):
            X = v.reshape(chi, chi)
            Y = transfer_step_transpose(theta_second, X, op_second)
            Y = transfer_step_transpose(theta_first, Y, op_first)
            return Y.reshape(-1)
    else:
        def matvec(v):
            X = v.reshape(chi, chi)
            Y = transfer_step(theta_first, X, op_first)
            Y = transfer_step(theta_second, Y, op_second)
            return Y.reshape(-1)

    return LinearOperator((chi * chi, chi * chi), matvec=matvec, dtype=complex)


ARPACK_V0_SEED = 20250731  # fixed: see leading_eigenpairs' determinism note


def _deterministic_v0(n: int) -> np.ndarray:
    """A fixed, operator-independent ARPACK start vector.

    Drawn from a LOCAL RandomState rather than the global numpy RNG, so it is
    unaffected by (and does not perturb) any seeding the caller does. Complex
    and generic, so it has probability zero of being orthogonal to the
    dominant eigenvector -- unlike an all-ones vector, which can be.
    """
    rng = np.random.RandomState(ARPACK_V0_SEED)
    v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return v0 / np.linalg.norm(v0)


def leading_eigenpairs(
    op: LinearOperator,
    k: int = 2,
    dense_threshold: int = 100,
    tol: float = 1e-12,
    maxiter: int = 2000,
    v0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant k eigenpairs of a transfer LinearOperator, by descending |eigenvalue|.

    Determinism (this is load-bearing -- read before removing v0)
    -------------------------------------------------------------
    ARPACK is called with an EXPLICIT start vector. Without one,
    scipy.sparse.linalg.eigs lets ARPACK generate its own from an internal
    Fortran RNG that np.random.seed() does not reach, and whose state persists
    within a process across calls. Under multiprocessing that made results
    depend on how many jobs a worker had already handled, i.e. on pool
    scheduling.

    That was not cosmetic. canonicalize() truncates, so a different eigenvector
    on a near-degenerate transfer spectrum gives a different truncation, and
    the evolution forks. Measured before this fix: identical settings gave a
    steady state with purity 0.808264 (positive, physical) or 1.246957
    (negative eigenvalues, not a density matrix), the switch appearing as a
    DISCRETE jump mid-run after which the state was stationary again. R was
    unaffected -- 1.3043e-03 either way, to seven digits -- so the correlator
    results never depended on this, but any absolute quantity did.

    Below dense_threshold (op.shape[0], i.e. chi**2 ~ chi <~ 10), builds the
    matrix explicitly (matvec against each standard basis vector -- cheap at
    these sizes) and uses np.linalg.eig: ARPACK requires k < N-1 and is
    unreliable/cannot run at all at these sizes (e.g. chi=1, a bond-dim-1
    product state, has op.shape == (1,1)). Never used at production chi
    (64-256); only a correctness fallback and the tiny-state guard.

    At or above dense_threshold, uses scipy.sparse.linalg.eigs(which='LM').
    `k` defaults above the strict minimum needed (1) as headroom against a
    near-degenerate leading eigenvalue -- e.g. the model's strong Z2 symmetry
    conceivably makes this transfer operator imprimitive; not solved
    analytically here, just given room to be seen (inspect the returned
    eigenvalues' magnitudes if leading_eigenpairs is used at k=1 in new code).
    On ArpackNoConvergence, retries once with a larger maxiter/ncv before
    falling back to the dense path.

    Input: op; k, number of eigenpairs wanted; dense_threshold; tol, maxiter:
        ARPACK convergence controls; v0, explicit start vector (defaults to a
        fixed deterministic one -- pass your own only to probe start-vector
        sensitivity, never in production).
    Output: (eigenvalues, eigenvectors) -- eigenvectors as columns (each
        reshape-able to (chi, chi) via .reshape(chi, chi)), sorted by
        descending |eigenvalue|. May return fewer than k pairs if op is too
        small.
    """
    n = op.shape[0]

    def _dense_fallback(k_want: int) -> tuple[np.ndarray, np.ndarray]:
        dense = np.column_stack([op.matvec(e) for e in np.eye(n, dtype=complex)])
        vals, vecs = np.linalg.eig(dense)
        order = np.argsort(-np.abs(vals))
        k_eff = min(k_want, n)
        return vals[order[:k_eff]], vecs[:, order[:k_eff]]

    if n <= dense_threshold:
        return _dense_fallback(k)

    k_eff = min(k, n - 2)
    ncv = min(n, max(4 * k_eff + 1, 20))
    start = _deterministic_v0(n) if v0 is None else v0
    try:
        vals, vecs = eigs(op, k=k_eff, which="LM", tol=tol, maxiter=maxiter,
                          ncv=ncv, v0=start)
    except ArpackNoConvergence:
        try:
            vals, vecs = eigs(
                op, k=k_eff, which="LM", tol=tol, maxiter=maxiter * 5,
                ncv=min(n, max(8 * k_eff + 1, 40)), v0=start,
            )
        except ArpackNoConvergence:
            return _dense_fallback(k)

    order = np.argsort(-np.abs(vals))
    return vals[order], vecs[:, order]


def theta_pair(state: iMPS, bond: str) -> tuple[np.ndarray, np.ndarray]:
    """Right-weighted Theta tensors, in forward chain order, for a given bond.

    Input: state; bond, 'A' or 'B'.
    Output: (theta_first, theta_second), see build_transfer_operator().
    """
    Theta_A = right_weighted(state.Gamma["A"], state.Lambda["A"])
    Theta_B = right_weighted(state.Gamma["B"], state.Lambda["B"])
    if bond == "A":
        return Theta_B, Theta_A
    if bond == "B":
        return Theta_A, Theta_B
    raise ValueError(f"bond must be 'A' or 'B', got {bond!r}")


def _clip_and_sqrt_factor(M: np.ndarray, rtol: float) -> np.ndarray:
    """Hermitian-PSD square-root factor L (L @ L.conj().T ~= M), with regularization.

    Symmetrizes M (guards floating-point asymmetry), eigendecomposes, and
    floors eigenvalues at rtol * (largest eigenvalue) before taking a square
    root. Perron-Frobenius guarantees the true fixed point is PSD (the
    transfer operator has the Kronecker-with-conjugate/Choi structure of a
    completely-positive map, real-positive leading eigenvalue and PSD leading
    eigenvectors, regardless of the underlying dynamics being unitary or
    dissipative) -- the floor guards solver noise and near-singular fixed
    points, not a physical assumption being relaxed.

    Input: M, (chi, chi) approximately Hermitian PSD; rtol, relative eigenvalue floor.
    Output: L, (chi, chi) with L @ L.conj().T approximating the clipped M.
    """
    Msym = 0.5 * (M + M.conj().T)
    w, v = np.linalg.eigh(Msym)
    wmax = max(float(w.max()), 0.0)
    floor = rtol * wmax if wmax > 0 else 0.0
    w_clipped = np.clip(w, floor, None)
    return v * np.sqrt(w_clipped)[None, :]


def _normalize_fixed_point(vec: np.ndarray, chi: int) -> np.ndarray:
    """Reshape an eigenvector to (chi,chi), Hermitize, fix its sign, unit-trace it.

    Eigenvectors from eig/eigs are defined only up to an arbitrary complex
    scale; Perron-Frobenius says the true fixed point is a positive multiple
    of a PSD matrix, so once symmetrized its trace is real and of one sign --
    fixing that sign and rescaling to unit trace picks a canonical
    representative before the sqrt-factor regularization is applied.
    """
    M = vec.reshape(chi, chi)
    M = 0.5 * (M + M.conj().T)
    tr = np.trace(M).real
    if tr < 0:
        M, tr = -M, -tr
    return M / tr if tr > 0 else np.eye(chi, dtype=complex) / chi


def _single_site_transfer_operator(theta: np.ndarray, transpose: bool = False) -> LinearOperator:
    """LinearOperator for ONE (not composed) transfer_step/transfer_step_transpose
    application, undressed. Used only by canonicalize() on the coarse-grained
    unit-cell tensor (see its docstring for why the merged picture needs a
    single-step operator rather than build_transfer_operator's two-step
    composition)."""
    chi = theta.shape[0]
    if transpose:
        def matvec(v):
            return transfer_step_transpose(theta, v.reshape(chi, chi), None).reshape(-1)
    else:
        def matvec(v):
            return transfer_step(theta, v.reshape(chi, chi), None).reshape(-1)
    return LinearOperator((chi * chi, chi * chi), matvec=matvec, dtype=complex)


def _merge_unit_cell(state: iMPS) -> np.ndarray:
    """Contract Gamma_A, Lambda_A, Gamma_B into one physical-dim-phys_dim**2 tensor.

    Output: (chi_B, phys_dim**2, chi_B) ndarray, chi_B = len(Lambda['B']) on
        both legs (the unit cell's only external bond, shared by construction
        since it is literally the same array on both sides -- see
        apply_bond_gate's docstring for the same periodicity fact). NOT
        weighted by the outer Lambda['B'] -- callers that need the
        transfer-ready Theta form should apply right_weighted() themselves.
    """
    d = state.phys_dim
    merged = np.tensordot(
        state.Gamma["A"] * state.Lambda["A"][None, None, :], state.Gamma["B"], axes=(2, 0)
    )  # (chi_B, d, d, chi_B)
    chi_l, chi_r = merged.shape[0], merged.shape[3]
    return merged.reshape(chi_l, d * d, chi_r)


def canonicalize(
    state: iMPS,
    chi_max: int | None = None,
    cutoff: float | None = None,
    dense_threshold: int = 100,
    eig_reg_rtol: float = 1e-10,
    pinv_rtol: float = 1e-10,
    lambda_reg: float = 1e-12,
) -> dict:
    """Restore true Vidal canonical form and (optionally) compress, in place.

    Why coarse-grain first: an earlier version of this function gauge-fixed
    bond A and bond B independently, each from its own composed 2-site
    transfer operator (exactly mirroring how a bond in a FINITE chain
    separates two independent semi-infinite regions). That is wrong here:
    because the unit cell wraps around periodically, fixing bond B's gauge
    also rewrites Gamma_A's LEFT leg -- a leg that bond A's OWN transfer
    operator depends on (it is built from Gamma_A's full tensor, both legs,
    not only the leg bond A's own fix touches) -- so the two bonds' fixes
    interfere through a leg neither one treats as "theirs", and the identity-
    environment proof (valid for a genuinely isolated bond) does not carry
    over. This was caught empirically: after "fixing" both bonds, inserting
    the identity into either bond's freshly-rebuilt transfer operator did not
    reproduce the identity, and repeated sweeps diverged rather than
    converged.

    The fix used here has no such coupling because it leaves no second bond
    to interfere: merge Gamma_A, Lambda_A, Gamma_B into ONE coarse-grained
    tensor of physical dimension phys_dim**2 (_merge_unit_cell) -- now there
    is exactly one bond (Lambda['B'], appearing on both the left and right
    leg of the merged tensor by the unit cell's own periodicity), so this
    reduces to the standard, textbook single-bond infinite-MPS
    canonicalization: find that one bond's dominant left/right ENVIRONMENT
    fixed points and gauge-fix via rho_left_env = X^dagger X,
    rho_right_env = Y Y^dagger, M = X @ diag(Lambda_B) @ Y,
    SVD(M) = U S V^dagger, Lambda_B_new = S / norm(S), and rebuild the merged
    tensor's two legs via regularized pinv (Xinv @ U on the right leg,
    Vh @ Yinv on the left leg -- both act on the SAME tensor's two different
    legs, unlike the old per-bond version, which is exactly what removes the
    cross-bond coupling).

    The two environments need DIFFERENT Lambda-weightings of the merged
    tensor, not the same one (this implementation's second, more subtle bug,
    caught because the LEFT target below held only approximately, not
    exactly, until fixed): the Vidal LEFT-canonical target condition
    sum_s Gamma_s^dagger Lambda^2 Gamma_s = I is exactly
    transfer_step(left_weighted(merged, Lambda_B), I) -- forward, on a
    LEFT-weighted theta -- while the RIGHT-canonical target
    sum_s Gamma_s Lambda^2 Gamma_s^dagger = I is exactly
    transfer_step_transpose(right_weighted(merged, Lambda_B), I) -- backward,
    on a RIGHT-weighted theta. rho_left_env therefore comes from the forward
    map applied to left_weighted(merged, ...), and rho_right_env from the
    backward map applied to right_weighted(merged, ...) -- using
    right_weighted for both (the natural first guess, since it is the only
    weighting transfer_step_transpose needs correctly) silently satisfies
    only the right-canonical condition.

    This is provably exact (not merely close): the new reduced left
    environment is (Xinv U)^dagger rho_left_env (Xinv U)
    = U^dagger X^-dagger (X^dagger X) X^-1 U = U^dagger U = I exactly (U has
    orthonormal columns from SVD, an isometry even under truncation), and
    symmetrically V^dagger V = I for the new right environment -- and because
    there is only one bond, this is the WHOLE unit cell's canonical form, not
    an approximation awaiting a second bond's correction.

    Finally, split the gauge-fixed merged tensor back into Gamma_A, Lambda_A,
    Gamma_B via an ordinary (optionally truncated) SVD: since both of the
    merged tensor's OUTER legs are now exactly canonical, this internal split
    needs no separate environment-finding step, exactly as an ordinary
    two-site block splits after a finite-chain TEBD gate
    (mps.MPS.apply_two_site_gate).

    Global rescale: divide BOTH Gamma_A and Gamma_B by eta**0.25, eta = the
    mean of the merged bond's (pre-rescale) left/right environment
    eigenvalues (should already agree to solver tolerance). One application
    of the merged transfer step touches the merged tensor once conjugated,
    once not, i.e. touches Gamma_A and Gamma_B once each (conjugated once,
    unconjugated once) -- so rescaling each by eta**-0.25 divides the
    eigenvalue by (eta**-0.5)*(eta**-0.5) = eta**-1, giving
    eta_new = eta * eta**-1 = 1 exactly. This -- not a per-Trotter-step
    operation -- is the infinite-system analogue of mps.MPS.normalize(); see
    the module docstring for why it can't be done as cheaply as the finite
    case's per-step version.

    Input: chi_max, cutoff: truncation applied to both the outer (unit-cell)
        and inner (Lambda_A) SVDs; dense_threshold, eig_reg_rtol, pinv_rtol:
        see leading_eigenpairs / _clip_and_sqrt_factor.
    Output: dict with 'eigenvalue_A' (alias for the merged bond's left
        environment eigenvalue, pre-rescale -- kept under this name for
        itebd.evolve_infinite's drift diagnostic), 'eigenvalue_left_env',
        'eigenvalue_right_env', 'max_discarded_weight' (max over the outer
        and inner SVDs).
    """
    d = state.phys_dim
    merged = _merge_unit_cell(state)
    chi = merged.shape[0]
    Lambda_B_old = state.Lambda["B"]
    # Forward (transfer_step) and backward (transfer_step_transpose) need
    # DIFFERENT weightings -- the Vidal LEFT-canonical target
    # sum_s Gamma_s^dagger Lambda^2 Gamma_s = I is transfer_step(left_weighted
    # theta, I), while the RIGHT-canonical target sum_s Gamma_s Lambda^2
    # Gamma_s^dagger = I is transfer_step_transpose(right_weighted theta, I).
    # Using right_weighted for both (this implementation's second bug, after
    # the L/R environment swap) satisfies the second identity but not the
    # first -- verified by the failure of the first identity to hold even
    # directly, before any further transform was applied.
    theta_L = left_weighted(merged, Lambda_B_old)
    theta_R = right_weighted(merged, Lambda_B_old)

    op_fwd = _single_site_transfer_operator(theta_L, transpose=False)
    op_bwd = _single_site_transfer_operator(theta_R, transpose=True)
    vals_fwd, vecs_fwd = leading_eigenpairs(op_fwd, k=2, dense_threshold=dense_threshold)
    vals_bwd, vecs_bwd = leading_eigenpairs(op_bwd, k=2, dense_threshold=dense_threshold)
    eta_left_env, eta_right_env = vals_fwd[0], vals_bwd[0]
    rho_left_env = _normalize_fixed_point(vecs_fwd[:, 0], chi)
    rho_right_env = _normalize_fixed_point(vecs_bwd[:, 0], chi)

    X = _clip_and_sqrt_factor(rho_left_env, eig_reg_rtol).conj().T  # X^dagger X ~= rho_left_env
    Y = _clip_and_sqrt_factor(rho_right_env, eig_reg_rtol)          # Y Y^dagger ~= rho_right_env

    M = X @ (Lambda_B_old[:, None] * Y)
    U, S, Vh, S_full = mps_module._truncated_svd(M, chi_max, cutoff)
    norm = np.linalg.norm(S)
    Lambda_B_new = S / norm if norm > 0 else S

    Xinv = np.linalg.pinv(X, rcond=pinv_rtol)
    Yinv = np.linalg.pinv(Y, rcond=pinv_rtol)
    merged_new = np.tensordot(merged, Xinv @ U, axes=(2, 0))   # right leg
    merged_new = np.tensordot(Vh @ Yinv, merged_new, axes=(1, 0))  # left leg

    outer_total = np.sum(S_full**2)
    outer_kept = np.sum(S**2)
    max_discarded = 0.0 if outer_total == 0 else 1.0 - outer_kept / outer_total

    # Inner split, in TRUE Vidal form. The outer bond weighting must be applied
    # before the SVD and divided back out after -- exactly as apply_bond_gate
    # does. SVD-ing merged_new directly (the first version of this function)
    # makes the MERGED cell canonical but leaves the individual Gamma_A /
    # Lambda_A / Gamma_B decomposition NOT a Vidal decomposition: measured on
    # a converged state, the merged-cell canonical conditions held to 1e-7
    # while the per-site conditions were violated by ~1e+7. Nothing in the
    # evolution notices (the gates and the merged canonicalization only ever
    # use the composite), but any per-SITE quantity is then wrong --
    # iobservables.correlator_profile sweeps site by site and inserts operators
    # at single sites, so it silently returned a badly wrong correlator.
    chi_B_new = len(Lambda_B_new)
    theta = (Lambda_B_new[:, None, None] * merged_new) * Lambda_B_new[None, None, :]
    mat = theta.reshape(chi_B_new, d, d, chi_B_new).reshape(chi_B_new * d, d * chi_B_new)
    Ua, Sa, Vha, Sa_full = mps_module._truncated_svd(mat, chi_max, cutoff)
    norm_a = np.linalg.norm(Sa)
    Lambda_A_new = Sa / norm_a if norm_a > 0 else Sa
    chi_A_new = len(Lambda_A_new)

    inner_total = np.sum(Sa_full**2)
    inner_kept = np.sum(Sa**2)
    if inner_total > 0:
        max_discarded = max(max_discarded, 1.0 - inner_kept / inner_total)

    # Undo the outer weighting with a floored reciprocal (Lambda_B_new is
    # diagonal, so an elementwise floor is the right regularization -- see
    # apply_bond_gate), and distribute sqrt(norm_a) onto both halves so that
    # Gamma_A . Lambda_A . Gamma_B reconstructs merged_new exactly despite
    # Lambda_A being unit-normalized.
    sqrt_norm_a = np.sqrt(norm_a) if norm_a > 0 else 1.0
    Lo_inv = 1.0 / np.maximum(Lambda_B_new, lambda_reg)
    Gamma_A_new = (Ua.reshape(chi_B_new, d, chi_A_new) * sqrt_norm_a) * Lo_inv[:, None, None]
    Gamma_B_new = (Vha.reshape(chi_A_new, d, chi_B_new) * sqrt_norm_a) * Lo_inv[None, None, :]

    # eta_left_env/eta_right_env were measured BEFORE the X,Y fix even ran --
    # using them here silently ignores the additional scale the inner split
    # above introduces (sqrt_norm_a), so the rescale undershoots and eta
    # drifts to some other value instead of 1 (caught empirically: repeated
    # canonicalize() calls did not converge to eta=1, they converged to a
    # STABLE WRONG value instead, since each call rescaled by the wrong
    # amount consistently). Measure eta fresh on merged_new -- which already
    # reflects everything up to and including the inner split, since
    # Gamma_A_new/Gamma_B_new (with sqrt_norm_a applied) reconstruct
    # merged_new exactly regardless of how the split's overall scale is
    # divided between the two factors -- and use that instead.
    theta_new_left = left_weighted(merged_new, Lambda_B_new)
    op_fresh = _single_site_transfer_operator(theta_new_left, transpose=False)
    vals_fresh, _ = leading_eigenpairs(op_fresh, k=1, dense_threshold=dense_threshold)
    eta = max(vals_fresh[0].real, 1e-300)
    scale = eta**0.25
    state.Gamma["A"] = Gamma_A_new / scale
    state.Gamma["B"] = Gamma_B_new / scale
    state.Lambda["A"] = Lambda_A_new
    state.Lambda["B"] = Lambda_B_new

    return {
        "eigenvalue_A": eta_left_env,
        "eigenvalue_left_env": eta_left_env,
        "eigenvalue_right_env": eta_right_env,
        "max_discarded_weight": max_discarded,
    }


def apply_bond_gate(
    state: iMPS,
    bond: str,
    gate: np.ndarray,
    chi_max: int | None = None,
    cutoff: float | None = None,
    lambda_reg: float = 1e-12,
) -> float:
    """Vidal simple update: apply a bond gate in place, cheaply, without full regauging.

    Contracts diag(Lambda_outer) Gamma_1 diag(Lambda_mid) Gamma_2
    diag(Lambda_outer) -- the SAME Lambda array on both outer legs: by 2-site
    periodicity, the bond immediately left of Gamma_A and the bond
    immediately right of Gamma_B genuinely are the same array (Lambda['B']
    when bond='A'; Lambda['A'] when bond='B'), not a bug -- applies the gate,
    truncated-SVDs, sets Lambda_mid_new = S/norm(S), and undoes the outer
    dressing via a FLOORED ELEMENTWISE RECIPROCAL (Lambda_outer is diagonal,
    so there is nothing to condition-check beyond individual near-zero
    entries -- unlike canonicalize()'s dense pinv, which is needed there
    because X/Y are not diagonal).

    Not variationally optimal in general (uses only the two adjacent Lambda
    as a stand-in for the true environment) -- see the module docstring;
    correct periodically via canonicalize().

    Input: bond, 'A' or 'B'; gate, (phys_dim**2, phys_dim**2) in local-vec order.
    Output: discarded_weight fraction, as mps.MPS.apply_two_site_gate.
    """
    left_name, right_name = ("A", "B") if bond == "A" else ("B", "A")
    outer_name = "B" if bond == "A" else "A"
    Lo = state.Lambda[outer_name]
    Lm = state.Lambda[bond]
    A_l = state.Gamma[left_name]
    A_r = state.Gamma[right_name]
    d = state.phys_dim
    chi_outer = len(Lo)

    theta = Lo[:, None, None] * A_l * Lm[None, None, :]
    theta = np.tensordot(theta, A_r, axes=(2, 0))  # (chi_outer, d, d, chi_r_of_Ar)
    theta = theta * Lo[None, None, None, :]

    gate_tensor = gate.reshape(d, d, d, d)  # (I, J, i, j)
    theta = np.tensordot(gate_tensor, theta, axes=([2, 3], [1, 2]))  # (I, J, l, r)
    theta = theta.transpose(2, 0, 1, 3)  # (l, I, J, r)

    mat = theta.reshape(chi_outer * d, d * chi_outer)
    U, S, Vh, S_full = mps_module._truncated_svd(mat, chi_max, cutoff)
    chi_new = len(S)
    norm = np.linalg.norm(S)
    Lambda_new = S / norm if norm > 0 else S

    # U @ diag(S) @ Vh == mat exactly, but Lambda_new is the UNIT-NORMALIZED
    # S/norm, so Gamma_left_new . Lambda_new . Gamma_right_new would
    # reconstruct mat/norm, not mat, unless the missing factor of norm is put
    # back -- same issue as canonicalize()'s inner split, same fix: distribute
    # sqrt(norm) onto both Gamma updates so the reconstruction is exact again.
    sqrt_norm = np.sqrt(norm) if norm > 0 else 1.0
    Lo_inv = 1.0 / np.maximum(Lo, lambda_reg)
    state.Gamma[left_name] = U.reshape(chi_outer, d, chi_new) * Lo_inv[:, None, None] * sqrt_norm
    state.Gamma[right_name] = Vh.reshape(chi_new, d, chi_outer) * Lo_inv[None, None, :] * sqrt_norm
    state.Lambda[bond] = Lambda_new

    total_weight = np.sum(S_full**2)
    kept_weight = np.sum(S**2)
    return 0.0 if total_weight == 0 else 1.0 - kept_weight / total_weight
