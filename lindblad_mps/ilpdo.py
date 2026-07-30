"""Manifestly-positive infinite ansatz: rho = X X^dagger (locally purified).

Why this exists
---------------
The vectorized infinite ansatz (imps.py + itebd.py) represents rho directly as
an MPS over the doubled space. Nothing in that scheme constrains rho to the
physical manifold, and measured on this model it does leave it: the converged
state is genuinely *stationary* (residual tiny and exactly N-independent) but
is NOT a density matrix -- 41-72% negative eigenvalue weight, growing with
system size, plus negative trace and a max eigenvalue of rho/Tr above 1.

The consequence was a clean factor of ~2: R came out at 1.987 +- 0.092 times
the ED-validated finite-chain answer, across all 10 independent L'' samples.
Projecting the state onto the positive manifold multiplied R by 0.506/0.507/
0.508 at three window sizes and landed within 2% of the finite value -- so the
positivity loss *is* the factor of 2. It is also structural, not a resolution
problem: R was identical to five digits across chi in {16,32,64} and
canonicalize_every in {2,10}.

The fix is to make the unphysical manifold unreachable by construction. Write

    rho = X X^dagger

with X an MPS carrying, per site, a physical index (dimension local_dim) and a
Kraus/ancilla index (dimension kappa). Then rho is positive semidefinite for
*any* X whatsoever, and Tr[rho] = ||X||^2 >= 0, so the trace pathology goes
away too -- and unlike the vectorized ansatz, the trace can actually be
normalized (for an infinite chain, unit trace per site and unit 2-norm per
site are incompatible; the vectorized code has to pick the 2-norm).

Why the Lindblad gate can be applied at all
--------------------------------------------
Evolution preserves the form only if the two-site gate exp(dt*L_bond) is
completely positive, so that it has a Kraus representation
rho -> sum_mu K_mu rho K_mu^dagger; then X -> K_mu X simply enlarges the Kraus
index. Verified numerically for this model before building any of this (see
kraus_operators, which re-asserts it every call): the Choi matrix is Hermitian
to 1e-16 with minimum eigenvalue -1e-16, i.e. PSD up to rounding, at every dt
in the schedule.

Its Kraus rank is 8 of a possible 16, so the Kraus index grows by a factor 8
per gate. Truncating it every step is therefore mandatory rather than
optional, and kappa_max is a genuine convergence parameter -- it limits how
much mixedness the ansatz can represent, and is a *different* approximation
from truncating a bond. Scan it.

Note also that truncating X minimizes ||Delta X||, not ||Delta rho||, so this
representation is less efficient per unit bond dimension than truncating rho
directly. Expect to need chi_X somewhat above sqrt(chi_rho).

Measurement reuses the vectorized path
---------------------------------------
to_vectorized_imps() converts X into exactly the imps.iMPS object the existing
(and now cross-validated) iobservables.correlator_profile /
correlation_length consume, at bond dimension chi_X**2. That is deliberate:
writing a fresh four-layer contraction would re-open precisely the class of
convention bug that twice produced a silently wrong correlator here.
"""

import numpy as np
import scipy.linalg

from . import imps as imps_module
from . import mps as mps_module
from . import tebd, vectorize


class iLPDO:
    """Infinite locally-purified density operator over a 2-site unit cell.

    rho = X X^dagger, where X is an infinite MPS whose site tensors each carry
    a physical index and a Kraus (ancilla) index.

    Attributes:
        X: {'A': ndarray(chi_left, local_dim, kappa, chi_right), 'B': ...}
            The purification tensors. Bond conventions match imps.iMPS: the
            chain reads ... X_A X_B X_A ..., so X['A']'s right bond dimension
            equals X['B']'s left bond dimension and vice versa (wrapping).
        local_dim: physical dimension of one spin (2 here).
    """

    def __init__(self, X_A: np.ndarray, X_B: np.ndarray, local_dim: int = 2):
        self.X = {"A": X_A, "B": X_B}
        self.local_dim = local_dim

    @property
    def bond_dims(self) -> dict:
        """{'A': right bond of X_A, 'B': right bond of X_B}."""
        return {"A": self.X["A"].shape[3], "B": self.X["B"].shape[3]}

    @property
    def kraus_dims(self) -> dict:
        """{'A': kappa of X_A, 'B': kappa of X_B}."""
        return {"A": self.X["A"].shape[2], "B": self.X["B"].shape[2]}

    def copy(self) -> "iLPDO":
        """Return a new iLPDO with independently-copied tensor arrays."""
        return iLPDO(self.X["A"].copy(), self.X["B"].copy(), self.local_dim)

    @classmethod
    def pure_product_state(cls, ket_A: np.ndarray, ket_B: np.ndarray,
                           local_dim: int = 2) -> "iLPDO":
        """Bond-dim-1, kappa-1 iLPDO for a 2-periodic pure product state.

        A pure state needs no ancilla at all (kappa=1): rho = |psi><psi| is
        already X X^dagger with X = |psi>. Passing ket_A=ket_B=|0> gives the
        'zero' start and ket_A=|0>, ket_B=|1> the 'neel' start, matching
        imps.iMPS.pure_product_state.

        Input: ket_A, ket_B, each (local_dim,); local_dim.
        Output: an iLPDO with all bond and Kraus dimensions 1.
        """
        XA = np.asarray(ket_A, dtype=complex).reshape(1, local_dim, 1, 1)
        XB = np.asarray(ket_B, dtype=complex).reshape(1, local_dim, 1, 1)
        return cls(XA, XB, local_dim)

    @classmethod
    def maximally_mixed(cls, local_dim: int = 2) -> "iLPDO":
        """Bond-dim-1 iLPDO for rho = I/local_dim per site.

        The maximally mixed state needs a full ancilla: X = I/sqrt(local_dim)
        with the Kraus index running over local_dim values, since
        X X^dagger = I/local_dim.
        """
        X = np.eye(local_dim, dtype=complex).reshape(1, local_dim, local_dim, 1)
        X = X / np.sqrt(local_dim)
        return cls(X.copy(), X.copy(), local_dim)

    def as_imps(self) -> "imps_module.iMPS":
        """View X as an ordinary iMPS with combined (physical, Kraus) site index.

        Fuses each site's (local_dim, kappa) pair into one index of dimension
        local_dim*kappa. That is exactly what makes imps.canonicalize() and
        the transfer-operator machinery reusable here without modification:
        they read only iMPS.phys_dim, never local_dim. Positivity of
        rho = X X^dagger is gauge invariant, so canonicalizing X cannot break
        it.

        Output: an imps.iMPS whose Gamma tensors are views/reshapes of X, with
            unit Lambda (X carries no separate bond spectra of its own until
            canonicalize() assigns them).
        """
        shapes = {k: self.X[k].shape for k in ("A", "B")}
        G = {k: self.X[k].reshape(shapes[k][0], shapes[k][1] * shapes[k][2], shapes[k][3])
             for k in ("A", "B")}
        chi_A = shapes["A"][3]
        chi_B = shapes["B"][3]
        return imps_module.iMPS(
            G["A"], G["B"], np.ones(chi_A), np.ones(chi_B),
            local_dim=self.local_dim,
            phys_dim=shapes["A"][1] * shapes["A"][2],
        )

    def trace_per_cell(self) -> float:
        """Tr[rho] contribution per unit cell: ||X||^2 over the cell.

        Positive by construction -- the property the vectorized ansatz lacks,
        where Tr[rho] came out negative. Not a normalization the evolution
        pins by itself; see itebd_lpdo.evolve_infinite_lpdo.
        """
        return float(sum(np.vdot(self.X[k], self.X[k]).real for k in ("A", "B")))


def kraus_operators(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    dt: float,
    d_site: int = 2,
    tol: float = 1e-12,
    check: bool = True,
) -> list[np.ndarray]:
    """Kraus decomposition of the two-site bond gate exp(dt * L_bond).

    Builds the same bond generator the vectorized codes use -- via
    vectorize.liouvillian_generator with weight_left = weight_right = 0.5, the
    uniform interior weighting itebd.unit_cell_gate also uses, so the
    generator matches the finite chain's exactly -- exponentiates it, reshapes
    the superoperator into its Choi matrix, and eigendecomposes:

        Choi = sum_i w_i |v_i><v_i|,   K_i = sqrt(w_i) * reshape(v_i)

    The vec convention is row-major (see vectorize.py), so the superoperator
    acts as S[(i,j),(k,l)] with vec(rho)_{i*d+j} = rho_{ij}, and the Choi
    matrix is the [(i,k),(j,l)] regrouping. This index shuffle is easy to get
    wrong and is therefore *verified numerically* rather than reasoned about:
    with check=True the reconstruction sum_i kron(K_i, conj(K_i)) == S is
    asserted, which pins the convention (measured agreement ~1e-16).

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: model terms, as in
            tebd.build_bond_gates / itebd.unit_cell_gate.
        dt: (imaginary) time step for this gate.
        d_site: physical dimension of one site.
        tol: relative Choi-eigenvalue threshold; smaller channels are dropped.
        check: assert complete positivity, trace preservation and exact
            reconstruction. Cheap (a 16x16 problem) -- leave it on.
    Output: list of (d_site**2, d_site**2) Kraus operators acting on the bond.
    """
    d2 = d_site * d_site
    generator = vectorize.build_bond_generator_global = None  # placeholder guard
    # Build the bond generator in GLOBAL vec order (bra/ket of the whole
    # two-site block), which is what liouvillian_generator returns and what
    # the Choi reshuffle below assumes -- NOT the local-vec (site-interleaved)
    # order that vectorize.build_bond_generator converts to for MPS use.
    I_site = np.eye(d_site, dtype=complex)
    H_terms = list(H2_terms)
    H_terms += [(np.kron(op, I_site), 0.5 * c) for op, c in (H1_terms or [])]
    H_terms += [(np.kron(I_site, op), 0.5 * c) for op, c in (H1_terms or [])]
    L_terms = list(L2_terms)
    L_terms += [(np.kron(op, I_site), 0.5 * g) for op, g in (L1_terms or [])]
    L_terms += [(np.kron(I_site, op), 0.5 * g) for op, g in (L1_terms or [])]
    generator = vectorize.liouvillian_generator(H_terms, L_terms, d=d2)

    S = scipy.linalg.expm(dt * generator)
    S4 = S.reshape(d2, d2, d2, d2)                          # [i, j, k, l]
    choi = S4.transpose(0, 2, 1, 3).reshape(d2 * d2, d2 * d2)  # [(i,k),(j,l)]

    w, V = np.linalg.eigh(0.5 * (choi + choi.conj().T))
    w_max = max(float(w.max()), 0.0)
    keep = w > tol * max(w_max, 1e-300)
    ops = [np.sqrt(w[i]) * V[:, i].reshape(d2, d2) for i in np.nonzero(keep)[0]]

    if check:
        assert w.min() > -1e-8 * max(w_max, 1e-300), (
            f"bond gate is not completely positive: min Choi eigenvalue {w.min():.3e}"
        )
        rec = sum(np.kron(K, K.conj()) for K in ops)
        rel = np.linalg.norm(rec - S) / max(np.linalg.norm(S), 1e-300)
        assert rel < 1e-8, f"Kraus reconstruction of the gate failed: rel err {rel:.3e}"
        tp = sum(K.conj().T @ K for K in ops)
        tp_err = np.linalg.norm(tp - np.eye(d2))
        assert tp_err < 1e-6, f"gate is not trace preserving: ||sum K^dag K - I|| = {tp_err:.3e}"

    return ops


def apply_bond_gate_cp(
    state: iLPDO,
    bond: str,
    kraus_ops: list[np.ndarray],
    chi_max: int | None = None,
    kappa_max: int | None = None,
    cutoff: float | None = None,
) -> dict:
    """Apply a CP bond gate to the LPDO in place, truncating bond and Kraus legs.

    rho -> sum_mu K_mu rho K_mu^dagger with rho = X X^dagger is simply
    X -> {K_mu X}_mu, i.e. the gate's Kraus index mu becomes an additional
    ancilla index. Since the LPDO form keeps one Kraus leg per site, mu is
    absorbed into the LEFT site's Kraus leg (kappa_left -> kappa_left *
    len(kraus_ops)); the choice of side is a convention, and alternating bonds
    means both sites' legs grow over a full Trotter step either way.

    With Kraus rank 8 for this model, kappa grows x8 per gate, so both
    truncations below run every call -- this is not an optional refinement.

    Truncation, in order:
      1. bond: SVD across the bond, keep chi_max / cutoff (via
         mps._truncated_svd, reusing its gesdd->gesvd fallback).
      2. Kraus legs: for each of the two sites, reshape to
         (kappa) x (everything else), SVD, keep the largest kappa_max singular
         directions. This minimizes ||Delta X|| for that leg, which is NOT the
         same as minimizing ||Delta rho|| -- hence kappa_max is a convergence
         parameter to scan, not a free knob.

    Input:
        state: iLPDO to update in place.
        bond: 'A' (between X_A and X_B) or 'B' (between X_B and the next cell's X_A).
        kraus_ops: list of (d**2, d**2) bond Kraus operators, from kraus_operators().
        chi_max, cutoff: bond truncation.
        kappa_max: cap on each site's Kraus dimension (None = no cap).
    Output: dict with 'discarded_weight' (bond) and 'kraus_discarded_weight'
        (max over the two sites).
    """
    left, right = ("A", "B") if bond == "A" else ("B", "A")
    XL, XR = state.X[left], state.X[right]
    d = state.local_dim
    chi_l, _, kL, chi_m = XL.shape
    _, _, kR, chi_r = XR.shape
    n_mu = len(kraus_ops)

    # Contract the two sites across the bond: (chi_l, d, kL, d, kR, chi_r)
    theta = np.tensordot(XL, XR, axes=(3, 0))

    # Apply the gate's Kraus operators on the two physical indices. K has
    # shape (d*d, d*d) = ((s1 s2), (t1 t2)); mu indexes the list.
    K = np.stack(kraus_ops, axis=0).reshape(n_mu, d, d, d, d)   # [mu, s1, s2, t1, t2]
    # theta axes: 0 chi_l, 1 t1, 2 kL, 3 t2, 4 kR, 5 chi_r
    theta = np.tensordot(K, theta, axes=([3, 4], [1, 3]))
    # -> [mu, s1, s2, chi_l, kL, kR, chi_r]
    theta = theta.transpose(3, 1, 4, 0, 2, 5, 6)
    # -> [chi_l, s1, kL, mu, s2, kR, chi_r]; fuse (kL, mu) into the left Kraus leg
    theta = theta.reshape(chi_l, d, kL * n_mu, d, kR, chi_r)

    # Bond SVD: split (chi_l, s1, kL*mu) from (s2, kR, chi_r)
    kL_new = kL * n_mu
    mat = theta.reshape(chi_l * d * kL_new, d * kR * chi_r)
    U, S, Vh, S_full = mps_module._truncated_svd(mat, chi_max, cutoff)
    chi_new = len(S)
    total = np.sum(S_full**2)
    kept = np.sum(S**2)
    discarded = 0.0 if total == 0 else 1.0 - kept / total

    # Distribute the singular values symmetrically. rho = X X^dagger is
    # invariant under a unitary acting on the Kraus/bond gauge, so any split
    # is admissible; sqrt/sqrt keeps both tensors comparably scaled.
    sqrtS = np.sqrt(S)
    XL_new = (U * sqrtS[None, :]).reshape(chi_l, d, kL_new, chi_new)
    XR_new = (sqrtS[:, None] * Vh).reshape(chi_new, d, kR, chi_r)

    # Kraus truncation, per site.
    kraus_disc = 0.0
    XL_new, dL = _truncate_kraus(XL_new, kappa_max, cutoff)
    XR_new, dR = _truncate_kraus(XR_new, kappa_max, cutoff)
    kraus_disc = max(dL, dR)

    state.X[left] = XL_new
    state.X[right] = XR_new
    _match_kraus_dims(state)
    return {"discarded_weight": discarded, "kraus_discarded_weight": kraus_disc}


def _match_kraus_dims(state: iLPDO) -> None:
    """Zero-pad the two sites' Kraus legs to a common dimension, in place.

    The gate's mu index is absorbed into the LEFT site only, so a gate
    application leaves kappa_A != kappa_B. That is fine for rho itself but
    breaks the reuse of imps.canonicalize: iMPS assumes ONE physical dimension
    for the whole unit cell (its _merge_unit_cell reshapes both sites with the
    same d), and a mismatch surfaces as an unrelated-looking reshape error
    deep inside the gauge fixing.

    Padding with zeros is exact rather than an approximation:
    rho = sum_k X_k X_k^dagger is unchanged by appending all-zero X_k terms.
    """
    kA = state.X["A"].shape[2]
    kB = state.X["B"].shape[2]
    if kA == kB:
        return
    target = max(kA, kB)
    for key in ("A", "B"):
        X = state.X[key]
        k = X.shape[2]
        if k == target:
            continue
        pad = np.zeros((X.shape[0], X.shape[1], target - k, X.shape[3]), dtype=X.dtype)
        state.X[key] = np.concatenate([X, pad], axis=2)


def _truncate_kraus(
    X: np.ndarray, kappa_max: int | None, cutoff: float | None
) -> tuple[np.ndarray, float]:
    """Truncate one site's Kraus leg to kappa_max by SVD against the other legs.

    Input: X, (chi_l, d, kappa, chi_r); kappa_max; cutoff.
    Output: (X_truncated, discarded_weight_fraction).
    """
    chi_l, d, kappa, chi_r = X.shape
    if kappa_max is None or kappa <= kappa_max:
        return X, 0.0

    # (kappa) x (chi_l, d, chi_r)
    mat = X.transpose(2, 0, 1, 3).reshape(kappa, chi_l * d * chi_r)
    U, S, Vh, S_full = mps_module._truncated_svd(mat, kappa_max, cutoff)
    total = np.sum(S_full**2)
    kept = np.sum(S**2)
    discarded = 0.0 if total == 0 else 1.0 - kept / total

    # Keep the retained Kraus directions: the new leg is the SVD's own index.
    reduced = (S[:, None] * Vh).reshape(len(S), chi_l, d, chi_r)
    return reduced.transpose(1, 2, 0, 3), discarded


def to_vectorized_imps(state: iLPDO) -> "imps_module.iMPS":
    """Convert rho = X X^dagger into the vectorized iMPS the observables consume.

    Builds, per site,

        M[(a,a'), (s,s'), (b,b')] = sum_k X[a,s,k,b] * conj(X[a',s',k,b'])

    so the resulting iMPS has bond dimension chi_X**2 and site dimension
    local_dim**2, exactly the object imps.iMPS / iobservables expect. The
    returned state is NOT canonical -- call imps.canonicalize() on it before
    measuring (iobservables requires a canonical state, and says so).

    The (s,s') ordering must match vectorize's row-major vec convention
    (vec(rho)_{s*d+s'} = rho_{s,s'}). That is pinned by a test comparing a
    converted product state against imps.iMPS.pure_product_state rather than
    by reasoning about it -- the same class of convention detail that twice
    produced a silently wrong correlator in the vectorized path.

    Input: state, an iLPDO.
    Output: an imps.iMPS representing the same rho, bond dimension chi_X**2.
    """
    G = {}
    for key in ("A", "B"):
        X = state.X[key]
        chi_l, d, _, chi_r = X.shape
        # sum over the Kraus index; keep bra/ket copies of every other leg
        M = np.einsum("askb,ptkq->apstbq", X, X.conj(), optimize=True)
        G[key] = M.reshape(chi_l * chi_l, d * d, chi_r * chi_r)

    chi_A = state.X["A"].shape[3] ** 2
    chi_B = state.X["B"].shape[3] ** 2
    return imps_module.iMPS(
        G["A"], G["B"], np.ones(chi_A), np.ones(chi_B), local_dim=state.local_dim
    )


def canonicalize(
    state: iLPDO,
    chi_max: int | None = None,
    cutoff: float | None = None,
    **kwargs,
) -> dict:
    """Gauge-fix X in place by reusing imps.canonicalize on the fused view.

    X viewed with its (physical, Kraus) indices fused is an ordinary iMPS
    (see iLPDO.as_imps), so the existing Orus-Vidal gauge fixing applies
    verbatim -- and must be reused rather than reimplemented: that routine
    needed three separate bug fixes (L/R environment swap, left- vs
    right-weighted theta, and a non-Vidal inner split) before it was correct.
    Positivity of rho = X X^dagger is gauge invariant, so this cannot move the
    state off the physical manifold.

    Note the Lambda spectra produced here are Schmidt values of X, not of rho.
    That is the right thing for truncating X, and is exactly why LPDO
    truncation is not optimal for rho (see the module docstring).

    Input: state; chi_max, cutoff: truncation; kwargs forwarded to
        imps.canonicalize (dense_threshold, eig_reg_rtol, pinv_rtol, lambda_reg).
    Output: imps.canonicalize's diagnostics dict.
    """
    view = state.as_imps()
    diag = imps_module.canonicalize(view, chi_max, cutoff, **kwargs)

    # Push the gauge-fixed tensors (with Lambda absorbed so X alone carries
    # the state) back into LPDO shape.
    d = state.local_dim
    for key, lam_left in (("A", view.Lambda["B"]), ("B", view.Lambda["A"])):
        G = imps_module.left_weighted(view.Gamma[key], lam_left)
        chi_l, fused, chi_r = G.shape
        state.X[key] = G.reshape(chi_l, d, fused // d, chi_r)

    # Pin the overall scale: Tr[rho] = 1 per unit cell. as_imps() hands
    # imps.canonicalize a view with Lambda = ones, which is not a canonical
    # starting point, so its eta**0.25 rescale is computed against an
    # arbitrary scale and leaves X's magnitude undetermined. Left alone this
    # drifts hard -- it drove singular values past 1e154 and overflowed the
    # discarded-weight sums. rho = X X^dagger is only defined up to this
    # scale (R is a ratio of quadratics in rho, hence invariant), so fixing
    # it here costs nothing physical and keeps the arithmetic in range.
    tr = state.trace_per_cell()
    if tr > 0 and np.isfinite(tr):
        scale = np.sqrt(tr) ** 0.5
        for key in ("A", "B"):
            state.X[key] = state.X[key] / scale
    return diag
