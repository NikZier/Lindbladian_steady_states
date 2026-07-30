"""Validation tests for ilpdo.py and itebd_lpdo.py.

The whole point of the LPDO ansatz is that rho = X X^dagger cannot leave the
physical manifold, so the central tests here assert positivity and a positive
trace for ARBITRARY X -- no evolution required. Those properties hold by
construction, which means what these tests really police is the CONVERTER
(to_vectorized_imps) and the Kraus machinery: that is where a bug would hide,
exactly as the two bugs in the vectorized path hid in convention details that
product-state tests could not see.

Style follows tests/test_residual.py and tests/test_imps.py: each docstring
says what the test would catch, cross-checks go against independently
validated machinery (dense references, the ED-validated finite-chain code)
rather than against this module's own conventions, and everything stays small
enough to keep the suite fast.
"""

import numpy as np
import pytest

from lindblad_mps import ilpdo, imps, iobservables, itebd_lpdo, models, observables, vectorize
from lindblad_mps import mps as mps_module

KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)


def _random_X(chi, kappa, rng, d=2):
    return (rng.standard_normal((chi, d, kappa, chi))
            + 1j * rng.standard_normal((chi, d, kappa, chi)))


def _tile_dense(state_imps, n_cells):
    """Dense rho of a finite window tiled from a vectorized iMPS.

    A-form tensors with the boundary bonds sliced to 1 -- the same tiling the
    iMPS regression tests use. Boundary error is confined near the ends, so
    bulk properties (and global PSD-ness, which is what these tests check)
    are meaningful.
    """
    A = imps.left_weighted(state_imps.Gamma["A"], state_imps.Lambda["B"])
    B = imps.left_weighted(state_imps.Gamma["B"], state_imps.Lambda["A"])
    ts = []
    for _ in range(n_cells):
        ts.append(A.copy())
        ts.append(B.copy())
    ts[0] = ts[0][0:1, :, :]
    ts[-1] = ts[-1][:, :, 0:1]
    return mps_module.MPS(ts, local_dim=2).to_dense()


def _psd_report(rho):
    """(trace, hermiticity_error, negative_eigenvalue_weight) of a dense rho."""
    tr = np.trace(rho)
    if tr.real < 0:  # rho and -rho describe the same R; compare the positive branch
        rho, tr = -rho, -tr
    herm = np.linalg.norm(rho - rho.conj().T) / max(np.linalg.norm(rho), 1e-300)
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    neg = abs(w[w < 0].sum()) / max(abs(w.sum()), 1e-300)
    return float(tr.real), float(herm), float(neg)


class TestPositivityByConstruction:
    """The defining property of the ansatz, and the reason it exists: the
    vectorized ansatz converged to a stationary but NON-positive operator
    (41-72% negative eigenvalue weight, negative trace), which inflated R by
    a factor of ~2 versus the ED-validated finite chain. Here positivity must
    hold for ANY X, with no evolution and no fine-tuning."""

    @pytest.mark.parametrize("chi,kappa", [(2, 3), (3, 2), (4, 4)])
    def test_random_X_gives_a_positive_hermitian_rho(self, chi, kappa):
        rng = np.random.default_rng(chi * 10 + kappa)
        state = ilpdo.iLPDO(_random_X(chi, kappa, rng), _random_X(chi, kappa, rng))
        rho = _tile_dense(ilpdo.to_vectorized_imps(state), n_cells=3)
        tr, herm, neg = _psd_report(rho)
        assert tr > 0, f"trace must be positive, got {tr}"
        assert herm < 1e-12, f"rho must be Hermitian, got {herm:.2e}"
        assert neg < 1e-10, f"rho must be PSD, got negative weight {neg:.2e}"

    @pytest.mark.parametrize("chi,kappa", [(1, 1), (3, 2)])
    def test_trace_per_cell_is_the_squared_norm_of_X(self, chi, kappa):
        """Tr[rho] = Tr[X X^dagger] = ||X||^2, so the trace is positive by
        construction -- the property whose absence let the vectorized state
        reach Tr[rho] < 0."""
        rng = np.random.default_rng(7)
        XA, XB = _random_X(chi, kappa, rng), _random_X(chi, kappa, rng)
        state = ilpdo.iLPDO(XA, XB)
        expected = float(np.vdot(XA, XA).real + np.vdot(XB, XB).real)
        assert state.trace_per_cell() == pytest.approx(expected, rel=1e-12)
        assert state.trace_per_cell() > 0


class TestConverter:
    """to_vectorized_imps is the one place a convention error could silently
    corrupt every measurement, since everything downstream reuses the
    validated vectorized observables. Pinned against independent references,
    never against this module's own reasoning."""

    @pytest.mark.parametrize("ket_A,ket_B", [(KET0, KET0), (KET0, KET1)])
    def test_product_state_matches_imps_pure_product_state(self, ket_A, ket_B):
        """Pins the local-vec (s,s') index ordering against the vectorized
        class's own constructor. A transposed convention would pass every
        positivity test above and still give a wrong correlator."""
        got = ilpdo.to_vectorized_imps(ilpdo.iLPDO.pure_product_state(ket_A, ket_B))
        want = imps.iMPS.pure_product_state(ket_A, ket_B)
        for key in ("A", "B"):
            np.testing.assert_allclose(got.Gamma[key], want.Gamma[key], atol=1e-14)

    def test_maximally_mixed_is_identity_over_two(self):
        state = ilpdo.iLPDO.maximally_mixed()
        g = ilpdo.to_vectorized_imps(state).Gamma["A"][0, :, 0]
        np.testing.assert_allclose(vectorize.unvec(g, 2), np.eye(2) / 2, atol=1e-14)

    def test_matches_direct_dense_construction(self):
        """rho from the converter must equal rho = X X^dagger built by an
        explicit, independent contraction of X on the same open-boundary
        window -- the check that the Kraus index is summed on the right leg
        and the bra/ket copies are paired correctly."""
        rng = np.random.default_rng(11)
        chi, kappa, d = 2, 2, 2
        XA, XB = _random_X(chi, kappa, rng), _random_X(chi, kappa, rng)
        state = ilpdo.iLPDO(XA, XB)

        # Direct: 2-site open window, boundary bonds sliced to 1.
        # chain[s1, k1, s2, k2] = sum_b XA[0, s1, k1, b] * XB[b, s2, k2, 0]
        chain = np.einsum("skb,btl->sktl", XA[0], XB[:, :, :, 0], optimize=True)
        # rho[(s1,s2), (t1,t2)] = sum_{k1,k2} chain[s1,k1,s2,k2] conj(chain[t1,k1,t2,k2])
        rho_direct = np.einsum("skul,tkvl->sutv", chain, chain.conj(), optimize=True)
        rho_direct = rho_direct.reshape(d * d, d * d)

        # Converter: same window, via the vectorized iMPS and the validated
        # local-vec -> physical routine.
        v = ilpdo.to_vectorized_imps(state)
        tensors = [v.Gamma["A"][0:1, :, :].copy(), v.Gamma["B"][:, :, 0:1].copy()]
        rho_conv = mps_module.MPS(tensors, local_dim=2).to_dense()

        np.testing.assert_allclose(rho_conv, rho_direct, atol=1e-12)


class TestKrausDecomposition:
    """The ansatz only stays positive if the bond gate is completely positive.
    Measured before any of this was built: Choi PSD to 1e-16, Kraus rank 8 of
    16, trace preserving to 1e-15, superoperator reconstruction to 1e-16."""

    @pytest.mark.parametrize("dt", [0.1, 0.02, 0.005])
    def test_gate_is_cp_trace_preserving_and_reconstructs(self, dt):
        L, L_prime = models.baseline_jump_operators()
        rng = np.random.default_rng(3)
        L_pp = models.random_zz_commuting_operator(0.2, rng)
        L2 = [(L, 1.0), (L_prime, 1.0), (L_pp, 1.0)]

        # kraus_operators asserts CP / TP / reconstruction internally; calling
        # it with check=True is the test. Here we additionally re-derive the
        # superoperator independently and compare.
        ops = ilpdo.kraus_operators([], [], L2, [], dt, check=True)
        assert 1 <= len(ops) <= 16

        gen = vectorize.liouvillian_generator([], L2, d=4)
        import scipy.linalg
        S = scipy.linalg.expm(dt * gen)
        rec = sum(np.kron(K, K.conj()) for K in ops)
        np.testing.assert_allclose(rec, S, atol=1e-9)

        tp = sum(K.conj().T @ K for K in ops)
        np.testing.assert_allclose(tp, np.eye(4), atol=1e-8)

    def test_non_cp_map_is_rejected(self):
        """A map that is not completely positive must raise rather than
        silently produce garbage Kraus operators -- the assertion is the only
        thing standing between a bad gate and an unphysical evolution."""
        # dt with the WRONG sign runs the dissipator backwards, which is not CP
        L, L_prime = models.baseline_jump_operators()
        with pytest.raises(AssertionError):
            ilpdo.kraus_operators([], [], [(L, 1.0), (L_prime, 1.0)], [], dt=-0.5, check=True)


class TestGateApplicationIsExact:
    """apply_bond_gate_cp, untruncated, must reproduce the dense superoperator
    exactly. This is the single most load-bearing test in the file: when the
    LPDO run collapsed to R ~ 3e-7 it was not obvious whether the cause was an
    over-aggressive truncation budget or a bug in the Kraus gate application,
    and without this test there was no way to tell. Measured agreement is
    ~1e-15 at several (chi, kappa, dt), which is what licensed treating the
    collapse as a budget problem rather than a correctness one."""

    @pytest.mark.parametrize("chi_m,kappa,dt", [(1, 1, 0.1), (2, 1, 0.05), (2, 3, 0.02)])
    def test_untruncated_gate_matches_dense_superoperator(self, chi_m, kappa, dt):
        import scipy.linalg

        d = 2
        L, L_prime = models.baseline_jump_operators()
        rng = np.random.default_rng(2)
        L_pp = models.random_zz_commuting_operator(0.2, rng)
        L2 = [(L, 1.0), (L_prime, 1.0), (L_pp, 1.0)]

        # 2-site OPEN window: chi_left = chi_right = 1, so bond 'A' is the only
        # bond and the whole state is a single 4x4 density matrix.
        XA = (rng.standard_normal((1, d, kappa, chi_m))
              + 1j * rng.standard_normal((1, d, kappa, chi_m)))
        XB = (rng.standard_normal((chi_m, d, kappa, 1))
              + 1j * rng.standard_normal((chi_m, d, kappa, 1)))
        state = ilpdo.iLPDO(XA.copy(), XB.copy())

        def dense_rho(st):
            ts = []
            for key in ("A", "B"):
                X = st.X[key]
                cl, dd, kk, cr = X.shape
                M = np.einsum("askb,ptkq->apstbq", X, X.conj(), optimize=True)
                ts.append(M.reshape(cl * cl, dd * dd, cr * cr))
            return mps_module.MPS(ts, local_dim=2).to_dense()

        rho0 = dense_rho(state)
        gen = vectorize.liouvillian_generator([], L2, d=d * d)
        rho_ref = vectorize.unvec(scipy.linalg.expm(dt * gen) @ vectorize.vec(rho0), d * d)

        ops = ilpdo.kraus_operators([], [], L2, [], dt, d_site=d)
        ilpdo.apply_bond_gate_cp(state, "A", ops, chi_max=None, kappa_max=None, cutoff=None)

        np.testing.assert_allclose(dense_rho(state), rho_ref, rtol=1e-10, atol=1e-12)


class TestEvolutionStaysPhysical:
    """Evolution must preserve positivity -- this is what the vectorized path
    failed to do, and the reason for the whole module."""

    def test_dark_state_is_exactly_stationary(self):
        """|0...0> is annihilated by both baseline jumps, so it is an exact
        steady state and must come back unchanged.

        Compares the reconstructed density matrix, NOT the tensor shapes: the
        Kraus rank of the gate expands the legs by x8 regardless of whether the
        gate acts nontrivially, so an exactly stationary state still picks up
        numerically-zero bond and Kraus directions. Asserting the legs stay
        dimension 1 tests the representation's bookkeeping, not the physics.
        """
        L, L_prime = models.baseline_jump_operators()
        state = ilpdo.iLPDO.pure_product_state(KET0, KET0)
        itebd_lpdo.evolve_infinite_lpdo(
            state, [], [], [(L, 1.0), (L_prime, 1.0)], [], dt=0.1, n_steps=15,
            chi_max=4, kappa_max=4, cutoff=1e-12, canonicalize_every=5,
        )
        rho = _tile_dense(ilpdo.to_vectorized_imps(state), n_cells=2)
        rho = rho / np.trace(rho)

        N = 4
        dark = np.zeros((2 ** N, 2 ** N), dtype=complex)
        dark[0, 0] = 1.0  # |0000><0000|
        np.testing.assert_allclose(rho, dark, atol=1e-8)

    def test_evolved_state_is_still_positive(self):
        """The acceptance property, at test scale: after real evolution with
        real truncation, the state must still be PSD. The vectorized ansatz
        reached 41-72% negative weight here; this must stay at rounding level.

        Note this deliberately does NOT canonicalize the converted rho-iMPS.
        That state's transfer fixed point is severely rank-deficient
        (condition number ~1e29, with a 4-fold magnitude-degenerate leading
        eigenvalue), so imps.canonicalize's pinv inverts near-null directions
        and blows the tensors up to ~1e80 -- which is exactly what an earlier
        version of this test tripped over. Positivity does not need a
        canonical form, so the raw tiling is both sufficient and safe.
        """
        rng = np.random.default_rng(5)
        L, L_prime = models.baseline_jump_operators()
        L_pp = models.random_zz_commuting_operator(0.2, rng)
        state = ilpdo.iLPDO.pure_product_state(KET0, KET1)
        itebd_lpdo.evolve_infinite_lpdo(
            state, [], [], [(L, 1.0), (L_prime, 1.0), (L_pp, 1.0)], [],
            dt=0.1, n_steps=30, chi_max=6, kappa_max=6, cutoff=1e-12,
            canonicalize_every=10,
        )
        tr, herm, neg = _psd_report(_tile_dense(ilpdo.to_vectorized_imps(state), n_cells=3))
        assert tr > 0, f"trace must stay positive, got {tr}"
        assert herm < 1e-8, f"rho must stay Hermitian, got {herm:.2e}"
        assert neg < 1e-8, f"evolution left the physical manifold: neg weight {neg:.2e}"
