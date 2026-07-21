"""Validation tests for mps.py and tebd.py.

Cross-checks the MPS tensor-network machinery against the dense/local-vec
reference implementations already validated in test_vectorize_exact_observables.py,
and checks that a full finite-chain TEBD run converges to the exact (dense
ED) steady state.
"""

import numpy as np
import pytest

from lindblad_mps import exact, mps, observables, tebd, vectorize

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^- (lowering)


def heisenberg_bond(J: float = 1.0) -> np.ndarray:
    """Build the 4x4 nearest-neighbour Heisenberg coupling J*(XX+YY+ZZ)."""
    return J * (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ))


class TestMPSRoundTrip:
    """from_dense / to_dense / to_local_vec must be exact inverses (no truncation)."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_from_dense_to_dense(self, N):
        """Converting a random matrix to an untruncated MPS and back must
        reproduce it exactly."""
        rng = np.random.default_rng(0)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        state = mps.MPS.from_dense(rho, N)
        rho_back = state.to_dense()

        assert np.allclose(rho, rho_back, atol=1e-10)

    def test_to_local_vec_matches_vectorize(self):
        """MPS.to_local_vec() must agree with vectorize.physical_to_local_vec
        for the same dense matrix."""
        N = 3
        rng = np.random.default_rng(1)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        state = mps.MPS.from_dense(rho, N)
        expected = vectorize.physical_to_local_vec(rho, N)

        assert np.allclose(state.to_local_vec(), expected, atol=1e-10)


class TestMPSInnerProducts:
    """norm2/overlap must match the dense Frobenius inner product."""

    def test_norm2_matches_dense(self):
        N = 3
        rng = np.random.default_rng(2)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        state = mps.MPS.from_dense(rho, N)
        expected = np.trace(rho.conj().T @ rho).real

        assert state.norm2() == pytest.approx(expected, abs=1e-8)

    def test_overlap_matches_dense(self):
        N = 3
        rng = np.random.default_rng(3)
        dim = 2**N
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        B = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        mps_a = mps.MPS.from_dense(A, N)
        mps_b = mps.MPS.from_dense(B, N)
        expected = np.trace(A.conj().T @ B)

        assert mps_a.overlap(mps_b) == pytest.approx(expected, abs=1e-8)


class TestApplyTwoSiteGate:
    """MPS.apply_two_site_gate must match applying the same gate to the dense
    local-vec vector directly."""

    @pytest.mark.parametrize("N,bond", [(2, 0), (3, 0), (3, 1), (4, 1)])
    def test_gate_application_matches_dense(self, N, bond):
        rng = np.random.default_rng(4)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        H2_terms = [(heisenberg_bond(), 0.7)]
        L2_terms = [(np.kron(SM, I2), 0.3)]
        gate = vectorize.bond_gate(H2_terms, L2_terms, dt=0.1, weight_left=1.0, weight_right=1.0)

        state = mps.MPS.from_dense(rho, N)
        state.apply_two_site_gate(bond, gate)

        v = vectorize.physical_to_local_vec(rho, N)
        full_gate = vectorize.embed_bond_operator(gate, bond, N, local_dim=state.phys_dim)
        v_expected = full_gate @ v

        assert np.allclose(state.to_local_vec(), v_expected, atol=1e-8)


class TestCanonicalize:
    """canonicalize() without truncation must not change the represented state."""

    def test_canonicalize_preserves_state(self):
        N = 4
        rng = np.random.default_rng(5)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        state = mps.MPS.from_dense(rho, N)
        before = state.to_dense()
        state.canonicalize()
        after = state.to_dense()

        assert np.allclose(before, after, atol=1e-8)

    def test_canonicalize_produces_right_orthonormal_tensors(self):
        """canonicalize() does a lossless left-to-right QR sweep followed by a
        right-to-left SVD sweep that absorbs U*S into the left neighbour;
        the net gauge this leaves behind is right-orthonormal tensors for
        sites 1..N-1 (sum_{i,r} conj(A)[l,i,r] A[l',i,r] = delta_{l,l'}),
        with all remaining weight/normalization carried by tensor 0."""
        N = 4
        rng = np.random.default_rng(6)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        state = mps.MPS.from_dense(rho, N)
        state.canonicalize()

        for n in range(1, N):
            A = state.tensors[n]
            gram = np.einsum("lir,mir->lm", A.conj(), A)
            assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-8)


class TestRenyi2ViaMPS:
    """renyi2_correlator_mps must agree with the dense reference."""

    @pytest.mark.parametrize("i,j", [(0, 1), (0, 2), (1, 1)])
    def test_matches_dense(self, i, j):
        N = 3
        H2_terms = [(heisenberg_bond(), 1.0)]
        H1_terms = [(SX, 0.5)]
        L1_terms = [(SM, 0.3)]
        rho = exact.steady_state(H2_terms, H1_terms, [], L1_terms, N)

        state = mps.MPS.from_dense(rho, N)
        r_mps = observables.renyi2_correlator_mps(state, SZ, i, j)
        r_dense = observables.renyi2_correlator_dense(rho, SZ, i, j, N)

        assert r_mps == pytest.approx(r_dense, abs=1e-8)


class TestTEBDConvergesToExactSteadyState:
    """A full finite-chain TEBD run must converge to the exact ED steady state."""

    def test_convergence(self):
        N = 4
        H2_terms = [(heisenberg_bond(), 1.0)]
        H1_terms = []
        L2_terms = []
        L1_terms = [(SM, 0.5), (SZ, 0.1)]

        rho_exact = exact.steady_state(H2_terms, H1_terms, L2_terms, L1_terms, N)
        rho_exact = rho_exact / np.trace(rho_exact)

        state, history = tebd.find_steady_state(
            H2_terms,
            H1_terms,
            L2_terms,
            L1_terms,
            N,
            dt_schedule=[0.5, 0.1, 0.02],
            steps_per_dt=150,
            chi_max=16,
            recanonicalize_every=5,
        )

        # overlap between successive normalized states should approach 1
        assert history["overlap"][-1] == pytest.approx(1.0, abs=1e-4)

        rho_tebd = state.to_dense()
        rho_tebd = rho_tebd / np.trace(rho_tebd)

        assert np.allclose(rho_tebd, rho_exact, atol=1e-3)

    def test_renyi2_correlator_matches_exact(self):
        """The Renyi-2 correlator computed on the converged TEBD state must
        match the one computed from the exact steady state."""
        N = 4
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.5), (SZ, 0.1)]

        rho_exact = exact.steady_state(H2_terms, [], [], L1_terms, N)

        state, _ = tebd.find_steady_state(
            H2_terms,
            [],
            [],
            L1_terms,
            N,
            dt_schedule=[0.5, 0.1, 0.02],
            steps_per_dt=150,
            chi_max=16,
            recanonicalize_every=5,
        )

        for i, j in [(0, 1), (0, 3), (1, 2)]:
            r_tebd = observables.renyi2_correlator_mps(state, SZ, i, j)
            r_exact = observables.renyi2_correlator_dense(rho_exact, SZ, i, j, N)
            assert r_tebd == pytest.approx(r_exact, abs=1e-3)
