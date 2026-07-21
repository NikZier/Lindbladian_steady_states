"""Validation tests for diagnostics.py and mps.MPS.trace()."""

import numpy as np
import pytest

from lindblad_mps import diagnostics, exact, mps, observables, vectorize

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^- (lowering)


def heisenberg_bond(J: float = 1.0) -> np.ndarray:
    """Build the 4x4 nearest-neighbour Heisenberg coupling J*(XX+YY+ZZ)."""
    return J * (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ))


class TestMPSTrace:
    """MPS.trace() must match np.trace(rho) for an untruncated MPS."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_matches_dense_trace(self, N):
        rng = np.random.default_rng(0)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        state = mps.MPS.from_dense(rho, N)
        assert state.trace() == pytest.approx(np.trace(rho), abs=1e-8)

    def test_matches_exact_steady_state_trace_one(self):
        """exact.steady_state() normalizes Tr[rho]=1; MPS.trace() on its
        (untruncated) MPS form must reproduce that."""
        N = 3
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.4)]
        rho = exact.steady_state(H2_terms, [], [], L1_terms, N)

        state = mps.MPS.from_dense(rho, N)
        assert state.trace() == pytest.approx(1.0, abs=1e-8)


class TestSchmidtSpectrum:
    """schmidt_spectrum/entanglement_entropies must be consistent with a
    dense reference and behave correctly on trivial (product) states."""

    def test_product_state_has_zero_entanglement(self):
        N = 4
        state = mps.MPS.maximally_mixed(N)
        entropies = diagnostics.entanglement_entropies(state)
        assert len(entropies) == N - 1
        assert all(e == pytest.approx(0.0, abs=1e-10) for e in entropies)

    def test_spectrum_normalized_and_matches_dense_svd(self):
        """Each bond's Schmidt spectrum must be unit-normalized (sum S^2 = 1)
        and match the singular values of the corresponding dense bipartition cut."""
        N = 4
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.4)]
        rho = exact.steady_state(H2_terms, [], [], L1_terms, N)

        state = mps.MPS.from_dense(rho, N)
        state.normalize()
        spectra = diagnostics.schmidt_spectrum(state)

        v = state.to_local_vec()
        d = state.phys_dim
        for bond, spectrum in enumerate(spectra):
            left_dim = d ** (bond + 1)
            mat = v.reshape(left_dim, d ** (N - bond - 1))
            expected = np.linalg.svd(mat, compute_uv=False)
            expected = expected / np.linalg.norm(expected)

            assert np.sum(spectrum**2) == pytest.approx(1.0, abs=1e-8)
            k = min(len(spectrum), len(expected))
            assert np.allclose(sorted(spectrum, reverse=True)[:k], expected[:k], atol=1e-6)

    def test_schmidt_spectrum_does_not_mutate_state(self):
        N = 3
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.4)]
        rho = exact.steady_state(H2_terms, [], [], L1_terms, N)
        state = mps.MPS.from_dense(rho, N)
        before = state.to_dense()

        diagnostics.schmidt_spectrum(state)

        assert np.allclose(state.to_dense(), before, atol=1e-10)


class TestChiMaxBinding:
    def test_low_entanglement_state_not_binding_at_large_chi(self):
        state = mps.MPS.maximally_mixed(4)
        assert diagnostics.is_chi_max_binding(state, chi_max=16) is False

    def test_binding_when_bond_dim_equals_cap(self):
        N = 4
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.4)]
        rho = exact.steady_state(H2_terms, [], [], L1_terms, N)
        # force a tight cap so at least one bond saturates it
        state = mps.MPS.from_dense(rho, N, chi_max=2)
        assert diagnostics.is_chi_max_binding(state, chi_max=2) is True


class TestChiConvergenceScan:
    def test_scan_shape_and_convergence_trend(self):
        """observable_diff should shrink as chi grows for a well-resolved
        model, and by the largest chi tested chi_max should no longer be binding."""
        N = 4
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.5), (SZ, 0.1)]
        chi_list = [1, 2, 4, 8]

        def observable_fn(state):
            return observables.renyi2_correlator_mps(state, SZ, 0, 3)

        results = diagnostics.chi_convergence_scan(
            H2_terms,
            [],
            [],
            L1_terms,
            N,
            dt_schedule=[0.5, 0.1, 0.02],
            steps_per_dt=100,
            chi_list=chi_list,
            observable_fn=observable_fn,
            # a cutoff is what makes chi_max_binding a meaningful accuracy
            # signal: without one, canonicalize() keeps every singular value
            # up to chi_max regardless of size, so bond dims saturate at
            # chi_max even for a weakly-entangled state. Too tight a cutoff
            # (e.g. 1e-10) instead keeps SVD numerical noise around machine
            # precision; 1e-8 clears that floor for this model.
            cutoff=1e-8,
            recanonicalize_every=5,
        )

        assert results["chi"] == chi_list
        assert len(results["observable"]) == len(chi_list)
        assert results["observable_diff"][0] is None
        assert all(v is not None for v in results["observable_diff"][1:])

        # this model is weakly entangled, so the observable should already be
        # converged (to floating-point/truncation noise) by chi=8
        assert results["observable_diff"][-1] < 1e-6

        # chi=8 comfortably exceeds what this small, weakly-entangled model needs
        assert results["chi_max_binding"][-1] is False

    def test_scan_result_matches_exact_at_largest_chi(self):
        N = 4
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.5), (SZ, 0.1)]

        rho_exact = exact.steady_state(H2_terms, [], [], L1_terms, N)
        r_exact = observables.renyi2_correlator_dense(rho_exact, SZ, 1, 2, N)

        def observable_fn(state):
            return observables.renyi2_correlator_mps(state, SZ, 1, 2)

        results = diagnostics.chi_convergence_scan(
            H2_terms,
            [],
            [],
            L1_terms,
            N,
            dt_schedule=[0.5, 0.1, 0.02],
            steps_per_dt=150,
            chi_list=[8],
            observable_fn=observable_fn,
            recanonicalize_every=5,
        )

        assert results["observable"][0] == pytest.approx(r_exact, abs=1e-3)
