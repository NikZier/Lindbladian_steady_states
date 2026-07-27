"""Tests for lindblad_mps.models: parity-symmetric baseline jumps and the
random parity-commuting perturbation ensemble.
"""

import numpy as np
import pytest

from lindblad_mps import models


def global_parity(N: int) -> np.ndarray:
    """Dense global parity operator P = Z_1 ... Z_N as a (2^N, 2^N) matrix."""
    P = np.array([[1]], dtype=complex)
    for _ in range(N):
        P = np.kron(P, models.Z)
    return P


class TestBaselineJumps:
    def test_commute_with_zz(self):
        for L in models.baseline_jump_operators():
            assert models.commutes_with_zz(L)

    def test_L_is_hermitian_L_prime_is_not(self):
        L, L_prime = models.baseline_jump_operators()
        assert np.allclose(L, L.conj().T)
        assert not np.allclose(L_prime, L_prime.conj().T)

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_embedded_commutes_with_global_parity(self, N):
        # Embed each baseline jump on bond (0, 1) and check [O, P] = 0 globally.
        from lindblad_mps import vectorize

        P = global_parity(N)
        for L in models.baseline_jump_operators():
            O = vectorize.embed_bond_operator(L, 0, N)
            assert np.allclose(O @ P, P @ O)


class TestRandomOperator:
    def test_commutes_and_has_target_norm(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            M = models.random_zz_commuting_operator(0.1, rng)
            assert models.commutes_with_zz(M)
            assert models.operator_norm(M) == pytest.approx(0.1)

    def test_generically_non_hermitian(self):
        rng = np.random.default_rng(1)
        M = models.random_zz_commuting_operator(0.1, rng)
        assert not np.allclose(M, M.conj().T)

    def test_projector_zeroes_cross_parity_entries(self):
        rng = np.random.default_rng(2)
        M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        P = models.project_to_zz_commutant(M)
        # Even parity indices {0, 3}, odd {1, 2}; cross entries must vanish.
        for i in range(4):
            for j in range(4):
                same = ({i, j} <= {0, 3}) or ({i, j} <= {1, 2})
                if not same:
                    assert P[i, j] == 0
        assert models.commutes_with_zz(P)

    def test_reproducible_from_seed(self):
        a = models.random_zz_commuting_operator(0.1, np.random.default_rng(7))
        b = models.random_zz_commuting_operator(0.1, np.random.default_rng(7))
        assert np.allclose(a, b)


class TestPauliDecomposition:
    def test_roundtrip_random(self):
        rng = np.random.default_rng(3)
        M = models.random_zz_commuting_operator(0.1, rng)
        coeffs = models.pauli_string_decomposition(M)
        assert len(coeffs) == 8
        assert np.allclose(models.operator_from_pauli_coeffs(coeffs), M)

    def test_roundtrip_baseline(self):
        for L in models.baseline_jump_operators():
            coeffs = models.pauli_string_decomposition(L)
            assert np.allclose(models.operator_from_pauli_coeffs(coeffs), L)

    def test_forbidden_string_rejected(self):
        # X (x) I does not commute with Z (x) Z; its coefficient must trip the assert.
        M = models.kron(models.X, models.I2)
        with pytest.raises(AssertionError):
            models.pauli_string_decomposition(M)
