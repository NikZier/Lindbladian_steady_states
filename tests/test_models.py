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


class TestClassicalDriftAnnihilation:
    """The biased-hopping / pair-annihilation circuit in Lindblad form."""

    P_VALUES = [0.5, 0.8, 1.0]

    @pytest.mark.parametrize("p", P_VALUES)
    def test_commute_with_zz(self, p):
        for L in models.classical_drift_annihilation_jump_operators(p):
            assert models.commutes_with_zz(L)

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_embedded_commutes_with_global_parity(self, N):
        from lindblad_mps import vectorize

        P = global_parity(N)
        for L in models.classical_drift_annihilation_jump_operators(0.8):
            O = vectorize.embed_bond_operator(L, 0, N)
            assert np.allclose(O @ P, P @ O)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_rates_match_the_classical_probabilities(self, p):
        L_R, L_L, L_A = models.classical_drift_annihilation_jump_operators(p)
        # |01> is index 1, |10> is index 2, |11> is index 3, |00> is index 0.
        assert abs(L_R[1, 2]) ** 2 == pytest.approx(p)
        assert abs(L_L[2, 1]) ** 2 == pytest.approx(1.0 - p)
        assert abs(L_A[0, 3]) ** 2 == pytest.approx(1.0)
        for L in (L_R, L_L, L_A):
            assert np.count_nonzero(L) == (0 if np.allclose(L, 0) else 1)

    def test_vacuum_is_dark(self):
        # |00> must be annihilated by every jump, so |0...0><0...0| is a dark
        # state of the chain and the 'zero' start begins at exactly R = 0.
        ket00 = np.zeros(4, dtype=complex)
        ket00[0] = 1.0
        for L in models.classical_drift_annihilation_jump_operators(0.8):
            assert np.allclose(L @ ket00, 0)

    @pytest.mark.parametrize("p", P_VALUES)
    def test_reproduces_the_classical_master_equation(self, p):
        """On diagonal rho the dissipator must be the classical rate equation.

        Populations of the four bond configurations evolve as
            d n_00/dt = +n_11,   d n_11/dt = -n_11,
            d n_01/dt = +p n_10 - (1-p) n_01,
            d n_10/dt = +(1-p) n_01 - p n_10,
        and no coherence may be generated from a diagonal state.
        """
        from lindblad_mps import vectorize

        ops = models.classical_drift_annihilation_jump_operators(p)
        gen = vectorize.liouvillian_generator(
            [], [(L, 1.0) for L in ops], d=4
        )
        rng = np.random.default_rng(11)
        n = rng.random(4)
        rho = np.diag(n).astype(complex)
        drho = vectorize.unvec(gen @ vectorize.vec(rho), 4)

        expected = np.zeros(4)
        expected[0] = n[3]
        expected[3] = -n[3]
        expected[1] = p * n[2] - (1.0 - p) * n[1]
        expected[2] = (1.0 - p) * n[1] - p * n[2]
        assert np.allclose(np.diag(drho).real, expected)
        assert np.allclose(drho - np.diag(np.diag(drho)), 0)

    @pytest.mark.parametrize("p, sign", [(0.8, +1), (0.5, 0), (0.2, -1)])
    def test_bias_direction(self, p, sign):
        """The NET current of a lone particle must run right for p > 1/2.

        Not the gross rate: out of |10> the rate into |01> is p >= 0 whatever p
        is, so only the balance against the reverse hop carries the bias. On the
        unbiased mixture (|01><01| + |10><10|)/2 the two cancel exactly at
        p = 1/2 and d n_01/dt = (2p - 1)/2 otherwise.
        """
        from lindblad_mps import vectorize

        ops = models.classical_drift_annihilation_jump_operators(p)
        gen = vectorize.liouvillian_generator([], [(L, 1.0) for L in ops], d=4)
        rho = np.zeros((4, 4), dtype=complex)
        rho[1, 1] = rho[2, 2] = 0.5
        drho = vectorize.unvec(gen @ vectorize.vec(rho), 4)
        assert drho[1, 1].real == pytest.approx((2.0 * p - 1.0) / 2.0)
        assert np.sign(np.round(drho[1, 1].real, 12)) == sign


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
