"""Validation tests for vectorize.py, exact.py, and observables.py.

These cross-check the local-vec/bond-gate machinery (which the future TEBD
code will rely on) against an independent dense exact-diagonalization
reference, plus check basic physical sanity properties (trace preservation,
maximally-mixed steady state for pure dephasing).
"""

import numpy as np
import pytest

from lindblad_mps import exact, observables, vectorize

# Pauli matrices, reused across tests.
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^- (lowering: |1> -> |0>, annihilates |0>)


def heisenberg_bond(J: float = 1.0) -> np.ndarray:
    """Build the 4x4 nearest-neighbour Heisenberg coupling J*(XX+YY+ZZ).

    Input: J, coupling constant (folded into the returned operator, not a
        separate coefficient) -- used directly as a bond Hamiltonian term
        with coefficient 1.0 in tests.
    Output: (4, 4) ndarray.
    """
    return J * (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ))


def bond_weights(bond_index: int, n_bonds: int) -> tuple[float, float]:
    """Return (weight_left, weight_right) for single-site terms on a given bond.

    Interior sites split their single-site term 0.5/0.5 across their two
    adjacent bonds; chain-end sites get full weight 1.0 on their only bond.

    Input: bond_index, index of the bond (0 to n_bonds-1); n_bonds, total
        number of bonds (N-1 for an N-site chain).
    Output: (weight_left, weight_right) tuple of floats.
    """
    weight_left = 1.0 if bond_index == 0 else 0.5
    weight_right = 1.0 if bond_index == n_bonds - 1 else 0.5
    return weight_left, weight_right


def build_all_bond_generators(
    H2_terms, H1_terms, L2_terms, L1_terms, N, d_site=2
) -> list[np.ndarray]:
    """Build the list of N-1 vectorized bond generators for a chain, with
    single-site terms correctly weighted at interior vs. boundary bonds.

    Input: H2_terms, H1_terms, L2_terms, L1_terms -- interaction terms in the
        (operator, coefficient) format used throughout the package; N, number
        of sites; d_site, physical dimension of one site.
    Output: list of N-1 (d_site^4, d_site^4) ndarrays.
    """
    n_bonds = N - 1
    generators = []
    for b in range(n_bonds):
        wl, wr = bond_weights(b, n_bonds)
        generators.append(
            vectorize.build_bond_generator(
                H2_terms,
                L2_terms,
                H1_left_terms=H1_terms,
                L1_left_terms=L1_terms,
                H1_right_terms=H1_terms,
                L1_right_terms=L1_terms,
                weight_left=wl,
                weight_right=wr,
                d_site=d_site,
            )
        )
    return generators


class TestLiouvillianGenerator:
    """Basic algebraic properties of vectorize.liouvillian_generator."""

    def test_trace_preserving(self):
        """A valid Lindblad generator must satisfy Tr[L(rho)] = 0 for all rho,
        i.e. vec(I)^T @ generator = 0 (the generator has I in its left null space)."""
        H_terms = [(heisenberg_bond(), 1.3), (np.kron(SX, I2), 0.7)]
        L_terms = [(np.kron(SM, I2), 0.4), (np.kron(I2, SZ), 0.2)]
        generator = vectorize.liouvillian_generator(H_terms, L_terms, d=4)

        vec_I = vectorize.vec(np.eye(4, dtype=complex))
        residual = vec_I @ generator
        assert np.allclose(residual, 0, atol=1e-10)

    def test_matches_manual_expm_evolution(self):
        """exp(dt*generator) applied to vec(rho) must equal the direct definition
        d(vec rho)/dt = generator @ vec(rho) integrated by scipy, checked via a
        first-order finite-difference consistency check at small dt. Both sides
        must use the same (local-vec) index convention that build_bond_generator
        and physical_to_local_vec produce."""
        H_terms = [(heisenberg_bond(), 1.0)]
        L_terms = [(np.kron(SM, I2), 0.5)]
        generator = vectorize.build_bond_generator(
            H_terms, L_terms, weight_left=1.0, weight_right=1.0
        )

        rho0 = np.eye(4, dtype=complex) / 4
        v0 = vectorize.physical_to_local_vec(rho0, N=2)
        dt = 1e-6
        gate = vectorize.bond_gate(H_terms, L_terms, dt, weight_left=1.0, weight_right=1.0)
        v_evolved = gate @ v0
        v_expected = v0 + dt * (generator @ v0)
        assert np.allclose(v_evolved, v_expected, atol=1e-9)


class TestBondGateVsExact:
    """Cross-check vectorize.py's bond-based generator against exact.py's
    independently-built dense physical generator, on small chains."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_assembled_generator_annihilates_exact_steady_state(self, N):
        """The steady state found by exact.py (built via direct N-site
        Kronecker embedding) must also be annihilated by vectorize.py's
        generator assembled bond-by-bond with correctly weighted single-site
        terms -- i.e. the two independent constructions describe the same
        physical generator."""
        H2_terms = [(heisenberg_bond(), 1.0)]
        H1_terms = [(SX, 0.5)]
        L2_terms = []
        L1_terms = [(SM, 0.3), (SZ, 0.1)]

        rho = exact.steady_state(H2_terms, H1_terms, L2_terms, L1_terms, N)
        rho_lv = vectorize.physical_to_local_vec(rho, N)

        bond_generators = build_all_bond_generators(
            H2_terms, H1_terms, L2_terms, L1_terms, N
        )
        full_generator = vectorize.assemble_chain_generator(bond_generators, N, local_dim=2)

        residual = full_generator @ rho_lv
        assert np.allclose(residual, 0, atol=1e-8)

    def test_pure_decay_gives_all_down_product_state(self):
        """With only single-site spontaneous decay (jump operator sigma^-, no
        Hamiltonian, no coupling between sites), each site independently and
        irreversibly relaxes to its dark state |0><0| (sigma^- annihilates
        |0>), so the unique N-site steady state is the pure product state
        |0...0><0...0|. Note: dephasing alone (jump op sigma^z) would NOT be a
        good test here, since sigma^z leaves every diagonal (in the Z basis)
        density matrix invariant -- that steady state is highly degenerate,
        not uniquely the maximally mixed state."""
        N = 3
        L1_terms = [(SM, 1.0)]
        rho = exact.steady_state([], [], [], L1_terms, N)

        ground = np.zeros(2**N, dtype=complex)
        ground[0] = 1.0  # |0...0> in the computational basis (SM|0>=0 is the dark state)
        expected = np.outer(ground, ground.conj())
        assert np.allclose(rho, expected, atol=1e-6)


class TestRenyi2Correlator:
    """Cross-check the dense reference and local-vec implementations of the
    Renyi-2 correlator against each other."""

    @pytest.mark.parametrize("N", [3, 4])
    @pytest.mark.parametrize("i,j", [(0, 1), (0, 2), (1, 1), (0, 0)])
    def test_dense_matches_localvec(self, N, i, j):
        """renyi2_correlator_dense and renyi2_correlator_localvec must agree
        exactly (up to floating point) since they compute the same quantity
        in two different index orderings."""
        if max(i, j) >= N:
            pytest.skip("site index out of range for this N")

        H2_terms = [(heisenberg_bond(), 1.0)]
        H1_terms = [(SX, 0.5)]
        L1_terms = [(SM, 0.3)]
        rho = exact.steady_state(H2_terms, H1_terms, [], L1_terms, N)
        rho_lv = vectorize.physical_to_local_vec(rho, N)

        r_dense = observables.renyi2_correlator_dense(rho, SZ, i, j, N)
        r_localvec = observables.renyi2_correlator_localvec(rho_lv, SZ, i, j, N)

        assert r_dense == pytest.approx(r_localvec, abs=1e-8)

    def test_diagonal_case_is_purity_like_and_nonnegative(self):
        """For i == j, R(i,i) reduces to a purity-like quantity built from
        O_i O_i^dagger and should come out real and non-negative for a
        physical (Hermitian, positive-ish) steady state."""
        N = 3
        H2_terms = [(heisenberg_bond(), 1.0)]
        L1_terms = [(SM, 0.4)]
        rho = exact.steady_state(H2_terms, [], [], L1_terms, N)

        r = observables.renyi2_correlator_dense(rho, SZ, 1, 1, N)
        assert r >= -1e-10


class TestLocalVecRoundTrip:
    """physical_to_local_vec / local_vec_to_physical must be exact inverses."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_round_trip(self, N):
        """Converting a random dense density-matrix-shaped matrix to local-vec
        and back must reproduce the original matrix exactly."""
        rng = np.random.default_rng(0)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        rho_lv = vectorize.physical_to_local_vec(rho, N)
        rho_back = vectorize.local_vec_to_physical(rho_lv, N)

        assert np.allclose(rho, rho_back)

    def test_inner_product_invariant_under_reordering(self):
        """The Frobenius inner product Tr[A^dagger B] must be unchanged by the
        local-vec reordering, since it is just a permutation of the same
        entries -- this is what makes Tr[rho^dagger rho] usable as a
        denominator in either representation."""
        N = 3
        rng = np.random.default_rng(1)
        dim = 2**N
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        B = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        global_ip = np.trace(A.conj().T @ B)
        local_ip = np.vdot(
            vectorize.physical_to_local_vec(A, N), vectorize.physical_to_local_vec(B, N)
        )
        assert np.isclose(global_ip, local_ip)
