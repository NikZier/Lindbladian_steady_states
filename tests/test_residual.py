"""Validation tests for residual.py.

The Liouvillian MPO and the four-layer contraction are checked against the
dense generator from vectorize/exact -- the same reference
renyi2_swssb.validate_against_exact uses -- and against the two states whose
residual is known analytically: an exact dark state (zero) and a state the
Liouvillian demonstrably moves (nonzero).
"""

import numpy as np
import pytest

from lindblad_mps import exact, models, mps, residual, tebd, vectorize

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^- (lowering)


def heisenberg_bond(J: float = 1.0) -> np.ndarray:
    """Build the 4x4 nearest-neighbour Heisenberg coupling J*(XX+YY+ZZ)."""
    SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return J * (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ))


def dense_residual(
    state: mps.MPS,
    H2_terms: list,
    H1_terms: list,
    L2_terms: list,
    L1_terms: list,
    N: int,
) -> float:
    """Reference ||L vec(rho)|| / ||vec(rho)|| built from the dense generator.

    Mirrors renyi2_swssb.validate_against_exact. The dense generator uses the
    global vec convention rather than local-vec, but the two differ by a
    permutation of basis indices, which leaves both norms invariant.

    Input: state, the MPS; the model terms; N, number of sites.
    Output: float residual.
    """
    rho = state.to_dense()
    H = exact.build_hamiltonian(H2_terms, H1_terms, N)
    jump_ops = exact.build_jump_operators(L2_terms, L1_terms, N)
    generator = vectorize.liouvillian_generator([(H, 1.0)], jump_ops, d=2**N)
    v = vectorize.vec(rho)
    return float(np.linalg.norm(generator @ v) / np.linalg.norm(v))


def swssb_terms(epsilon: float = 0.2, seed: int = 7) -> list:
    """The study's bond jump terms: the two baseline jumps plus a random L''."""
    L, L_prime = models.baseline_jump_operators()
    rng = np.random.default_rng(seed)
    L_pp = models.random_zz_commuting_operator(epsilon, rng)
    return [(L, 1.0), (L_prime, 1.0), (L_pp, 1.0)]


class TestLiouvillianMPO:
    """The MPO must reproduce the dense chain generator exactly."""

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_matches_dense_generator_on_random_states(self, N):
        """||L|rho>||/|||rho>|| from the MPO must equal the dense value for a
        generic (non-steady) state, at several sizes."""
        L2_terms = swssb_terms()
        rng = np.random.default_rng(N)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        state = mps.MPS.from_dense(rho, N)

        got = residual.steady_state_residual(state, [], [], L2_terms, [])
        expected = dense_residual(state, [], [], L2_terms, [], N)

        assert got["residual"] == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_matches_dense_with_hamiltonian_and_single_site_terms(self, N):
        """The MPO must also be right when H and single-site jumps are present:
        those enter through the bond weighting, which is where an off-by-one
        between interior and boundary sites would hide."""
        H2_terms = [(heisenberg_bond(0.7), 1.0)]
        H1_terms = [(SZ, 0.3)]
        L2_terms = swssb_terms()
        L1_terms = [(SM, 0.4)]
        rng = np.random.default_rng(100 + N)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        state = mps.MPS.from_dense(rho, N)

        got = residual.steady_state_residual(state, H2_terms, H1_terms, L2_terms, L1_terms)
        expected = dense_residual(state, H2_terms, H1_terms, L2_terms, L1_terms, N)

        assert got["residual"] == pytest.approx(expected, rel=1e-9)

    def test_bond_dimension_is_schmidt_rank_plus_two(self):
        """A sum of nearest-neighbour terms of operator-Schmidt rank r needs
        MPO bond dimension exactly r + 2 -- the claim that makes L^dagger L a
        finite (r+2)^2 object rather than a non-local one."""
        L2_terms = swssb_terms()
        N = 6
        generators = residual.bond_generators([], [], L2_terms, [], N)
        A, _ = residual.split_bond_generator(generators[N // 2], phys_dim=4)
        mpo = residual.liouvillian_mpo([], [], L2_terms, [], N)

        assert len(A) == 16  # full rank for this model
        assert max(W.shape[0] for W in mpo) == len(A) + 2
        assert mpo[0].shape[0] == 1 and mpo[-1].shape[3] == 1

    def test_mpo_cutoff_is_a_controlled_approximation(self):
        """Truncating the MPO's Schmidt spectrum must shrink the bond dimension
        while perturbing the residual only at the cutoff's own scale."""
        L2_terms = swssb_terms()
        N = 4
        rng = np.random.default_rng(3)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        state = mps.MPS.from_dense(rho, N)

        exact_res = residual.steady_state_residual(state, [], [], L2_terms, [])
        cut_res = residual.steady_state_residual(
            state, [], [], L2_terms, [], mpo_cutoff=1e-3
        )

        assert cut_res["mpo_bond_dim"] < exact_res["mpo_bond_dim"]
        assert cut_res["residual"] == pytest.approx(exact_res["residual"], rel=1e-2)


class TestKnownResiduals:
    """States whose residual is known without computing anything."""

    @pytest.mark.parametrize("N", [2, 4, 6])
    def test_dark_state_residual_is_zero(self, N):
        """|0...0> is annihilated by both baseline jumps, so it is an exact
        steady state and its residual must vanish to machine precision --
        the test the diagnostic exists to pass."""
        L, L_prime = models.baseline_jump_operators()
        state = mps.MPS.pure_product_state([np.array([1, 0], dtype=complex)] * N)

        got = residual.steady_state_residual(state, [], [], [(L, 1.0), (L_prime, 1.0)], [])

        assert got["residual"] < 1e-12

    @pytest.mark.parametrize("N", [4, 6])
    def test_neel_state_residual_is_large(self, N):
        """|0101...> is driven by L on every bond, so the same diagnostic must
        report a residual of order the jump rates -- it is not returning zero
        for everything."""
        L, L_prime = models.baseline_jump_operators()
        kets = [np.array([1, 0], dtype=complex) if i % 2 == 0
                else np.array([0, 1], dtype=complex) for i in range(N)]
        state = mps.MPS.pure_product_state(kets)

        got = residual.steady_state_residual(state, [], [], [(L, 1.0), (L_prime, 1.0)], [])

        assert got["residual"] > 1.0
        assert got["residual_per_bond"] == pytest.approx(
            got["residual"] / (N - 1), rel=1e-12
        )

    def test_residual_is_invariant_under_rescaling(self):
        """The residual is normalized by the state norm, so scaling the MPS
        must not change it: TEBD renormalizes every step and the diagnostic
        must not depend on where in that cycle it is measured."""
        L2_terms = swssb_terms()
        N = 4
        rng = np.random.default_rng(11)
        dim = 2**N
        rho = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        state = mps.MPS.from_dense(rho, N)
        scaled = state.copy()
        scaled.tensors[0] = scaled.tensors[0] * 37.0

        a = residual.steady_state_residual(state, [], [], L2_terms, [])
        b = residual.steady_state_residual(scaled, [], [], L2_terms, [])

        assert b["residual"] == pytest.approx(a["residual"], rel=1e-10)


class TestResidualTracksConvergence:
    """The residual must fall as TEBD approaches the fixed point."""

    def test_residual_falls_along_a_tebd_run(self):
        """Evolving longer must lower the residual monotonically, by orders of
        magnitude -- the property that makes it usable as a stopping criterion."""
        N = 4
        L2_terms = swssb_terms()
        initial = mps.MPS.pure_product_state(
            [np.array([1, 0], dtype=complex) if i % 2 == 0
             else np.array([0, 1], dtype=complex) for i in range(N)]
        )

        residuals = []
        for steps in (0, 50, 300):
            state, _ = tebd.find_steady_state(
                H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[], N=N,
                dt_schedule=[0.05], steps_per_dt=steps, chi_max=64, cutoff=1e-12,
                initial_state=initial,
            )
            residuals.append(
                residual.steady_state_residual(state, [], [], L2_terms, [])["residual"]
            )

        assert residuals[0] > residuals[1] > residuals[2]
        # Five orders, not more: the run bottoms out on the dt^2 Trotter floor
        # of the fixed dt=0.05 schedule, which the next test pins down.
        assert residuals[2] < 1e-5 * residuals[0]

    def test_converged_residual_is_a_dt_squared_trotter_floor(self):
        """A relaxed run does not reach zero residual, it reaches C*dt^2: TEBD
        converges to the fixed point of the Trotterized propagator, not of L.

        This is what turns the residual into a convergence test -- a run sitting
        on its dt^2 floor has relaxed, one orders of magnitude above it has not,
        and halving dt separates the two cases.
        """
        N = 4
        L2_terms = swssb_terms()
        initial = mps.MPS.pure_product_state([np.array([1, 0], dtype=complex)] * N)

        floors = {}
        for dt in (0.05, 0.02):
            state, _ = tebd.find_steady_state(
                H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[], N=N,
                dt_schedule=[dt], steps_per_dt=int(30 / dt), chi_max=64,
                cutoff=1e-13, initial_state=initial,
            )
            floors[dt] = residual.steady_state_residual(
                state, [], [], L2_terms, []
            )["residual"]

        assert floors[0.05] / floors[0.02] == pytest.approx((0.05 / 0.02) ** 2, rel=0.05)

    def test_converged_residual_matches_dense_steady_state(self):
        """At N=4 the TEBD fixed point can be compared against dense ED: the
        MPS residual must be as small as the dense steady state's own."""
        N = 4
        L2_terms = swssb_terms()
        state, _ = tebd.find_steady_state(
            H2_terms=[], H1_terms=[], L2_terms=L2_terms, L1_terms=[], N=N,
            dt_schedule=[0.1, 0.05, 0.01], steps_per_dt=300, chi_max=64,
            cutoff=1e-12,
            initial_state=mps.MPS.pure_product_state(
                [np.array([1, 0], dtype=complex)] * N
            ),
        )

        got = residual.steady_state_residual(state, [], [], L2_terms, [])
        expected = dense_residual(state, [], [], L2_terms, [], N)

        # Looser than the random-state tests above by design: at a residual of
        # ~2e-06 the MPO route is computing sqrt of a near-total cancellation
        # between terms of order ||L||^2, so its last few digits are noise
        # (see the arithmetic floor in residual.py's docstring). The dense
        # reference cancels differently and need not agree past that.
        assert got["residual"] == pytest.approx(expected, rel=1e-4)
        assert got["residual"] < 1e-5
