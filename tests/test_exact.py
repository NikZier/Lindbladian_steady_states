"""Tests for lindblad_mps.exact.sector_steady_state -- the dense reference
inside one strong-symmetry sector.

This is the only absolute reference the project has at a size that genuinely
relaxes, so it is what a TEBD schedule gets calibrated against (CLAUDE.md,
Traps 2 and 5). Two things therefore need pinning: that it solves the right
problem, and that its gap numbers are not misread.
"""

import numpy as np
import pytest

from lindblad_mps import exact, models, observables, vectorize


def baseline_L2() -> list[tuple[np.ndarray, float]]:
    """The SWSSB baseline bond jumps at rate 1."""
    return [(op, 1.0) for op in models.baseline_jump_operators()]


def draw_h(epsilon: float, seed: int) -> np.ndarray:
    return models.random_zz_commuting_hermitian_operator(
        epsilon, np.random.default_rng(20260919 + 1 + seed)
    )


class TestSectorIndices:
    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_both_charges_partition_the_vec_space(self, N):
        plus = exact.sector_indices(N, +1)
        minus = exact.sector_indices(N, -1)
        # The two DIAGONAL blocks do not exhaust vec space -- the off-diagonal
        # (q_L != q_R) blocks are the rest -- but they must be disjoint.
        assert not set(plus.tolist()) & set(minus.tolist())
        assert len(plus) == (2 ** (N - 1)) ** 2


class TestBaselineIsTheDarkVacuum:
    """L'' = 0, H = 0: the (+,+) steady state is the pure dark vacuum."""

    def test_purity_one_and_zero_correlator(self):
        N = 4
        out = exact.sector_steady_state([], [], baseline_L2(), [], N)
        rho = out["rho"]
        assert np.trace(rho).real == pytest.approx(1.0)
        assert np.trace(rho @ rho).real == pytest.approx(1.0, abs=1e-10)
        R = observables.renyi2_correlator_dense(rho, models.X, 1, 3, N)
        assert abs(R) < 1e-12


class TestHamiltonianPerturbation:
    """The Hamiltonian branch -- the reason this function moved into the library."""

    def test_returns_a_genuine_steady_state(self):
        N = 4
        h = draw_h(0.2, 0)
        out = exact.sector_steady_state([(h, 1.0)], [], baseline_L2(), [], N)
        rho = out["rho"]

        assert np.allclose(rho, rho.conj().T)
        assert np.trace(rho).real == pytest.approx(1.0)
        assert np.linalg.eigvalsh(rho).min() > -1e-10

        H = exact.build_hamiltonian([(h, 1.0)], [], N)
        jumps = exact.build_jump_operators(baseline_L2(), [], N)
        gen = vectorize.liouvillian_generator([(H, 1.0)], jumps, d=2**N)
        v = vectorize.vec(rho)
        assert np.linalg.norm(gen @ v) / np.linalg.norm(v) < 1e-10

    def test_leakage_assertion_fires_for_a_symmetry_breaking_hamiltonian(self):
        # X (x) I does not commute with Z(x)Z, so the embedded term breaks the
        # strong symmetry. Nothing else in the pipeline would notice; this
        # assertion is the guard, so it must actually be live.
        bad = np.kron(models.X, np.eye(2, dtype=complex))
        assert not models.commutes_with_zz(bad)
        with pytest.raises(AssertionError, match="sector leakage"):
            exact.sector_steady_state([(bad, 1.0)], [], baseline_L2(), [], 4)

    def test_R_is_proportional_to_the_pair_creation_amplitude(self):
        # R = const * |<11|h|00>|^2 across independent draws. The constant
        # itself is a measured property of the model (see the study); what is
        # tested here is that ONE matrix element fixes R, which is what licenses
        # sizing schedules from that element.
        N = 4
        ratios = []
        for s in range(6):
            h = draw_h(0.01, s)  # small epsilon: O(epsilon^2) corrections off
            out = exact.sector_steady_state([(h, 1.0)], [], baseline_L2(), [], N)
            R = observables.renyi2_correlator_dense(out["rho"], models.X, 1, 3, N)
            ratios.append(R / abs(models.pair_creation_element(h)) ** 2)
        a = np.array(ratios)
        assert a.std() / a.mean() < 1e-3
        assert a.mean() > 0


class TestGapNumbersAreNotInterchangeable:
    """The finding that kept this study's schedules from being undersized."""

    def test_small_N_gap_rate_is_epsilon_independent(self):
        # At N=4 the slowest mode of the SWSSB baseline is the BASELINE
        # diffusive one, not the perturbation-induced mode at rate ~epsilon^2.
        # So gap_rate here says nothing about how long an infinite-system run
        # must evolve for -- a schedule sized from it would be ~1000x short.
        # Pinned so the warning in the docstring is not quietly invalidated.
        rates = [
            exact.sector_steady_state([(draw_h(eps, 0), 1.0)], [], baseline_L2(), [], 4)["gap_rate"]
            for eps in (0.2, 0.05)
        ]
        assert rates[0] == pytest.approx(rates[1], rel=1e-2)
        assert rates[0] > 1.0  # an O(baseline rate) number, not an O(epsilon^2) one

    def test_gap_rate_and_gap_singular_differ(self):
        # The generator is non-normal, so the smallest singular value is not a
        # relaxation rate. Kept distinct on purpose; this fails if someone
        # aliases them.
        out = exact.sector_steady_state([], [], baseline_L2(), [], 4)
        assert out["gap_rate"] != pytest.approx(out["gap_singular"], rel=1e-6)
