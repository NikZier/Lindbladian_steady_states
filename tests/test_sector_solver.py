"""sector_solver against the dense exact.sector_steady_state reference."""

import numpy as np
import pytest

from lindblad_mps import exact, models, observables, sector_solver


def _terms(seed: int, p: float = 0.8) -> list:
    rng = np.random.default_rng(seed)
    L_pp = models.random_zz_commuting_operator(0.2, rng)
    ops = models.classical_drift_annihilation_jump_operators(p) + [L_pp]
    return [(op, 1.0) for op in ops]


@pytest.mark.parametrize("N", [4, 6])
@pytest.mark.parametrize("seed", [1, 2])
def test_matches_dense_sector_solve(N, seed):
    terms = _terms(seed)
    ref = exact.sector_steady_state([], [], terms, [], N)["rho"]
    Lv = sector_solver.SectorLiouvillian(terms, N)
    sol = Lv.steady_state()
    full = sector_solver.embed_sector(sol["rho"], Lv.configs, N)
    assert np.linalg.norm(full - ref) < 1e-9
    assert sol["residual"] < 1e-9
    for i, j in [(0, N - 1), (1, 2), (N // 4, 3 * N // 4)]:
        R_dense = observables.renyi2_correlator_dense(ref, models.X, i, j, N)
        R_fast = sector_solver.renyi2_xx(sol["rho"], Lv.configs, N, i, j)
        assert R_fast == pytest.approx(R_dense, rel=1e-8)


def test_unperturbed_is_dark_vacuum():
    terms = [(op, 1.0) for op in models.classical_drift_annihilation_jump_operators(0.8)]
    Lv = sector_solver.SectorLiouvillian(terms, 6)
    rho = Lv.steady_state()["rho"]
    assert abs(rho[Lv.vac, Lv.vac] - 1.0) < 1e-10
    assert sector_solver.renyi2_xx(rho, Lv.configs, 6, 1, 4) == pytest.approx(0.0, abs=1e-12)


def test_apply_matches_vectorized_generator():
    N = 4
    terms = _terms(3)
    Lv = sector_solver.SectorLiouvillian(terms, N)
    gen = exact.vectorize.liouvillian_generator(
        [], exact.build_jump_operators(terms, [], N), d=2**N)
    rng = np.random.default_rng(0)
    rho = rng.standard_normal((Lv.n, Lv.n)) + 1j * rng.standard_normal((Lv.n, Lv.n))
    full = sector_solver.embed_sector(rho, Lv.configs, N)
    want = (gen @ full.reshape(-1)).reshape(2**N, 2**N)[np.ix_(Lv.configs, Lv.configs)]
    assert np.allclose(Lv.apply(rho), want, atol=1e-12)


def test_block_apply_matches_sparse_apply():
    N = 8
    Lv = sector_solver.SectorLiouvillian(_terms(4), N)
    rng = np.random.default_rng(1)
    rho = rng.standard_normal((Lv.n, Lv.n)) + 1j * rng.standard_normal((Lv.n, Lv.n))
    assert np.allclose(Lv.apply(rho), Lv.apply_sparse(rho), atol=1e-12)
