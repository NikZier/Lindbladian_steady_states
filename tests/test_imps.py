"""Validation tests for imps.py, itebd.py and iobservables.py.

Style matches test_residual.py: docstrings state what each test would catch,
cross-checks against independently-coded references wherever cheaply
possible, kept fast (chi <~ 8, steps <~ 30) since these run in the same
pytest suite as everything else.
"""

import numpy as np
import pytest
import scipy.linalg

from lindblad_mps import exact, imps, iobservables, itebd, models, observables, vectorize
from lindblad_mps import mps as mps_module

KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)


def _einsum_step(theta, X, op=None):
    """Independently-coded (full einsum, not tensordot chains) reference for
    imps.transfer_step -- a direct translation of its docstring formula."""
    if op is None:
        return np.einsum("lsr,lm,msq->rq", theta.conj(), X, theta, optimize=True)
    return np.einsum("lsr,lm,st,mtq->rq", theta.conj(), X, op, theta, optimize=True)


def _einsum_step_transpose(theta, X, op=None):
    """Independently-coded reference for imps.transfer_step_transpose."""
    if op is None:
        return np.einsum("lsr,rq,msq->lm", theta, X, theta.conj(), optimize=True)
    return np.einsum("lsr,rq,st,mtq->lm", theta, X, op, theta.conj(), optimize=True)


def _random_theta(chi_l, d, chi_r, rng):
    return rng.standard_normal((chi_l, d, chi_r)) + 1j * rng.standard_normal((chi_l, d, chi_r))


def _drift_to_noncanonical(chi_max=4, seed=1, n_steps=8, local_dim=2):
    """A state produced by pure simple-update (no canonicalize call), for
    testing canonicalize() on something genuinely non-canonical: the cheap
    local update is not variationally optimal once truncation has bitten,
    exactly the situation canonicalize() exists to correct (see imps.py's
    module docstring). The random generator is scaled down (matching the
    physical study's epsilon ~ 0.2 scale, not an unscaled Ginibre draw) so
    the drift is genuine without pushing Lambda entries to the point of
    triggering apply_bond_gate's floored-reciprocal regularization mid-run --
    that regime is exercised deliberately in TestGaugeFixingRegularization,
    not here."""
    rng = np.random.default_rng(seed)
    d = local_dim * local_dim
    L_rand = 0.3 * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    state = imps.iMPS.maximally_mixed(local_dim)
    itebd.evolve_infinite(
        state, [], [], [(L_rand, 1.0)], [], dt=0.02, n_steps=n_steps,
        chi_max=chi_max, cutoff=0.0, canonicalize_every=10**9,
    )
    return state


class TestDarkStateExactInvariance:
    """|00><00| is an exact steady state of the baseline (both L, L' annihilate
    it at every bond -- proven this session in residual.py's dark-state test),
    so it must come back exactly unchanged from evolve_infinite, at bond
    dimension 1, with canonicalize() reporting eigenvalue 1.0. Compares the
    reconstructed physical content (not raw Gamma/Lambda, which carry an
    arbitrary SVD phase gauge)."""

    def test_zero_state_unchanged_under_evolution(self):
        L, L_prime = models.baseline_jump_operators()
        state = imps.iMPS.pure_product_state(KET0, KET0)
        itebd.evolve_infinite(
            state, [], [], [(L, 1.0), (L_prime, 1.0)], [], dt=0.1, n_steps=20,
            chi_max=4, cutoff=1e-10, canonicalize_every=5,
        )

        expected = vectorize.vec(np.outer(KET0, KET0.conj()))
        expected_op = np.outer(expected, expected.conj())
        for name in ("A", "B"):
            assert state.Lambda[name].shape == (1,)
            v = state.Gamma[name][0, :, 0]
            got_op = np.outer(v, v.conj())
            np.testing.assert_allclose(got_op, expected_op, atol=1e-8)

        diag = state.canonicalize(chi_max=4, cutoff=1e-10)
        assert abs(diag["eigenvalue_left_env"] - 1.0) < 1e-6
        assert abs(diag["eigenvalue_right_env"] - 1.0) < 1e-6


class TestCanonicalFormSelfConsistency:
    """The anchor for the whole gauge-fixing implementation: after
    canonicalize(), the merged 2-site unit cell (treated as a single
    translation-invariant site with one external bond, Lambda['B']) must
    satisfy the true Vidal canonical-form conditions -- rebuilt independently
    here (not by trusting canonicalize()'s own reported diagnostics) starting
    from a deliberately non-canonical state.

    Checked via the merged cell, not via theta_pair/build_transfer_operator
    (used elsewhere for correlation_length): those always right-weight both
    composed steps, which is correct for the RIGHT-canonical target
    (sum_s Gamma_s Lambda^2 Gamma_s^dagger = I, matching
    transfer_step_transpose on a right-weighted theta) but not the LEFT one
    (sum_s Gamma_s^dagger Lambda^2 Gamma_s = I, which needs a LEFT-weighted
    theta) -- exactly the distinction that was this implementation's second
    bug, so the test needs to use the same left/right-weighted construction
    canonicalize() itself does, not a shortcut that would silently pass on
    a half-fixed state.
    """

    def test_merged_cell_eigenvalue_is_one_after_canonicalize(self):
        state = _drift_to_noncanonical()
        state.canonicalize(chi_max=8, cutoff=0.0)

        merged = imps._merge_unit_cell(state)
        theta_L = imps.left_weighted(merged, state.Lambda["B"])
        theta_R = imps.right_weighted(merged, state.Lambda["B"])
        op_fwd = imps._single_site_transfer_operator(theta_L, transpose=False)
        op_bwd = imps._single_site_transfer_operator(theta_R, transpose=True)
        vals_fwd, _ = imps.leading_eigenpairs(op_fwd, k=1, dense_threshold=10_000)
        vals_bwd, _ = imps.leading_eigenpairs(op_bwd, k=1, dense_threshold=10_000)
        assert abs(vals_fwd[0] - 1.0) < 1e-7
        assert abs(vals_bwd[0] - 1.0) < 1e-7

    def test_vidal_conditions_hold_after_canonicalize(self):
        """This is the test that would catch a transpose-vs-adjoint bug in
        transfer_step_transpose, or a left/right-weighting mixup, before it
        hides inside ARPACK: sum_s Gamma_s^dagger Lambda^2 Gamma_s = I and
        sum_s Gamma_s Lambda^2 Gamma_s^dagger = I are the two defining
        conditions of Vidal canonical form, checked directly (one
        contraction, no eigenvalue solve) on the merged cell."""
        state = _drift_to_noncanonical()
        state.canonicalize(chi_max=8, cutoff=0.0)

        merged = imps._merge_unit_cell(state)
        Lambda_B = state.Lambda["B"]
        chi = merged.shape[0]
        cond_a = np.einsum("lsr,l,lsq->rq", merged.conj(), Lambda_B**2, merged, optimize=True)
        cond_b = np.einsum("lsr,r,msr->lm", merged, Lambda_B**2, merged.conj(), optimize=True)
        np.testing.assert_allclose(cond_a, np.eye(chi), atol=1e-6)
        np.testing.assert_allclose(cond_b, np.eye(chi), atol=1e-6)

    def test_canonicalize_is_idempotent_in_observables(self):
        """Calling canonicalize() twice must not change any observable --
        NOT asserting the raw tensors are unchanged, since SVD gauge freedom
        (arbitrary phases) is not physically meaningful and would make that
        comparison flaky."""
        state = _drift_to_noncanonical()
        state.canonicalize(chi_max=8, cutoff=0.0)
        xi1 = iobservables.correlation_length(state)["xi"]
        prof1 = iobservables.correlator_profile(state, models.X, r_max=5)

        state.canonicalize(chi_max=8, cutoff=0.0)
        xi2 = iobservables.correlation_length(state)["xi"]
        prof2 = iobservables.correlator_profile(state, models.X, r_max=5)

        assert (xi1 is None) == (xi2 is None)
        if xi1 is not None:
            assert abs(xi1 - xi2) < 1e-6
        for (r1, v1), (r2, v2) in zip(prof1, prof2):
            assert r1 == r2
            assert abs(v1 - v2) < 1e-6


class TestTransferOperatorPrimitives:
    """build_transfer_operator's matvec checked against dense references built
    via full einsum (not tensordot chains) -- coded independently enough that
    an axis-ordering bug in the tensordot implementation would not silently
    reproduce here."""

    def test_transfer_step_matches_independent_einsum(self):
        rng = np.random.default_rng(0)
        theta = _random_theta(3, 4, 5, rng)
        X = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        op = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))

        np.testing.assert_allclose(
            imps.transfer_step(theta, X, None), _einsum_step(theta, X, None), atol=1e-10
        )
        np.testing.assert_allclose(
            imps.transfer_step(theta, X, op), _einsum_step(theta, X, op), atol=1e-10
        )

    def test_transfer_step_transpose_matches_independent_einsum(self):
        rng = np.random.default_rng(1)
        theta = _random_theta(3, 4, 5, rng)
        X = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        op = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))

        np.testing.assert_allclose(
            imps.transfer_step_transpose(theta, X, None),
            _einsum_step_transpose(theta, X, None), atol=1e-10,
        )
        np.testing.assert_allclose(
            imps.transfer_step_transpose(theta, X, op),
            _einsum_step_transpose(theta, X, op), atol=1e-10,
        )

    def test_build_transfer_operator_matches_composed_independent_steps(self):
        rng = np.random.default_rng(2)
        chi, d = 3, 4
        theta1 = _random_theta(chi, d, chi, rng)
        theta2 = _random_theta(chi, d, chi, rng)
        v = rng.standard_normal(chi * chi) + 1j * rng.standard_normal(chi * chi)
        X = v.reshape(chi, chi)

        op_fwd = imps.build_transfer_operator((theta1, theta2), transpose=False)
        expected_fwd = _einsum_step(theta2, _einsum_step(theta1, X, None), None)
        np.testing.assert_allclose(op_fwd.matvec(v), expected_fwd.reshape(-1), atol=1e-9)

        op_bwd = imps.build_transfer_operator((theta1, theta2), transpose=True)
        expected_bwd = _einsum_step_transpose(theta1, _einsum_step_transpose(theta2, X, None), None)
        np.testing.assert_allclose(op_bwd.matvec(v), expected_bwd.reshape(-1), atol=1e-9)

    def test_leading_eigenvalue_dense_and_arpack_paths_agree(self):
        """A physically-generated (not fully random) transfer operator, since
        an unstructured random matrix has no well-separated leading
        eigenvalue and would make this comparison flaky by construction."""
        state = _drift_to_noncanonical(chi_max=6, seed=7, n_steps=6)
        op = imps.build_transfer_operator(imps.theta_pair(state, "A"), transpose=False)
        vals_dense, _ = imps.leading_eigenpairs(op, k=1, dense_threshold=10**9)
        vals_arpack, _ = imps.leading_eigenpairs(op, k=1, dense_threshold=1)
        assert abs(vals_dense[0] - vals_arpack[0]) < 1e-6 * max(abs(vals_dense[0]), 1.0)


class TestSimpleUpdateGate:
    """Same pattern as test_mps_tebd.py's TestApplyTwoSiteGate: an untruncated
    simple-update step must exactly reproduce the dense two-site contraction
    it is a truncated SVD of."""

    def test_apply_bond_gate_matches_dense_two_site_contraction_untruncated(self):
        rng = np.random.default_rng(4)
        Gamma_A = _random_theta(2, 4, 3, rng)
        Gamma_B = _random_theta(3, 4, 2, rng)
        Lambda_A = np.abs(rng.standard_normal(3)) + 0.1
        Lambda_B = np.abs(rng.standard_normal(2)) + 0.1
        state = imps.iMPS(Gamma_A, Gamma_B, Lambda_A, Lambda_B)

        gate = scipy.linalg.expm(
            0.1 * (rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))
        )

        theta_before = np.einsum(
            "l,lsm,m,mtr,r->lstr", Lambda_B, Gamma_A, Lambda_A, Gamma_B, Lambda_B, optimize=True
        )
        expected = np.einsum(
            "IJst,lstr->lIJr", gate.reshape(4, 4, 4, 4), theta_before, optimize=True
        )

        imps.apply_bond_gate(state, "A", gate, chi_max=None, cutoff=None)

        theta_after = np.einsum(
            "l,lsm,m,mtr,r->lstr",
            state.Lambda["B"], state.Gamma["A"], state.Lambda["A"], state.Gamma["B"],
            state.Lambda["B"], optimize=True,
        )
        np.testing.assert_allclose(theta_after, expected, atol=1e-8)


class TestGaugeFixingRegularization:
    """A state with one Lambda entry near zero (one truncating gate step from
    a product state, so most of the newly-opened bond's weight is
    negligible) must canonicalize to finite output, not NaN/Inf from an
    ill-conditioned pinv."""

    def test_near_singular_state_does_not_crash(self):
        rng = np.random.default_rng(5)
        state = imps.iMPS.pure_product_state(KET0, KET0)
        L_rand = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        itebd.evolve_infinite(
            state, [], [], [(L_rand, 1.0)], [], dt=0.1, n_steps=1,
            chi_max=4, cutoff=0.0, canonicalize_every=10**9,
        )
        diag = state.canonicalize(chi_max=4, cutoff=0.0)

        for name in ("A", "B"):
            assert np.all(np.isfinite(state.Gamma[name]))
            assert np.all(np.isfinite(state.Lambda[name]))
            assert np.all(state.Lambda[name] >= -1e-10)
        assert np.isfinite(diag["eigenvalue_left_env"])
        assert np.isfinite(diag["eigenvalue_right_env"])


class TestCorrelationLength:
    def test_product_state_reports_no_subleading_channel(self):
        """A genuine bond-dimension-1 product steady state has no sub-leading
        transfer channel at all -- must report xi=None, not raise on log(0)."""
        state = imps.iMPS.pure_product_state(KET0, KET0)
        state.canonicalize()
        result = iobservables.correlation_length(state)
        assert result["xi"] is None
        assert result["reason"] == "no_subleading_channel"

    def test_near_degenerate_eigenvalues_report_unresolved_not_a_huge_number(self):
        """xi = -2/ln|eta2/eta1| diverges as eta2 -> eta1, which happens
        exactly when a profile has genuinely flattened (the interesting
        case, not an error case) -- measured directly on production data:
        the same sample's xi came back as 750, 11299, and 455 at three
        different chi, not a converging sequence, just noise amplified by
        the log singularity. A near-degenerate ratio must report xi=None
        with a reason, not a number that looks like real data."""
        chi, d = 2, 4
        # An almost-unitary-like theta (small perturbation off identity)
        # gives eta2 close to eta1 by construction, without needing to find
        # one empirically.
        rng = np.random.default_rng(9)
        theta = np.zeros((chi, d, chi), dtype=complex)
        theta[0, 0, 0] = 1.0
        theta[1, 0, 1] = 1.0
        theta += 1e-6 * (rng.standard_normal(theta.shape) + 1j * rng.standard_normal(theta.shape))
        state = imps.iMPS(theta.copy(), theta.copy(), np.ones(chi), np.ones(chi))

        result = iobservables.correlation_length(state, bond="A", dense_threshold=10_000)
        assert result["xi"] is None
        assert result["reason"] == "near_degenerate"
        assert 1.0 - abs(result["eta2"] / result["eta1"]) < 1e-3

    def test_matches_independently_computed_toy_spectrum(self):
        rng = np.random.default_rng(6)
        chi, d = 2, 4
        theta1 = _random_theta(chi, d, chi, rng)
        theta2 = _random_theta(chi, d, chi, rng)

        dense = np.zeros((chi * chi, chi * chi), dtype=complex)
        col = 0
        for l in range(chi):
            for lp in range(chi):
                X = np.zeros((chi, chi), dtype=complex)
                X[l, lp] = 1.0
                Y = _einsum_step(theta2, _einsum_step(theta1, X, None), None)
                dense[:, col] = Y.reshape(-1)
                col += 1
        expected = np.sort(np.abs(np.linalg.eigvals(dense)))[::-1][:2]

        # theta_pair(state, 'B') = (Theta_A, Theta_B) = (theta1, theta2) when
        # Lambda_A = Lambda_B = 1, matching dense's (theta1 first, theta2 second).
        state = imps.iMPS(theta1.copy(), theta2.copy(), np.ones(chi), np.ones(chi))
        result = iobservables.correlation_length(state, bond="B", dense_threshold=10_000)
        got = np.sort(np.abs([result["eta1"], result["eta2"]]))[::-1]
        np.testing.assert_allclose(got, expected, rtol=1e-6)


class TestEvolveInfiniteConvergence:
    """evolve_infinite must find a genuinely known steady state, not just
    something self-consistent. Uses a single-site dissipative toy model (not
    the full SWSSB model) with no bond terms at all: the infinite chain then
    factorizes into independent single-site problems, so its steady state
    can be cross-checked against dense ED at N=1 (exact.steady_state)."""

    def test_finds_known_single_site_steady_state(self):
        SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^- (lowering)
        SX = np.array([[0, 1], [1, 0]], dtype=complex)
        H1_terms = [(SX, 0.3)]
        L1_terms = [(SM, 1.0)]

        rho_exact = exact.steady_state([], H1_terms, [], L1_terms, N=1)

        state, _ = itebd.find_steady_state_infinite(
            [], H1_terms, [], L1_terms,
            dt_schedule=[0.2, 0.05, 0.01], steps_per_dt=200,
            chi_max=4, cutoff=1e-12, canonicalize_every=10,
            initial_state=imps.iMPS.pure_product_state(KET0, KET0),
        )

        for name in ("A", "B"):
            v = state.Gamma[name][0, :, 0]
            rho_raw = vectorize.unvec(v, 2)
            rho_site = rho_raw / np.trace(rho_raw)
            np.testing.assert_allclose(rho_site, rho_exact, atol=1e-4)


class TestPerSiteVidalConditions:
    """canonicalize() must leave the INDIVIDUAL Gamma/Lambda in true Vidal
    form, not merely make the merged 2-site cell canonical.

    This distinction was a real bug: canonicalize() coarse-grains the unit
    cell and gauge-fixes that, and an early version then split the merged
    tensor back with a plain SVD -- no outer-Lambda weighting, none divided
    back out. The merged-cell conditions then held to ~1e-7 while the
    per-site conditions were violated by ~1e+7. Evolution never noticed
    (gates and the merged canonicalization only use the composite), but
    correlator_profile sweeps site by site, so it silently returned a badly
    wrong correlator. Checking the merged cell alone -- which is what
    TestCanonicalFormSelfConsistency does -- cannot catch this."""

    @pytest.mark.parametrize("seed", [1, 5])
    def test_individual_gammas_are_vidal_canonical(self, seed):
        state = _drift_to_noncanonical(chi_max=8, seed=seed, n_steps=10)
        state.canonicalize(chi_max=8, cutoff=1e-12)

        for name, lam_left, lam_right in (
            ("A", state.Lambda["B"], state.Lambda["A"]),
            ("B", state.Lambda["A"], state.Lambda["B"]),
        ):
            G = state.Gamma[name]
            left = np.einsum("lsr,l,lsq->rq", G.conj(), lam_left**2, G, optimize=True)
            right = np.einsum("lsr,r,msr->lm", G, lam_right**2, G.conj(), optimize=True)
            np.testing.assert_allclose(left, np.eye(left.shape[0]), atol=1e-6,
                                       err_msg=f"Gamma_{name} not left-canonical")
            np.testing.assert_allclose(right, np.eye(right.shape[0]), atol=1e-6,
                                       err_msg=f"Gamma_{name} not right-canonical")


class TestCorrelatorAgainstFiniteChain:
    """correlator_profile cross-checked against observables.renyi2_correlator_mps
    -- the finite-chain routine already validated against dense ED.

    This is the test that catches a wrong Lambda-weighting in the sweep. Two
    such bugs shipped before this existed: (1) correlator_profile used
    RIGHT-weighted tensors for a forward sweep with no Lambda**2 closure, and
    (2) canonicalize() produced non-Vidal per-site tensors (see
    TestPerSiteVidalConditions). Either alone made the infinite-system
    correlator manufacture an exponential DECAY in r out of a state whose
    correlator is genuinely flat -- and neither was visible to the
    product-state tests below, where bond dimension 1 makes every weighting
    equivalent."""

    def _tile(self, state, n_cells):
        """Finite open-boundary MPS from the iMPS, A-form tensors, boundary
        bonds sliced to 1. Boundary error is confined near the ends and decays
        with the correlation length, so BULK correlators are comparable."""
        A = imps.left_weighted(state.Gamma["A"], state.Lambda["B"])
        B = imps.left_weighted(state.Gamma["B"], state.Lambda["A"])
        tensors = []
        for _ in range(n_cells):
            tensors.append(A.copy())
            tensors.append(B.copy())
        tensors[0] = tensors[0][0:1, :, :]
        tensors[-1] = tensors[-1][:, :, 0:1]
        return mps_module.MPS(tensors, local_dim=2)

    def test_matches_finite_chain_in_the_bulk(self):
        rng = np.random.default_rng(3)
        L_rand = 0.3 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        L, L_prime = models.baseline_jump_operators()
        state = imps.iMPS.pure_product_state(KET0, KET1)
        itebd.evolve_infinite(
            state, [], [], [(L, 1.0), (L_prime, 1.0), (L_rand, 1.0)], [],
            dt=0.1, n_steps=120, chi_max=16, cutoff=1e-10, canonicalize_every=10,
        )
        state.canonicalize(chi_max=16, cutoff=1e-10)

        finite = self._tile(state, n_cells=12)
        finite.canonicalize(chi_max=None, cutoff=1e-12)
        finite.normalize()
        N = finite.N
        # Even reference site => same sublattice as the iMPS sweep's site 0.
        i = 2 * (N // 8)

        profile = dict(iobservables.correlator_profile(state, models.X, r_max=8))
        for rsep in (2, 4, 6):
            expected = observables.renyi2_correlator_mps(finite, models.X, i, i + rsep)
            assert profile[rsep] == pytest.approx(expected, rel=2e-3), (
                f"separation {rsep}: iMPS {profile[rsep]:.6e} vs finite {expected:.6e}"
            )


class TestCorrelatorProfileExactCases:
    """correlator_profile's actual VALUES checked against hand-computable
    cases -- idempotence (TestCanonicalFormSelfConsistency) only confirms
    consistency, not correctness, and this is otherwise untested at the
    value level. For ANY product state and O=X (which maps every
    computational basis state to an orthogonal one), R(i,j) factorizes into
    per-site factors Tr[rho_k O rho_k O]/Tr[rho_k^2] = 0 exactly, for every
    r -- so R must be identically 0. For O=Z (diagonal, leaves every
    computational basis state fixed up to a sign that squares away), the
    same factorization gives exactly 1 at every site, so R must be
    identically 1."""

    @pytest.mark.parametrize("ket_A,ket_B", [(KET0, KET0), (KET0, KET1)])
    def test_flip_operator_gives_zero_on_any_product_state(self, ket_A, ket_B):
        state = imps.iMPS.pure_product_state(ket_A, ket_B)
        state.canonicalize()
        profile = iobservables.correlator_profile(state, models.X, r_max=10)
        for r, v in profile:
            assert abs(v) < 1e-10, f"r={r}: {v}"

    @pytest.mark.parametrize("ket_A,ket_B", [(KET0, KET0), (KET0, KET1)])
    def test_diagonal_operator_gives_one_on_any_product_state(self, ket_A, ket_B):
        state = imps.iMPS.pure_product_state(ket_A, ket_B)
        state.canonicalize()
        profile = iobservables.correlator_profile(state, models.Z, r_max=10)
        for r, v in profile:
            assert abs(v - 1.0) < 1e-10, f"r={r}: {v}"
