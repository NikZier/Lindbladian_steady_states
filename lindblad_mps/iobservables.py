"""Renyi-2 correlator profile and correlation length for a canonical iMPS.

Both quantities need `state` to already be canonicalize()'d -- canonicalizing
is the expensive step (Arnoldi solves), so it is not repeated inside these
functions, letting one canonical state be reused for many operators/ranges.

Which Lambda-weighting the sweep must use (a fixed bug -- read this)
---------------------------------------------------------------------
The forward sweep MUST use LEFT-weighted ("A-form") tensors,
A_s = diag(Lambda_left) Gamma_s, and must close the right end against
diag(Lambda_right**2).

The first version of this module used RIGHT-weighted tensors for the forward
sweep (and no Lambda**2 closure), reasoning that computing a dressed
numerator and an undressed denominator "in lock-step" would cancel any
convention error in the ratio. That reasoning is wrong, and the resulting
correlator was badly wrong -- it manufactured an apparent exponential DECAY
in r out of a state whose correlator is genuinely flat. The lock-step ratio
cancels an overall scale, but not a wrong Lambda-weighting: the operator
insertions sit at two specific sites, so numerator and denominator weight the
intervening environment differently and the error does not divide out.

The correct pairing follows from the two Vidal canonical conditions (the same
ones canonicalize() enforces and tests/test_imps.py checks directly):

    transfer_step           (forward)  preserves I  <=>  LEFT-weighted theta
    transfer_step_transpose (backward) preserves I  <=>  RIGHT-weighted theta

Measured on a converged chi=48 state, one unit cell of undressed forward
sweep: ||E - I|| = 1.4e+01 with right-weighted tensors versus 1.2e-06 with
left-weighted. With the correct weighting the undressed denominator stays 1
by construction (sum of Lambda**2), which is itself a cheap self-check --
and it is computed rather than assumed, so a caller passing a
not-quite-canonical state still gets a correctly normalized ratio.

Why the original product-state tests did not catch it: at bond dimension 1
every weighting is trivially equivalent. The regression test that does catch
it (tests/test_imps.py) tiles the iMPS into a finite open-boundary MPS and
compares against observables.renyi2_correlator_mps, the finite-chain routine
already validated against dense ED -- cross-checking against independently
tested machinery, not against this module's own conventions.
"""

import numpy as np

from . import imps


def correlator_profile(state: "imps.iMPS", O: np.ndarray, r_max: int = 100) -> list[tuple[int, float]]:
    """R(0, r) = Tr[rho A rho^dagger A^dagger] / Tr[rho^dagger rho], A = O_0 O_r^dagger.

    Site 0 is fixed as the first 'A' tensor of the unit cell (translation
    invariance makes this an arbitrary but harmless choice); sites alternate
    A, B, A, B, ... Computed in ONE sweep for every r in 1..r_max: dress site
    0 with kron(O, conj(O)), then advance an UNDRESSED environment site by
    site (transfer_step with op=None); at each r, PEEK by dressing a COPY of
    the current environment with kron(O^dagger, conj(O^dagger)) and closing
    it against diag(Lambda_right**2) for that site, without committing the
    dressing to the propagated environment -- the same "one state, many r"
    pattern the finite study's per-run `profile` already uses
    (renyi2_swssb.run_steady_state_correlator).

    Uses LEFT-weighted (A-form) tensors and a Lambda_right**2 closure. See the
    module docstring: this pairing is required for correctness, and getting it
    wrong (right-weighted, no closure) silently fabricates an exponential
    decay in r.

    Input:
        state: a canonicalize()'d iMPS.
        O: (local_dim, local_dim) local order-parameter operator.
        r_max: largest separation to compute.
    Output: list of (r, R(0,r)) pairs for r = 1..r_max.
    """
    Odag = O.conj().T
    Op_i = np.kron(O, O.conj())
    Op_j = np.kron(Odag, Odag.conj())

    # A-form: each site weighted by the bond to its LEFT. In the chain
    # ... Lambda_B Gamma_A Lambda_A Gamma_B Lambda_B ..., the bond left of A
    # is Lambda_B and the bond left of B is Lambda_A.
    Theta = {
        "A": imps.left_weighted(state.Gamma["A"], state.Lambda["B"]),
        "B": imps.left_weighted(state.Gamma["B"], state.Lambda["A"]),
    }
    # Bond to the RIGHT of each site, for the closure.
    Lambda_right = {"A": state.Lambda["A"], "B": state.Lambda["B"]}

    def key_at(k: int) -> str:
        return "A" if k % 2 == 0 else "B"

    chi0 = state.Gamma["A"].shape[0]
    X0 = np.eye(chi0, dtype=complex)

    num_env = imps.transfer_step(Theta[key_at(0)], X0, Op_i)
    den_env = imps.transfer_step(Theta[key_at(0)], X0, None)

    profile: list[tuple[int, float]] = []
    for r in range(1, r_max + 1):
        k = key_at(r)
        th = Theta[k]
        lam2 = Lambda_right[k] ** 2
        numerator = np.sum(np.diag(imps.transfer_step(th, num_env, Op_j)) * lam2)
        denominator = np.sum(np.diag(imps.transfer_step(th, den_env, None)) * lam2)
        value = (numerator / denominator).real if abs(denominator) > 0 else float("nan")
        profile.append((r, value))
        num_env = imps.transfer_step(th, num_env, None)
        den_env = imps.transfer_step(th, den_env, None)

        # Rescale BOTH environments by ONE common factor. Exactly ratio-
        # preserving: R(r) is linear in each environment and both are advanced
        # by the same undressed transfer step, so a shared scale divides out of
        # every subsequent numerator/denominator identically -- this changes no
        # value, it only keeps them representable.
        #
        # Needed because the loop iterates the transfer step r_max (=100) times
        # with no normalization. The state is re-canonicalized only every N
        # steps, so between canonicalizations the transfer operator's leading
        # eigenvalue is not exactly 1 and both environments drift geometrically:
        # a 1% per-site excess is harmless (1.01^100 ~ 2.7), but a
        # far-from-canonical state mid-anneal overflows float64 well before
        # r=100 and poisons precisely the large-r values the SWSSB claim rests
        # on. Observed as "overflow/invalid encountered in dot" in production.
        scale = np.linalg.norm(den_env)
        if scale > 0.0 and np.isfinite(scale):
            num_env = num_env / scale
            den_env = den_env / scale

    return profile


def correlation_length(
    state: "imps.iMPS", bond: str = "A", dense_threshold: int = 100, degeneracy_rtol: float = 1e-3
) -> dict:
    """Correlation length from the sub-leading eigenvalue of the composed 2-site transfer operator.

    Reuses the same construction canonicalize() builds internally
    (imps.theta_pair + imps.build_transfer_operator on either bond --
    eigenvalues of A-then-B equal those of B-then-A by the cyclic-composition
    property, so either bond works equally well). eta1 should equal 1.0 to
    solver tolerance if `state` was canonicalize()'d immediately before this
    call (not re-verified here -- see tests/test_imps.py for where that IS
    checked).

    Since the composed operator advances 2 physical sites per application,
    and asymptotically R(r) ~ (eta2/eta1)^(r/2) (r counted in single sites,
    the composed operator advancing every 2 of them),

        xi = -2 / ln|eta2/eta1|

    in units of one physical site.

    This formula is numerically unstable exactly where the physics is most
    interesting: a state whose R(r) has genuinely flattened (SWSSB long-range
    order) drives eta2 -> eta1, and log|eta2/eta1| -> 0 blows the naive
    formula up to an arbitrarily large, non-repeatable number (measured: the
    same sample's xi came back as 750, 11299, and 455 at chi=64, 128, 256 --
    not a converging sequence, just noise in barely-resolved near-degenerate
    eigenvalues) that reads as an error rather than as "very flat". Rather
    than report that number, treat 1-|eta2/eta1| < degeneracy_rtol as
    UNRESOLVED and return xi=None with a reason -- symmetric with the
    already-existing xi=None for the opposite extreme (eta2 ~ 0, no
    sub-leading channel at all, e.g. a bond-dim-1 product state). Both ends
    of the ratio are cases where a specific finite xi is not a meaningful
    thing to report from this data; eta1/eta2 are always returned so a
    caller can inspect further (e.g. track how close to degenerate the ratio
    is across a chi sweep) rather than losing the information.

    Input: state, a canonicalize()'d iMPS; bond, which bond's transfer
        operator to use ('A' or 'B', physically equivalent); dense_threshold:
        see imps.leading_eigenpairs; degeneracy_rtol: how close |eta2/eta1|
        may approach 1 before xi is considered unresolved.
    Output: dict with 'xi' (float, or None -- see 'reason' when None),
        'eta1', 'eta2' (complex -- a nonzero phase on eta2 signals
        oscillatory rather than purely exponential decay), 'reason' (None if
        xi was computed; 'no_subleading_channel' or 'near_degenerate'
        otherwise).
    """
    thetas = imps.theta_pair(state, bond)
    op = imps.build_transfer_operator(thetas, transpose=False)
    vals, _ = imps.leading_eigenpairs(op, k=2, dense_threshold=dense_threshold)

    eta1 = vals[0]
    eta2 = vals[1] if len(vals) > 1 else 0.0 + 0.0j

    if abs(eta1) == 0 or abs(eta2) < 1e-13 * max(abs(eta1), 1e-300):
        return {"xi": None, "eta1": eta1, "eta2": eta2, "reason": "no_subleading_channel"}

    ratio = abs(eta2 / eta1)
    if 1.0 - ratio < degeneracy_rtol:
        return {"xi": None, "eta1": eta1, "eta2": eta2, "reason": "near_degenerate"}

    xi = -2.0 / np.log(ratio)
    return {"xi": float(xi), "eta1": eta1, "eta2": eta2, "reason": None}
