"""Bond-dimension (chi) convergence diagnostics for TEBD steady states.

Truncation puts a spurious floor under small quantities (e.g. long-distance
Renyi-2 correlators), which is easy to mistake for a genuine physical
signal. The tools here make chi-convergence a first-class, checkable output
instead of a spot check: the entanglement (Schmidt) spectrum at each bond
shows how much bond dimension a converged state actually needs, and
chi_convergence_scan() runs the full steady-state search at a range of
chi_max values so an observable's chi-dependence can be inspected directly.
"""

import numpy as np

from . import mps as mps_module
from . import tebd


def schmidt_spectrum(state: "mps_module.MPS") -> list[np.ndarray]:
    """Compute the Schmidt (singular value) spectrum at every bond.

    Operates on a copy of `state` via a lossless canonicalizing sweep (QR
    left-to-right, then untruncated SVD right-to-left), so it does not
    mutate the input and is not limited by whatever chi_max/cutoff the state
    was produced with -- it reports the entanglement actually present in the
    tensors, which is what determines how large chi needs to be to
    represent the state without loss.

    Input: state, an mps.MPS.
    Output: list of N-1 ndarrays (one per bond, left to right), each the
        Schmidt spectrum at that bond normalized so sum(S**2) == 1.
    """
    tmp = state.copy()
    tmp.left_canonicalize()
    d = tmp.phys_dim

    spectra: list[np.ndarray] = [np.array([]) for _ in range(tmp.N - 1)]
    for n in range(tmp.N - 1, 0, -1):
        A = tmp.tensors[n]
        chi_l, _, chi_r = A.shape
        mat = A.reshape(chi_l, d * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        norm = np.linalg.norm(S)
        spectra[n - 1] = S / norm if norm > 0 else S
        tmp.tensors[n] = Vh.reshape(len(S), d, chi_r)
        tmp.tensors[n - 1] = np.einsum("lim,ma->lia", tmp.tensors[n - 1], U * S[None, :])
    return spectra


def entanglement_entropies(state: "mps_module.MPS") -> list[float]:
    """Von Neumann entanglement entropy at every bond, from schmidt_spectrum().

    Input: state, an mps.MPS.
    Output: list of N-1 floats, S = -sum_i p_i log(p_i) with p_i = S_i^2,
        the (normalized) squared Schmidt values at that bond. A bond with
        entropy 0 carries no entanglement (bond dimension 1 suffices there);
        growing entropy as chi_max is relaxed signals the current chi is
        cutting off real correlations, not just noise.
    """
    entropies = []
    for spectrum in schmidt_spectrum(state):
        p = spectrum**2
        p = p[p > 1e-16]  # avoid log(0); values this small contribute ~0 anyway
        entropies.append(float(-np.sum(p * np.log(p))))
    return entropies


def is_chi_max_binding(state: "mps_module.MPS", chi_max: int) -> bool:
    """Check whether chi_max is actually constraining any bond of `state`.

    If no bond reaches chi_max, the entanglement in the true state is lower
    than the cap, and increasing chi_max would not change the result
    further (for this state, at this dt/step count). If some bond equals
    chi_max, that bond may be truncated and chi_max should be increased and
    rechecked.

    Input: state, an mps.MPS; chi_max, the bond-dimension cap used to produce it.
    Output: True if any bond dimension equals chi_max.
    """
    return any(chi == chi_max for chi in state.bond_dims)


def chi_convergence_scan(
    H2_terms: list[tuple[np.ndarray, float]],
    H1_terms: list[tuple[np.ndarray, float]],
    L2_terms: list[tuple[np.ndarray, float]],
    L1_terms: list[tuple[np.ndarray, float]],
    N: int,
    dt_schedule: list[float],
    steps_per_dt: int,
    chi_list: list[int],
    observable_fn,
    cutoff: float | None = None,
    d_site: int = 2,
    recanonicalize_every: int = 10,
) -> dict:
    """Run find_steady_state independently at each chi_max and compare an observable.

    Each chi in chi_list gets an independent TEBD run from a fresh
    maximally-mixed initial state (same dt_schedule/steps_per_dt for all, so
    only chi varies), making the resulting observable values directly
    comparable. This is the "run at chi, 2*chi and compare" check: if
    observable_diff has not become small and max_bond_dim has not saturated
    below chi_max, the last chi in the list has not converged and larger chi
    (or more steps) is needed before trusting the observable.

    Input:
        H2_terms, H1_terms, L2_terms, L1_terms: model terms, see
            tebd.build_bond_gates().
        N: number of sites.
        dt_schedule, steps_per_dt: passed to tebd.find_steady_state() at
            every chi (held fixed across the scan).
        chi_list: list of chi_max values to run, e.g. [4, 8, 16, 32].
        observable_fn: callable(mps.MPS) -> float, e.g.
            lambda state: observables.renyi2_correlator_mps(state, O, i, j).
        cutoff, d_site, recanonicalize_every: passed through to find_steady_state().
    Output: dict with parallel lists (one entry per chi_list element):
        'chi': the chi_max values used,
        'observable': observable_fn(state) at each chi,
        'observable_diff': None for the first entry, then
            |observable[k] - observable[k-1]| for successive entries,
        'max_bond_dim': the largest bond dimension actually present in the
            converged state at each chi (compare to 'chi' to see if the cap
            was binding),
        'chi_max_binding': is_chi_max_binding() at each chi,
        'discarded_weight': final per-step discarded SVD weight from the
            TEBD history at each chi,
        'max_entanglement_entropy': the largest per-bond entanglement
            entropy in the converged state at each chi.
    """
    results = {
        "chi": list(chi_list),
        "observable": [],
        "max_bond_dim": [],
        "chi_max_binding": [],
        "discarded_weight": [],
        "max_entanglement_entropy": [],
    }

    for chi in chi_list:
        state, history = tebd.find_steady_state(
            H2_terms,
            H1_terms,
            L2_terms,
            L1_terms,
            N,
            dt_schedule,
            steps_per_dt,
            chi_max=chi,
            cutoff=cutoff,
            d_site=d_site,
            recanonicalize_every=recanonicalize_every,
        )
        results["observable"].append(observable_fn(state))
        results["max_bond_dim"].append(max(state.bond_dims, default=1))
        results["chi_max_binding"].append(is_chi_max_binding(state, chi))
        results["discarded_weight"].append(
            history["discarded_weight"][-1] if history["discarded_weight"] else 0.0
        )
        entropies = entanglement_entropies(state)
        results["max_entanglement_entropy"].append(max(entropies, default=0.0))

    diffs: list[float | None] = [None]
    for k in range(1, len(results["observable"])):
        diffs.append(abs(results["observable"][k] - results["observable"][k - 1]))
    results["observable_diff"] = diffs

    return results
