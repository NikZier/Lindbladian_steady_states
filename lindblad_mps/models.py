"""Model definitions for the parity-symmetric dissipative chain and the
random parity-commuting perturbations used in the Renyi-2 / SWSSB study.

Physics setup
-------------
The global strong Z_2 symmetry is P = Z_1 Z_2 ... Z_N. For a nearest-neighbour
(two-site) operator O embedded with identities elsewhere,

    [O_embedded, P] = ((x)_{k != a,a+1} Z_k) (x) [O, Z_a (x) Z_{a+1}],

so O commutes with the *global* parity iff it commutes with Z (x) Z on the
bond -- a purely local condition. Operators commuting with
Z (x) Z = diag(+1, -1, -1, +1) are block-diagonal in the even-parity subspace
{|00>, |11>} and the odd-parity subspace {|01>, |10>}: two independent 2x2
blocks, an 8-complex-dimensional space. Equivalently, the eight Pauli strings
that commute with Z (x) Z are {I,Z} (x) {I,Z} and {X,Y} (x) {X,Y}.

Baseline jump operators (applied uniformly to every bond):
    L  = X_a X_{a+1} (1 - Z_a Z_{a+1})                       (Hermitian)
    L' = X_a X_{a+1} (1 - Z_a)(1 - Z_{a+1})                  (non-Hermitian)
both commute with Z (x) Z. A random perturbation L'' is drawn from the
8-dim parity-commuting space and rescaled to a fixed operator norm epsilon.

All operators here are dense 2x2 (single-site) or 4x4 (two-site) physical
operators in the ordinary computational basis, matching the (operator,
coefficient) term convention consumed by vectorize / exact / tebd.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Single-site Pauli operators (Z |0> = +|0>, Z |1> = -|1>).
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {"I": I2, "X": X, "Y": Y, "Z": Z}

# The eight two-site Pauli strings (name_a, name_b) that commute with Z (x) Z.
ZZ_COMMUTING_STRINGS = [
    (a, b)
    for a in ("I", "Z", "X", "Y")
    for b in ("I", "Z", "X", "Y")
    if (a in ("I", "Z")) == (b in ("I", "Z"))
]


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product, thin alias for np.kron kept for readability."""
    return np.kron(a, b)


def zz() -> np.ndarray:
    """Return the two-site parity operator Z (x) Z as a 4x4 matrix."""
    return kron(Z, Z)


def operator_norm(M: np.ndarray) -> float:
    """Operator (spectral) norm of M: its largest singular value.

    Input: M, a square ndarray.
    Output: float, max singular value of M.
    """
    return float(np.linalg.svd(M, compute_uv=False)[0])


def commutes_with_zz(M: np.ndarray, tol: float = 1e-12) -> bool:
    """Test whether a two-site (4x4) operator commutes with Z (x) Z.

    By the factorization in the module docstring this is equivalent to the
    embedded operator commuting with the global parity P = Z_1...Z_N.

    Input: M, a (4, 4) ndarray; tol, absolute tolerance on ||[M, ZZ]||.
    Output: bool.
    """
    ZZ = zz()
    return bool(np.linalg.norm(M @ ZZ - ZZ @ M) < tol)


def baseline_jump_operators() -> list[np.ndarray]:
    """Return the two baseline bond jump operators [L, L'] as 4x4 matrices.

        L  = X_a X_{a+1} (1 - Z_a Z_{a+1})
        L' = X_a X_{a+1} (1 - Z_a)(1 - Z_{a+1})

    Both commute with Z (x) Z (verified in tests). Output operators are in
    the ordinary two-site computational basis, ready to pass as bond terms
    (op, coefficient) to vectorize / tebd with coefficient = rate.

    Output: list [L, L_prime], each a (4, 4) complex ndarray.
    """
    XX = kron(X, X)
    ZaZb = kron(Z, Z)
    Za = kron(Z, I2)
    Zb = kron(I2, Z)
    I4 = np.eye(4, dtype=complex)

    L = XX @ (I4 - ZaZb)
    L_prime = XX @ (I4 - Za) @ (I4 - Zb)
    return [L, L_prime]


def project_to_zz_commutant(M: np.ndarray) -> np.ndarray:
    """Project a 4x4 operator onto the commutant of Z (x) Z.

    Z (x) Z = diag(+1, -1, -1, +1); its commutant keeps only matrix entries
    that connect basis states of equal parity (indices {0, 3} for +1,
    {1, 2} for -1). Entries mixing the two parity sectors are zeroed.

    Input: M, a (4, 4) ndarray.
    Output: (4, 4) ndarray, the parity-block-diagonal part of M.
    """
    parity = np.array([+1, -1, -1, +1])
    mask = parity[:, None] == parity[None, :]
    return M * mask


def random_zz_commuting_operator(
    epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a random parity-commuting bond operator of operator norm epsilon.

    Draws a complex Ginibre 4x4 matrix (iid standard-normal real and
    imaginary parts), projects it onto the 8-dim commutant of Z (x) Z, and
    rescales so its operator norm equals epsilon. The result commutes with
    the global parity P = Z_1...Z_N when applied to any bond, and is in
    general non-Hermitian -- the most generic parity-symmetric jump operator.

    Input:
        epsilon: target operator norm (largest singular value).
        rng: a numpy Generator (pass a seeded one for reproducibility).
    Output: (4, 4) complex ndarray with operator_norm(.) == epsilon.
    """
    while True:
        M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        M = project_to_zz_commutant(M)
        norm = operator_norm(M)
        if norm > 1e-9:  # essentially never fails; guard the degenerate draw
            return M * (epsilon / norm)


def pauli_string_decomposition(M: np.ndarray, tol: float = 1e-9) -> dict[str, complex]:
    """Decompose a two-site operator into its two-site Pauli-string coefficients.

    Uses c_{ab} = Tr[(P_a (x) P_b)^dagger M] / 4 for each of the 16 Pauli
    strings (Tr[(P_a (x) P_b)^dagger (P_c (x) P_d)] = 4 delta). Asserts that
    the coefficients on strings *not* commuting with Z (x) Z are negligible
    (they must vanish for a parity-commuting M), then returns only the eight
    commuting-string coefficients keyed as "AB" (e.g. "XX", "ZI").

    Input: M, a (4, 4) ndarray (expected parity-commuting); tol, tolerance on
        the forbidden coefficients.
    Output: dict mapping the 8 allowed two-site Pauli-string labels to their
        complex coefficients. Summing coeff * kron(P_a, P_b) reconstructs M.
    """
    coeffs: dict[str, complex] = {}
    for a in ("I", "X", "Y", "Z"):
        for b in ("I", "X", "Y", "Z"):
            P = kron(PAULIS[a], PAULIS[b])
            c = np.trace(P.conj().T @ M) / 4.0
            allowed = (a, b) in ZZ_COMMUTING_STRINGS
            if not allowed:
                assert abs(c) < tol, (
                    f"parity-forbidden Pauli string {a}{b} has coefficient {c}"
                )
            else:
                coeffs[a + b] = complex(c)
    return coeffs


def operator_from_pauli_coeffs(coeffs: dict[str, complex]) -> np.ndarray:
    """Reconstruct a two-site operator from its Pauli-string coefficients.

    Inverse of pauli_string_decomposition(): sum coeff * (P_a (x) P_b) over
    the labels in `coeffs`. Lets a pickled description be rebuilt exactly
    (and, applied bond by bond, extended to any system size).

    Input: coeffs, dict mapping two-site Pauli labels "AB" to complex values.
    Output: (4, 4) complex ndarray.
    """
    M = np.zeros((4, 4), dtype=complex)
    for label, c in coeffs.items():
        a, b = label[0], label[1]
        M += c * kron(PAULIS[a], PAULIS[b])
    return M


def describe_operator(
    M: np.ndarray, epsilon: float, seed: int | None = None
) -> dict:
    """Build a pickle-friendly description of a bond operator.

    Bundles everything needed to identify and exactly reconstruct L'' later
    (including at larger system sizes): the raw matrix, its parity-commuting
    Pauli-string coefficients, its operator norm, the target epsilon, and the
    seed that generated it.

    Input: M, the (4, 4) operator; epsilon, its intended operator norm; seed,
        the RNG seed used (or None).
    Output: dict with keys 'matrix', 'pauli_coeffs', 'operator_norm',
        'epsilon', 'seed'.
    """
    return {
        "matrix": M.copy(),
        "pauli_coeffs": pauli_string_decomposition(M),
        "operator_norm": operator_norm(M),
        "epsilon": epsilon,
        "seed": seed,
    }
