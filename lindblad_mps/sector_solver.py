"""Exact strong-symmetry-sector steady states beyond dense reach (N <= 12).

Why this exists
---------------
exact.sector_steady_state builds the dense (charge, charge) block of the
vectorized generator, which is (4^N/4)^2 entries: fine at N = 6, impossible at
N = 8 and beyond. TEBD reaches large N but, for the driven classical circuit,
has to relax for ~12/q time units (q = |<11|L''|00>|^2, i.e. 540-6800 units at
epsilon = 0.2) -- about 60 h per N = 12 run, and Trap 5 of CLAUDE.md is the
record of what happens when that schedule is cut short.

This module instead solves L(rho) = 0, Tr rho = 1 directly, matrix-free, with
rho kept as a dense (n, n) matrix on the n = 2^(N-1) basis states of one
parity sector. Nothing is truncated and no time is evolved, so the result
carries neither the chi caveat (Trap 1) nor the relaxation caveat (Traps 2, 5):
its only error is the linear-solver residual, which is computed and returned.

Method
------
Bordered system. With t the trace functional and e = |0...0><0...0|,

    A'(rho) = L(rho) + e * Tr(rho),     A'(rho) = e

is nonsingular whenever the sector steady state is unique, and its solution
has Tr rho = 1 and L(rho) = 0 (t^T L = 0 by trace preservation splits the two
equations). Solved with GMRES.

Matvec. L(rho) = -i Heff rho + i rho Heff^dag + sum_k L_k rho L_k^dag with
Heff = H - (i/2) sum_k L_k^dag L_k; every operator is a sparse (n, n) matrix
restricted to the sector, so one matvec is a few dozen sparse-dense products.

Preconditioner: an approximate inverse of the bordered *classical* part. On
the populations it is exact: the classical jumps sqrt(r)|f><i| act there as
the classical master equation W, inverted by a dense LU of (W + e 1^T). On
the coherences |a><b| it keeps only the damping -(G_a + G_b)/2 (G = diagonal of
sum_k L_k^dag L_k) and drops the jump terms, which DO move coherences between
configurations that differ outside the bond (|f,x><f,y| <- |i,x><i,y|). That
makes it approximate, never wrong: GMRES corrects for it and the returned
residual is computed with the full generator. L'' enters only through the
matvec.

Input/Output conventions follow exact.py: basis index = integer whose most
significant of N bits is site 0; bond operators act on (i, i+1) with local
index 2*left + right. With periodic=True the ring adds the bond (N-1, 0),
left = site N-1, so a rightward drift continues around the ring.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import vectorize


def sector_configs(N: int, charge: int = 1) -> np.ndarray:
    """Sorted basis-state integers of the given Z_2 parity sector.

    Input: N, number of sites; charge, +1 (even number of ones) or -1 (odd).
    Output: 1-D int array of length 2^(N-1).
    """
    states = np.arange(2**N)
    popcount = np.array([bin(k).count("1") for k in states])
    parity = np.where(popcount % 2, -1, 1)
    return states[parity == charge]


def restricted_bond_operator(
    op: np.ndarray, i: int, N: int, configs: np.ndarray
) -> sp.csr_matrix:
    """A (4, 4) bond operator on (i, i+1), embedded and restricted to a sector.

    The operator must preserve parity (commute with Z (x) Z); this is asserted
    by checking that no weight leaves the sector.

    Input: op, (4, 4) ndarray; i, left site of the bond; N, number of sites;
        configs, from sector_configs.
    Output: (n, n) CSR matrix, n = len(configs).
    """
    full = sp.kron(
        sp.kron(sp.identity(2**i, format="csr"), sp.csr_matrix(op)),
        sp.identity(2 ** (N - i - 2), format="csr"),
        format="csr",
    )
    cols = full[:, configs]
    restricted = cols[configs, :]
    leak = abs(abs(cols).sum() - abs(restricted).sum())
    assert leak < 1e-12, f"bond operator leaks out of the parity sector ({leak:.2e})"
    return restricted.tocsr()


def restricted_pair_operator(
    op: np.ndarray, a: int, b: int, N: int, configs: np.ndarray
) -> sp.csr_matrix:
    """A (4, 4) operator on the ordered site pair (a, b), restricted to a sector.

    Same as restricted_bond_operator but for any pair -- used for the wrap
    bond (N-1, 0) of a ring, which is not a contiguous Kronecker factor.
    Local index is 2 * bit_a + bit_b.

    Input: op, (4, 4) ndarray; a, b, distinct sites; N; configs.
    Output: (n, n) CSR matrix, n = len(configs).
    """
    sa, sb = N - 1 - a, N - 1 - b
    pos = np.full(2**N, -1, dtype=np.int64)
    pos[configs] = np.arange(len(configs))
    local = 2 * ((configs >> sa) & 1) + ((configs >> sb) & 1)
    rest = configs & ~((1 << sa) | (1 << sb))
    rows, cols, vals = [], [], []
    for t in range(4):
        for s in range(4):
            if op[t, s] == 0:
                continue
            src = np.flatnonzero(local == s)
            dst = rest[src] | ((t >> 1) << sa) | ((t & 1) << sb)
            tgt = pos[dst]
            assert np.all(tgt >= 0), "pair operator leaks out of the parity sector"
            rows.append(tgt)
            cols.append(src)
            vals.append(np.full(len(src), op[t, s], dtype=complex))
    n = len(configs)
    if not rows:
        return sp.csr_matrix((n, n), dtype=complex)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def is_classical_jump(op: np.ndarray, tol: float = 1e-14) -> bool:
    """Whether op = c |f><i| for basis states f != i (one off-diagonal entry)."""
    nz = np.argwhere(np.abs(op) > tol)
    return len(nz) == 1 and nz[0][0] != nz[0][1]


class SectorLiouvillian:
    """Matrix-free Lindblad generator on one parity sector of an open chain or ring.

    Input (constructor):
        L2_terms: list of (bond jump operator (4, 4), rate), each applied on
            every bond (i, i+1), i = 0..N-2, plus (N-1, 0) if periodic.
        N: number of sites.
        charge: sector parity (+1 or -1).
        H2_terms: optional list of (bond Hamiltonian (4, 4), coefficient).
        periodic: add the wrap bond (N-1, 0); needs N >= 3.
    """

    def __init__(self, L2_terms, N: int, charge: int = 1, H2_terms=(),
                 periodic: bool = False):
        if periodic and N < 3:
            raise ValueError("a ring needs N >= 3")
        self.N = N
        self.periodic = periodic
        self.bonds = [(i, i + 1) for i in range(N - 1)] + ([(N - 1, 0)] if periodic else [])
        self.charge = charge
        self.configs = sector_configs(N, charge)
        self.n = len(self.configs)

        n = self.n
        H = sp.csr_matrix((n, n), dtype=complex)
        for op, c in H2_terms:
            for a, b in self.bonds:
                H = H + c * self._bond(op, a, b)

        self.jumps = []          # (L, L^dag) pairs, rates folded in
        self.classical = []      # (f_idx, i_idx, rate) arrays, for the preconditioner
        K = sp.csr_matrix((n, n), dtype=complex)
        for op, rate in L2_terms:
            for a, b in self.bonds:
                L = np.sqrt(rate) * self._bond(op, a, b)
                L.eliminate_zeros()
                self.jumps.append((L, L.conj().T.tocsr()))
                K = K + (L.conj().T @ L)
                if is_classical_jump(op):
                    coo = L.tocoo()
                    self.classical.append((coo.row, coo.col, np.abs(coo.data) ** 2))

        self.K = K.tocsr()

        # d rho/dt = J rho + rho J^dag + sum_k L_k rho L_k^dag,  J = -i H - K/2
        self.J = (-1j * H - 0.5 * K).tocsr()
        self.vac = int(np.flatnonzero(self.configs == 0)[0]) if charge == 1 else 0

        # Block form of the same generator, used by apply(). Every term is a
        # bond term, so per bond the generator is one 16 x 16 local
        # superoperator G acting on |s><t| (s, t = bond states, row-major
        # 4 s + t as in vectorize). Grouping the sector basis by bond state,
        # pos[b][s] lists the configs whose bond-b bits are s, sorted by value
        # -- hence by the rest of the chain, consistently across the two s of
        # one parity, which is all G connects. rho then splits into 16 blocks
        # rho[pos[s], pos[t]] of size (n/4, n/4) and the bond acts as
        # out_blocks = G @ blocks, a single dense product.
        self.G = vectorize.liouvillian_generator(
            [(op, c) for op, c in H2_terms], list(L2_terms), d=4)
        self.pos = []
        # The wrap bond's bits are sites N-1 and 0; with both fixed, sorting by
        # value still sorts by sites 1..N-2, so the block alignment holds.
        for a, b in self.bonds:
            bond_bits = (2 * ((self.configs >> (N - 1 - a)) & 1)
                         + ((self.configs >> (N - 1 - b)) & 1))
            self.pos.append([np.flatnonzero(bond_bits == st) for st in range(4)])
        m = self.n // 4
        assert all(len(p) == m for ps in self.pos for p in ps)

    def _bond(self, op, a, b):
        if b == a + 1:
            return restricted_bond_operator(op, a, self.N, self.configs)
        return restricted_pair_operator(op, a, b, self.N, self.configs)

    # -- generator --------------------------------------------------------
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """L(rho) for a dense (n, n) matrix (Hermitian or not)."""
        n, m = self.n, self.n // 4
        out = np.zeros((n, n), dtype=complex)
        blocks = np.empty((16, m, m), dtype=complex)
        for pos in self.pos:
            for s in range(4):
                rows = rho[pos[s]]
                for t in range(4):
                    blocks[4 * s + t] = rows[:, pos[t]]
            new = np.tensordot(self.G, blocks, axes=(1, 0))
            for s in range(4):
                rs = pos[s][:, None]
                for t in range(4):
                    out[rs, pos[t]] += new[4 * s + t]
        return out

    def apply_sparse(self, rho: np.ndarray) -> np.ndarray:
        """L(rho) from the sparse many-body operators (reference for apply)."""
        out = self.J @ rho
        out += (self.J @ rho.conj().T).conj().T
        for L, Ld in self.jumps:
            out += (L @ (L @ rho).conj().T).conj().T   # (L (L rho)^dag)^dag = L rho L^dag
        return out

    def _bordered(self, v: np.ndarray) -> np.ndarray:
        rho = v.reshape(self.n, self.n)
        out = self.apply(rho)
        out[self.vac, self.vac] += np.trace(rho)
        return out.reshape(-1)

    # -- classical preconditioner ------------------------------------------
    def _build_preconditioner(self) -> None:
        n = self.n
        W = np.zeros((n, n))
        for f, i, r in self.classical:
            np.add.at(W, (f, i), r)
            np.add.at(W, (i, i), -r)
        W[self.vac, :] += 1.0    # the border e 1^T, on populations only
        self._lu = sla.lu_factor(W)
        G = np.real(self.K.diagonal())
        D = -0.5 * (G[:, None] + G[None, :])
        np.fill_diagonal(D, 1.0)  # populations are handled by the LU
        assert np.all(np.abs(D) > 1e-12), "a coherence has zero classical damping"
        self._D = D

    def _precondition(self, v: np.ndarray) -> np.ndarray:
        y = v.reshape(self.n, self.n)
        x = y / self._D
        np.fill_diagonal(x, sla.lu_solve(self._lu, np.diag(y).copy()))
        return x.reshape(-1)

    # -- solve --------------------------------------------------------------
    def steady_state(self, rtol: float = 1e-11, restart: int = 60,
                     maxiter: int = 200, method: str = "gmres") -> dict:
        """Solve L(rho) = 0, Tr rho = 1 in this sector.

        Output: dict with 'rho' ((n, n) Hermitian, trace 1), 'residual'
            (||L rho||_F / ||rho||_F, the same quantity residual.py reports
            for MPS), 'hermiticity' (||rho - rho^dag|| / ||rho|| before
            symmetrization), 'iterations' (GMRES inner iterations, or
            BiCGSTAB iterations at two matvecs each).

        method: 'gmres' (restarted, keeps restart+1 Krylov vectors of n^2
            complex entries -- ~1.07 GB each at N = 14) or 'bicgstab'
            (a fixed handful of vectors; the memory-bounded choice at N >= 14).
        """
        if self.charge != 1:
            raise NotImplementedError("border vector assumes the vacuum is in the sector")
        if not self.classical:
            raise ValueError("preconditioner needs the classical baseline jumps")
        self._build_preconditioner()
        dim = self.n * self.n
        A = spla.LinearOperator((dim, dim), matvec=self._bordered, dtype=complex)
        M = spla.LinearOperator((dim, dim), matvec=self._precondition, dtype=complex)
        b = np.zeros((self.n, self.n), dtype=complex)
        b[self.vac, self.vac] = 1.0

        count = [0]

        def cb(_):
            count[0] += 1

        if method == "gmres":
            x, info = spla.gmres(A, b.reshape(-1), M=M, rtol=rtol, atol=0.0,
                                 restart=restart, maxiter=maxiter,
                                 callback=cb, callback_type="pr_norm")
        elif method == "bicgstab":
            x, info = spla.bicgstab(A, b.reshape(-1), M=M, rtol=rtol, atol=0.0,
                                    maxiter=maxiter * restart, callback=cb)
        else:
            raise ValueError(f"unknown method '{method}'")
        if info != 0:
            raise RuntimeError(f"{method} did not converge (info={info})")
        rho = x.reshape(self.n, self.n)
        herm = np.linalg.norm(rho - rho.conj().T) / np.linalg.norm(rho)
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / np.trace(rho).real
        resid = np.linalg.norm(self.apply(rho)) / np.linalg.norm(rho)
        return {"rho": rho, "residual": float(resid), "hermiticity": float(herm),
                "iterations": count[0]}


def renyi2_xx(rho: np.ndarray, configs: np.ndarray, N: int, i: int, j: int) -> float:
    """R(i, j) = Tr[rho A rho^dag A] / Tr[rho^dag rho], A = X_i X_j, in-sector.

    A flips two bits, so it preserves parity and acts on the sector as a
    permutation pi; then (A rho^dag A)[b, a] = conj(rho[pi a, pi b]).

    Input: rho, (n, n) sector density matrix; configs, its basis; N; i != j.
    Output: real float.
    """
    mask = (1 << (N - 1 - i)) | (1 << (N - 1 - j))
    pos = np.empty(2**N, dtype=np.int64)
    pos[configs] = np.arange(len(configs))
    pi = pos[configs ^ mask]
    num = np.vdot(rho[np.ix_(pi, pi)], rho)  # sum conj(rho[pi,pi]) * rho
    return float(num.real / np.vdot(rho, rho).real)


def embed_sector(rho: np.ndarray, configs: np.ndarray, N: int) -> np.ndarray:
    """The full (2^N, 2^N) density matrix of a sector-supported rho (testing aid)."""
    full = np.zeros((2**N, 2**N), dtype=complex)
    full[np.ix_(configs, configs)] = rho
    return full
