"""BLAS/LAPACK thread control for the TEBD hot loop.

TEBD spends ~95% of its flops in one LAPACK call: the SVD that splits the
two-site tensor in mps.MPS.apply_two_site_gate, on a matrix of shape
(chi*d, chi*d). For the sizes this produces (chi=32, d=4 -> 128x128) the
threaded OpenBLAS that ships with numpy is *slower* than the serial path by
a large factor -- the thread launch/sync overhead dominates a factorization
this small, and unlike GEMM the divide-and-conquer SVD cannot amortize it.
Measured on a 128x128 complex matrix: 38.0 ms threaded vs 6.1 ms serial.

Environment variables (OPENBLAS_NUM_THREADS etc.) cannot fix this from
inside the package, because OpenBLAS reads them once when the shared library
loads -- i.e. at `import numpy`, which callers typically do first. threadpoolctl
instead re-configures the already-loaded library at runtime, so it works no
matter the import order.

The limit is scoped to a context manager rather than set globally: it is a
property of these particular small factorizations, not of the process, and
large dense work elsewhere (exact.py) is still better off threaded.
"""

import contextlib

try:
    import threadpoolctl
except ImportError:  # pragma: no cover - threadpoolctl is a declared dependency
    threadpoolctl = None


@contextlib.contextmanager
def limit_threads(n_threads: int | None = 1):
    """Context manager restricting BLAS/LAPACK to n_threads inside the block.

    Input:
        n_threads: thread cap to apply for the duration of the block. Pass
            None to leave the library's threading untouched (useful to
            benchmark the difference, or on a LAPACK that threads small
            factorizations well).
    Output:
        Yields None. The previous thread settings are restored on exit,
        including if the block raises.
    """
    if n_threads is None or threadpoolctl is None:
        yield
        return
    with threadpoolctl.threadpool_limits(limits=n_threads, user_api="blas"):
        yield
