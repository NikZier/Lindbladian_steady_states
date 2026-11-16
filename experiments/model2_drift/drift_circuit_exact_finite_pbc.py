"""The driven classical circuit's exact finite-N grid on a RING (periodic chain).

Identical to drift_circuit_exact_finite.py -- same model, p = 0.8 and 1/2, the
same ten L'' seeds, epsilon in {0.20, 0.15, 0.10, 0.05}, L''=0 controls, the
(+,+) sector, R at (N//4, 3N//4) -- except that every bond term also acts on
the wrap bond (N-1, 0), so drift continues around the ring. N = 4..12.

On a ring the steady state is translation invariant, so R(i, j) depends only
on j - i, and the separation N/2 used here is the largest the ring has. The
dense validation at N = 4, 6 uses exact.embed_wrap_bond_operator, a tensor-axis
permutation independent of the solver's bit arithmetic.

Outputs (experiments/model2_drift/results/): drift_circuit_exact_finite_pbc.pkl,
    drift_circuit_exact_finite_pbc_RvsN.png, drift_circuit_exact_finite_pbc_profiles.png
"""

import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  (puts the repo root and every study folder on sys.path)

import drift_circuit_exact_finite as dcef

dcef.PERIODIC = True
dcef.SIZES = [4, 6, 8, 10, 12]
dcef.OUTPUT_STEM = "drift_circuit_exact_finite_pbc"

if __name__ == "__main__":
    mp.freeze_support()
    dcef.main()
