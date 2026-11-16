"""Folder layout of experiments/: where each study's scripts and results live.

    model1_dissipative/   Model 1 (L, L' + random L''), finite TEBD and iTEBD
    model2_drift/         Model 2 (driven classical circuit), iTEBD and exact finite N
    model3_hamiltonian/   Model 3 (Hamiltonian perturbation of Model 1)
    benchmarks/           convergence and validation studies (chi sweeps,
                          schedule length, positivity, TEBD-vs-exact)
    paper/                the publication figures and the notebook drawing them
    cache/                one pickle per completed simulation, shared by all

Each study folder holds its scripts and a results/ subfolder with what they
write. Importing this module also puts the repository root and every study
folder on sys.path, so scripts keep importing each other by flat module name
(`import renyi2_swssb as ex`) wherever they live. Every script starts with

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import layout

Input: nothing.
Output: the directory constants below; sys.path extended as a side effect.
"""

import os
import sys

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EXPERIMENTS_DIR)

MODEL1_DIR = os.path.join(EXPERIMENTS_DIR, "model1_dissipative")
MODEL2_DIR = os.path.join(EXPERIMENTS_DIR, "model2_drift")
MODEL3_DIR = os.path.join(EXPERIMENTS_DIR, "model3_hamiltonian")
BENCHMARKS_DIR = os.path.join(EXPERIMENTS_DIR, "benchmarks")
PAPER_DIR = os.path.join(EXPERIMENTS_DIR, "paper")

MODEL1_RESULTS = os.path.join(MODEL1_DIR, "results")
MODEL2_RESULTS = os.path.join(MODEL2_DIR, "results")
MODEL3_RESULTS = os.path.join(MODEL3_DIR, "results")
BENCHMARKS_RESULTS = os.path.join(BENCHMARKS_DIR, "results")

CACHE_DIR = os.path.join(EXPERIMENTS_DIR, "cache")

for _d in (REPO_DIR, MODEL1_DIR, MODEL2_DIR, MODEL3_DIR, BENCHMARKS_DIR):
    if _d not in sys.path:
        sys.path.append(_d)
