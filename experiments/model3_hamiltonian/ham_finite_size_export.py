"""Flatten every FINITE-SIZE Hamiltonian-perturbation run into one table.

Reads the four study pickles written by ham_finite_size.py

    ham_finite_size.pkl                   exact N in {4,6} + TEBD N in {4,6,8,12}, chi=128
    ham_finite_size_highchi_N12_chi256.pkl   the clean N=12 row, chi=256
    ham_finite_size_highchi.pkl              N=16, chi=256   (NOT relaxed -- see below)
    ham_finite_size_convergence.pkl          the chi sweep at N in {12,14,16}

plus any `hamfin_*` entry in experiments/cache/ that no study pickle carries --
the N=12 leg of the chi sweep is only there, because --convergence overwrites
ham_finite_size_convergence.pkl and its last run used CONV_NS = [14, 16].

It writes `ham_finite_size_table.pkl`: one flat record per run carrying
(N, epsilon, correlator) plus the fields needed to decide whether a given
record may be quoted.

Every record has `method` ('exact' or 'tebd'), `kind` ('grid', 'control',
'neel', 'conv') and `source` (which pickle it came from), so the table is a
superset -- filter it, do not average it blindly. In particular:

  * chi=128 at N=12 is truncation-corrupted; the chi=256 rows supersede it.
    `supersedes_ok` marks, per record, whether a better chi exists for the
    same (kind, epsilon, sample, N) -- prefer the record with the largest
    chi_max in each group.
  * N=16 at chi=256 is NOT relaxed (spread 24-96%, drift up to 15%). It is
    included because it is real data, and flagged by `stage_drift`.
  * `kind='control'` runs have h=0 and hence c2=0, so R_over_c2 is None --
    they are the pipeline's false-positive floor, not signal.

Input:  nothing (reads experiments/model3_hamiltonian/results/*.pkl)
Output: experiments/model3_hamiltonian/results/ham_finite_size_table.pkl, a dict with
            'records'  list of dicts, one per run
            'arrays'   {'N', 'epsilon', 'correlator', 'sample', 'chi_max',
                        'c2', 'R_over_c2'} as numpy arrays, row-aligned with
                        'records' (R_over_c2 is nan where undefined)
            'columns'  the record field names, in order
            'sources'  {pickle name: config dict of that study}
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout  # noqa: E402  (puts the repo root and every study folder on sys.path)

import numpy as np

import renyi2_swssb as ex


SOURCES = [
    "ham_finite_size.pkl",
    "ham_finite_size_highchi_N12_chi256.pkl",
    "ham_finite_size_highchi.pkl",
    "ham_finite_size_convergence.pkl",
]

OUT_NAME = "ham_finite_size_table.pkl"

COLUMNS = [
    "method",
    "kind",
    "N",
    "epsilon",
    "sample",
    "chi_max",
    "init",
    "correlator",
    "c2",
    "R_over_c2",
    "stage_drift",
    "max_bond",
    "bond_at_cap",
    "max_discarded_weight",
    "residual",
    "min_profile",
    "gap_rate",
    "purity",
    "seconds",
    "source",
    "supersedes_ok",
]


def _load(name):
    """Input: pickle basename. Output: its contents, or None if absent."""
    path = os.path.join(layout.MODEL3_RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _blank():
    return {col: None for col in COLUMNS}


def _tebd_records(entries, source):
    """Input: a study's `results` dict, its pickle name.
    Output: list of flat records, one per TEBD run."""
    out = []
    for entry in entries.values():
        res = entry["result"]
        c2 = entry.get("c2")
        rec = _blank()
        rec.update(
            method="tebd",
            kind=entry["kind"],
            N=entry["N"],
            epsilon=entry["epsilon"],
            sample=entry["sample"],
            chi_max=entry["chi_max"],
            init=entry["init"],
            correlator=float(res["correlator"]),
            c2=c2,
            R_over_c2=(float(res["correlator"]) / c2 if c2 else None),
            stage_drift=res.get("stage_drift"),
            max_bond=res.get("max_bond"),
            bond_at_cap=res.get("bond_at_cap"),
            max_discarded_weight=res.get("max_discarded_weight"),
            residual=res.get("residual"),
            min_profile=res.get("min_profile"),
            seconds=entry.get("seconds"),
            source=source,
        )
        out.append(rec)
    return out


def _exact_records(entries, source):
    """Input: the `exact` dict keyed (N, epsilon, sample), its pickle name.
    Output: list of flat records, one per dense sector solve."""
    out = []
    for (N, eps, sample), val in entries.items():
        R = float(val["R"])
        ratio = val.get("R_over_c2")
        rec = _blank()
        rec.update(
            method="exact",
            kind="grid",
            N=N,
            epsilon=eps,
            sample=sample,
            correlator=R,
            c2=(R / ratio if ratio else None),
            R_over_c2=ratio,
            gap_rate=val.get("gap_rate"),
            purity=val.get("purity"),
            source=source,
        )
        out.append(rec)
    return out


def _cache_records(seen):
    """Input: set of (kind, epsilon, sample, N, init, chi_max) already collected.
    Output: list of records for the hamfin_* cache entries not among them."""
    cache_dir = ex.CACHE_DIR
    if not os.path.isdir(cache_dir):
        return []
    extra = {}
    for name in sorted(os.listdir(cache_dir)):
        if not (name.startswith("hamfin_") and name.endswith(".pkl")):
            continue
        with open(os.path.join(cache_dir, name), "rb") as fh:
            entry = pickle.load(fh)
        key = (
            entry["kind"],
            entry["epsilon"],
            entry["sample"],
            entry["N"],
            entry["init"],
            entry["chi_max"],
        )
        if key not in seen:
            extra[key] = entry
    return _tebd_records(extra, "_cache")


def _mark_superseded(records):
    """Flag, in place, whether each TEBD record is the largest chi available
    for its (kind, epsilon, sample, N, init). Exact records are always True."""
    best = {}
    for rec in records:
        if rec["method"] != "tebd":
            continue
        key = (rec["kind"], rec["epsilon"], rec["sample"], rec["N"], rec["init"])
        best[key] = max(best.get(key, 0), rec["chi_max"])
    for rec in records:
        if rec["method"] != "tebd":
            rec["supersedes_ok"] = True
            continue
        key = (rec["kind"], rec["epsilon"], rec["sample"], rec["N"], rec["init"])
        rec["supersedes_ok"] = rec["chi_max"] == best[key]


def build_table():
    """Input: none. Output: the table dict described in the module docstring."""
    records = []
    sources = {}
    for name in SOURCES:
        data = _load(name)
        if data is None:
            print(f"  (missing, skipped) {name}")
            continue
        sources[name] = data.get("config")
        records += _tebd_records(data["results"], name)
        if "exact" in data:
            records += _exact_records(data["exact"], name)

    seen = {
        (r["kind"], r["epsilon"], r["sample"], r["N"], r["init"], r["chi_max"])
        for r in records
        if r["method"] == "tebd"
    }
    records += _cache_records(seen)

    _mark_superseded(records)
    records.sort(
        key=lambda r: (
            r["method"],
            r["kind"],
            r["N"],
            -r["epsilon"],
            r["sample"],
            r["chi_max"] or 0,
        )
    )

    def col(name, dtype=float):
        vals = [r[name] for r in records]
        if dtype is float:
            return np.array(
                [np.nan if v is None else float(v) for v in vals], dtype=float
            )
        return np.array([-1 if v is None else int(v) for v in vals], dtype=int)

    arrays = {
        "N": col("N", int),
        "epsilon": col("epsilon"),
        "correlator": col("correlator"),
        "sample": col("sample", int),
        "chi_max": col("chi_max", int),
        "c2": col("c2"),
        "R_over_c2": col("R_over_c2"),
    }
    return {
        "records": records,
        "arrays": arrays,
        "columns": COLUMNS,
        "sources": sources,
    }


def report(table):
    """Input: the table dict. Output: none; prints one line per group."""
    records = table["records"]
    print(f"\n{len(records)} records\n")
    header = f"{'method':6s} {'kind':8s} {'N':>3s} {'chi':>4s} {'eps':>5s} {'n':>3s}  {'mean R':>11s}  {'mean R/|c|^2':>12s}  spread"
    print(header)
    print("-" * len(header))
    groups = {}
    for rec in records:
        key = (rec["method"], rec["kind"], rec["N"], rec["chi_max"], rec["epsilon"])
        groups.setdefault(key, []).append(rec)
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3] or 0, -k[4])):
        method, kind, N, chi, eps = key
        grp = groups[key]
        R = np.array([g["correlator"] for g in grp])
        ratios = [g["R_over_c2"] for g in grp if g["R_over_c2"] is not None]
        if ratios:
            ratios = np.array(ratios)
            ratio_txt = f"{ratios.mean():12.6f}"
            spread = f"{(ratios.max() - ratios.min()) / ratios.mean() * 100:6.2f}%"
        else:
            ratio_txt, spread = " " * 12, "     --"
        flag = "" if all(g["supersedes_ok"] for g in grp) else "   (superseded)"
        print(
            f"{method:6s} {kind:8s} {N:3d} {chi if chi else 0:4d} {eps:5.2f} "
            f"{len(grp):3d}  {R.mean():11.4e}  {ratio_txt}  {spread}{flag}"
        )


def main():
    table = build_table()
    report(table)
    os.makedirs(layout.MODEL3_RESULTS, exist_ok=True)
    path = os.path.join(layout.MODEL3_RESULTS, OUT_NAME)
    with open(path, "wb") as fh:
        pickle.dump(table, fh)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
