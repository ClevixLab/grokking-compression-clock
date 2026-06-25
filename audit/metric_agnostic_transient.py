#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric_agnostic_transient.py
---------------------------------------------------------------------------
Shows that the at-grok representation transient (effective rank read at
grokking sits above the converged floor) is NOT an artifact of the
effective-rank metric: it appears in participation ratio and stable rank
just as it does in effective rank, on the free-weight-decay modular-addition
MLP cells shipped in sample_data/.

For every generalizing cell we report the metric value at T_grok and at
convergence (the final logged step), for three spectral measures, and the
overstatement ratio at-grok / converged.

Run:  python audit/metric_agnostic_transient.py
"""
import glob
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = sorted(glob.glob(os.path.join(
    HERE, "sample_data", "mlp_freewd_p59", "configs", "mlp_modadd_wd*_s*", "metrics.npz")))

METRICS = [("eff_embed", "effective rank"),
           ("pr_embed", "participation ratio"),
           ("sr_embed", "stable rank")]


def main():
    if not CELLS:
        raise SystemExit("no free-wd cells found under sample_data/mlp_freewd_p59/")
    agg = defaultdict(lambda: defaultdict(list))
    for f in CELLS:
        d = np.load(f, allow_pickle=True)
        wd = float(d["wd"]); steps = d["steps"]; acc = d["test_acc"]; Tg = int(d["T_grok"])
        if Tg <= 0 or acc.max() < 0.9:          # generalizing cells only
            continue
        gi = int(np.argmin(np.abs(steps - Tg)))  # index at grokking
        for key, _ in METRICS:
            v = d[key]
            agg[wd][key].append((float(v[gi]), float(v[-1])))

    hdr = f"{'wd':<6} | " + " | ".join(f"{name+' (atgrok->conv, x)':<28}" for _, name in METRICS)
    print(hdr); print("-" * len(hdr))
    all_transient = True
    for wd in sorted(agg):
        cols = []
        for key, _ in METRICS:
            pairs = agg[wd][key]
            ag = np.median([p[0] for p in pairs]); cv = np.median([p[1] for p in pairs])
            cols.append(f"{ag:5.1f} -> {cv:4.1f}  ({ag/cv:3.1f}x)")
            if not all(a > c for a, c in pairs):
                all_transient = False
        print(f"{wd:<6} | " + " | ".join(f"{c:<28}" for c in cols))
    print()
    print("Transient (at-grok > converged) holds in ALL THREE metrics "
          f"for EVERY generalizing cell: {all_transient}")


if __name__ == "__main__":
    main()
