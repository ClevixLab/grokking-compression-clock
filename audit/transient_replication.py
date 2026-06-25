#!/usr/bin/env python3
r"""
transient_replication.py
===========================================================================
Checks that the paper's CENTRAL claim --- the effective rank read AT grokking
is a transient well above the converged value --- is not specific to the
norm-clamp protocol or to modular arithmetic. It recomputes, for a free
weight-decay sweep at p=59 on three tasks (modular addition, modular
multiplication, and parity), the at-grok rank vs. the converged floor in every
fully-generalizing cell.

It ALSO reports, honestly, the one thing that does NOT carry over: the depth
law's DIRECTION. Under the clamp protocol the converged floor falls with the
norm budget (Spearman -1.0); under free weight decay it rises with the decay
strength (Spearman +1.0), and the two do not collapse onto a single
floor-vs-converged-norm curve. So the transient replicates; the depth law's
sign is protocol-specific.

Usage: python transient_replication.py --data sample_data/mlp_freewd_p59
"""
import os, glob, re, argparse
import numpy as np
from scipy.stats import spearmanr


def load_freewd(cfg_dir, task):
    cells = {}
    for f in glob.glob(os.path.join(cfg_dir, f"mlp_{task}_wd*_s*", "metrics.npz")):
        wd = float(re.search(r"wd([\d.]+)_s", f).group(1))
        z = np.load(f, allow_pickle=True)
        cells.setdefault(wd, []).append(
            (z["eff_embed"].astype(float), z["test_acc"].astype(float),
             z["global_norm"].astype(float)))
    out = []
    for wd in sorted(cells):
        s = cells[wd]; T = min(len(x[0]) for x in s); nf = max(1, int(0.1 * T))
        eff = np.median(np.stack([x[0][:T] for x in s]), 0)
        acc = np.median(np.stack([x[1][:T] for x in s]), 0)
        gn = np.median(np.stack([x[2][:T] for x in s]), 0)
        gi = np.where(acc >= 0.9)[0]
        if len(gi) == 0 or acc[-1] < 0.99:
            continue
        out.append(dict(wd=wd, eff_grok=float(eff[gi[0]]),
                        floor=float(np.median(eff[-nf:])), norm=float(np.median(gn[-nf:]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data/mlp_freewd_p59")
    args = ap.parse_args()
    cfg = os.path.join(args.data, "configs")

    print("=" * 92)
    print("TRANSIENT REPLICATION — free weight-decay sweep, p=59, three tasks")
    print("Central claim: at-grok effective rank > converged floor in every generalizing cell")
    print("=" * 92)
    print(f"{'task':>10} {'cells':>6} {'at-grok(med)':>12} {'converged(med)':>14} {'overstate':>10} {'all>?':>6}")
    for task in ["modadd", "modmult", "parity"]:
        rows = load_freewd(cfg, task)
        if not rows:
            print(f"{task:>10}  (no generalizing cells found)"); continue
        eg = np.median([r["eff_grok"] for r in rows]); fl = np.median([r["floor"] for r in rows])
        ratios = [r["eff_grok"] / r["floor"] for r in rows]
        allgt = all(r["eff_grok"] > r["floor"] for r in rows)
        print(f"{task:>10} {len(rows):>6} {eg:>12.1f} {fl:>14.1f} "
              f"{np.median(ratios):>9.1f}x {str(allgt):>6}")

    print("\n" + "-" * 92)
    print("HONEST CAVEAT — the depth law's DIRECTION is protocol-specific:")
    for task in ["modadd", "modmult", "parity"]:
        rows = load_freewd(cfg, task)
        if len(rows) < 3:
            continue
        wd = [r["wd"] for r in rows]; fl = [r["floor"] for r in rows]; nm = [r["norm"] for r in rows]
        print(f"  {task:>8}: Spearman(wd, floor)={spearmanr(wd, fl).correlation:+.2f}  "
              f"Spearman(conv-norm, floor)={spearmanr(nm, fl).correlation:+.2f}")
    print("  (clamp protocol, for contrast: Spearman(rho, floor) = -1.0 on both modular tasks)")
    print("  => under free weight decay the floor orders OPPOSITELY to the clamp budget and")
    print("     does not unify through the converged norm; the depth law is a clamp-protocol result.")
    print("=" * 92)


if __name__ == "__main__":
    main()
