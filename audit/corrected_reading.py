#!/usr/bin/env python3
r"""corrected_reading.py — reproduces the paper's self-correction: on a small
budget grid the compression time appears ordered by the norm budget, but
enlarging the grid dissolves that ordering, leaving the lag and floor law intact.
Reports Spearman(rho, T_compress) on the small grid vs the full grid.
Usage: python corrected_reading.py --small sample_data/mlp_grid_small --full sample_data"""
import glob, argparse
import numpy as np
from scipy.stats import spearmanr

def tcompress_spearman(folder, eps=0.10, floor_frac=0.10, grok_thr=0.90, min_drop=1.0, drop_thr=0.25):
    rows = []
    for f in sorted(glob.glob(f"{folder}/metrics/*.npz")):
        z = np.load(f, allow_pickle=True); rho = float(z["rho"]); steps = z["steps"].astype(float)
        eff = z["eff"].astype(float); acc = z["acc"].astype(float); S, T = eff.shape; nf = max(1, int(floor_frac*T))
        tg, tc = [], []
        for s in range(S):
            e, a = eff[s], acc[s]; g = np.where(a >= grok_thr)[0]
            if len(g) == 0: continue
            gi = g[0]; Tg = steps[gi]; eg = e[gi]; fl = np.median(e[-nf:]); drop = eg - fl
            if drop < min_drop or drop/eg < drop_thr: continue  # boundary gate
            thr = fl + eps*drop; post = np.where(steps >= Tg)[0]; ci = next((i for i in post if e[i] <= thr), None)
            if ci is None: ci = post[-1]
            tg.append(Tg); tc.append(steps[ci])
        if tg: rows.append((rho, np.median(tc)))
    rows.sort(); rh = [r[0] for r in rows]; tc = [r[1] for r in rows]
    return rh, (spearmanr(rh, tc).correlation if len(rh) >= 3 else float("nan"))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--small", default="sample_data/mlp_grid_small")
    ap.add_argument("--full", default="sample_data"); args = ap.parse_args()
    print("="*84); print("CORRECTED READING — Spearman(rho, T_compress): small grid vs full grid"); print("="*84)
    for task in ["mlp_modadd_p59", "mlp_modmult_p59"]:
        rs, ss = tcompress_spearman(f"{args.small}/{task}"); rf, sf = tcompress_spearman(f"{args.full}/{task}")
        name = "addition" if "add" in task else "multiplication"
        print(f"\n{name}:")
        print(f"  small grid ({len(rs)} non-boundary cells {rs}): Spearman(rho,Tc) = {ss:+.2f}")
        print(f"  full  grid ({len(rf)} non-boundary cells {rf}): Spearman(rho,Tc) = {sf:+.2f}")
    print("\n" + "="*84)
    print("A same-signed ordering on the small grid (a handful of points) dissolves on the")
    print("full grid: the audit's own 'a correlation on a handful of points is not an ordering'.")

if __name__ == "__main__":
    main()
