#!/usr/bin/env python3
r"""make_floor_ci_figure.py — converged effective-rank floor vs norm budget,
with bootstrap-over-seeds 95% CI bands, for both modular tasks.
Usage: python make_floor_ci_figure.py --data sample_data --out figures/floor_law_ci"""
import os, glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def floors(folder, gate, B, rng):
    cells = []
    for f in sorted(glob.glob(f"{folder}/metrics/*.npz")):
        z = np.load(f, allow_pickle=True); rho = float(z["rho"])
        if round(rho, 2) in gate: continue
        eff = z["eff"].astype(float); nf = max(1, int(0.1*eff.shape[1]))
        cells.append((rho, np.median(eff[:, -nf:], axis=1)))
    cells.sort(); rhos = np.array([c[0] for c in cells]); pf = [c[1] for c in cells]
    pt = np.array([np.median(x) for x in pf]); S = pf[0].shape[0]; bo = np.empty((B, len(cells)))
    for b in range(B):
        idx = rng.integers(0, S, S); bo[b] = [np.median(x[idx]) for x in pf]
    lo, hi = np.percentile(bo, 2.5, axis=0), np.percentile(bo, 97.5, axis=0)
    sp = spearmanr(rhos, pt).correlation
    spb = np.array([spearmanr(rhos, bo[b]).correlation for b in range(min(B, 3000))])
    return rhos, pt, lo, hi, sp, np.percentile(spb, [2.5, 97.5])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="sample_data")
    ap.add_argument("--out", default="figures/floor_law_ci"); ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args(); rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (folder, gate, title) in zip(axes, [(f"{args.data}/mlp_modadd_p59", {1.00}, "Modular addition"),
                                                 (f"{args.data}/mlp_modmult_p59", set(), "Modular multiplication")]):
        rhos, pt, lo, hi, sp, spci = floors(folder, gate, args.bootstrap, rng)
        ax.fill_between(rhos, lo, hi, alpha=0.25, color="C0", label="95% CI (bootstrap over seeds)")
        ax.plot(rhos, pt, "o-", color="C0", lw=2, ms=6, label="converged eff-rank floor (median)")
        ax.set_xlabel("norm budget " + r"$\rho$"); ax.set_ylabel("converged effective-rank floor")
        ax.set_title(title + "\n" + r"Spearman$(\rho,\mathrm{floor})=%+.2f$  95%% CI $[%+.2f,%+.2f]$" % (sp, spci[0], spci[1]), fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out + ".pdf", bbox_inches="tight"); plt.savefig(args.out + ".png", dpi=140, bbox_inches="tight")
    print("wrote", args.out + ".pdf / .png")

if __name__ == "__main__":
    main()
