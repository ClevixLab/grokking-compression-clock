#!/usr/bin/env python3
r"""make_atgrok_vs_converged_figure.py — the failure-mode figure: effective rank
read AT grokking vs the converged floor, both as functions of the norm budget.
Shows that the at-grok snapshot gives a qualitatively wrong complexity-vs-budget
picture: it cannot separate the non-compressing boundary cell from a strongly-
compressing neighbour (they read nearly the same at grok), and it overstates the
converged complexity several-fold. Usage: python make_atgrok_vs_converged_figure.py"""
import glob, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def profile(folder):
    rows = []
    for f in sorted(glob.glob(f"{folder}/metrics/*.npz")):
        z = np.load(f, allow_pickle=True); rho = float(z["rho"])
        eff = z["eff"].astype(float); acc = z["acc"].astype(float)
        S, T = eff.shape; nf = max(1, int(0.1*T)); eg = []; fl = []; af = []
        for s in range(S):
            g = np.where(acc[s] >= 0.9)[0]
            if len(g) == 0: continue
            eg.append(eff[s, g[0]]); fl.append(np.median(eff[s, -nf:])); af.append(acc[s, -1])
        if eg: rows.append((rho, float(np.median(eg)), float(np.median(fl)), float(np.median(af))))
    rows.sort(); return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="sample_data")
    ap.add_argument("--out", default="figures/fig_atgrok_vs_converged"); args = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (fold, title) in zip(axes, [("mlp_modadd_p59", "Modular addition"),
                                         ("mlp_modmult_p59", "Modular multiplication")]):
        rows = profile(f"{args.data}/{fold}")
        rho = np.array([r[0] for r in rows]); eg = np.array([r[1] for r in rows])
        fl = np.array([r[2] for r in rows]); af = np.array([r[3] for r in rows])
        ax.plot(rho, eg, "o-", color="C3", lw=2, ms=6, label="read AT grokking (snapshot)")
        ax.plot(rho, fl, "s-", color="C0", lw=2, ms=6, label="converged (after long training)")
        ax.fill_between(rho, fl, eg, color="0.8", alpha=0.5, zorder=0, label="transient (discarded)")
        # mark the boundary cell if it did not fully generalize (acc < 0.999)
        bnd = np.where(af < 0.999)[0]
        for i in bnd:
            ax.scatter([rho[i]], [fl[i]], s=160, facecolors="none", edgecolors="k", lw=1.6, zorder=5)
            ax.annotate("boundary cell\n(does not compress)", (rho[i], fl[i]),
                        textcoords="offset points", xytext=(28, -6), fontsize=7.5,
                        arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.set_xlabel(r"norm budget $\rho$"); ax.set_ylabel("embedding effective rank")
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(args.out + ".pdf", bbox_inches="tight"); plt.savefig(args.out + ".png", dpi=140, bbox_inches="tight")
    print("wrote", args.out + ".pdf / .png")
    # print the sharp numbers for the text
    add = profile(f"{args.data}/mlp_modadd_p59")
    d = {round(r[0],2): r for r in add}
    print(f"\nADD: at-grok rho=1.00->{d[1.00][1]:.1f}, rho=1.05->{d[1.05][1]:.1f} (nearly identical, delta={d[1.05][1]-d[1.00][1]:.1f})")
    print(f"     converged rho=1.00->{d[1.00][2]:.1f}, rho=1.05->{d[1.05][2]:.1f} (factor {d[1.00][2]/d[1.05][2]:.1f} apart)")
    over = np.median([r[1]/r[2] for r in add if r[3] >= 0.999])
    print(f"     at-grok overstates converged by median {over:.1f}x across fully-generalizing cells")

if __name__ == "__main__":
    main()
