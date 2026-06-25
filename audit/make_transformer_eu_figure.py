#!/usr/bin/env python3
r"""
make_transformer_eu_figure.py
===========================================================================
Plot embedding (E) and unembedding (U) effective-rank trajectories for the
checkpointed transformer cells, showing that BOTH loci collapse around the
grokking step to a shared low converged floor.

Honest framing: weight checkpoints are log-spaced, so the exact at-grok value
is bracketed rather than pinned. We therefore show the full trajectory with
the grok step marked, plus the converged floor, rather than a single at-grok
scalar for U. The embedding's fine-grained trace (eff_rank_E logged every step)
is overlaid where available to validate the checkpoint computation.

Input : sample_data/transformer_noln_p59/struct_eu/tfnoln_eu_rho*.npz
        (optionally) metrics_struct/ for the fine embedding trace.
Output: figures/fig_transformer_eu.pdf / .png
"""
import os, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rho_of(p):
    m = re.search(r"rho(\d+\.\d+)", p)
    return float(m.group(1)) if m else None


def fine_E(metrics_dir, rho):
    if not metrics_dir or not os.path.isdir(metrics_dir):
        return None, None
    for f in glob.glob(os.path.join(metrics_dir, "*.npz")):
        rf = rho_of(f)
        if rf is not None and abs(rf - rho) < 1e-3:
            z = np.load(f, allow_pickle=True)
            if "eff_rank_E" in z.files:
                return z["steps"].astype(float), np.median(z["eff_rank_E"], axis=0)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eu_dir", default="sample_data/transformer_noln_p59/struct_eu")
    ap.add_argument("--metrics_dir", default="sample_data/transformer_noln_p59/metrics_struct")
    ap.add_argument("--rhos", nargs="*", type=float, default=[0.85, 1.00])
    ap.add_argument("--out", default="figures/fig_transformer_eu")
    args = ap.parse_args()

    files = {rho_of(f): f for f in glob.glob(os.path.join(args.eu_dir, "*.npz"))}
    rhos = [r for r in args.rhos if any(abs(r - k) < 1e-3 for k in files)]
    fig, axes = plt.subplots(1, len(rhos), figsize=(5.4 * len(rhos), 4.3), squeeze=False)

    for ax, rho in zip(axes[0], rhos):
        key = [k for k in files if abs(k - rho) < 1e-3][0]
        z = np.load(files[key], allow_pickle=True)
        ck = z["ckpt_steps"].astype(float)
        effE = np.median(z["eff_E"], axis=1)     # (ckpts,)
        effU = np.median(z["eff_U"], axis=1)
        Tg = int(np.median(z["T_grok"]))
        floorE, floorU = effE[-1], effU[-1]
        x = np.clip(ck, 1, None)                  # log axis: avoid step 0

        # fine embedding trace (validation / continuity)
        sf, ef = fine_E(args.metrics_dir, rho)
        if sf is not None:
            ax.plot(np.clip(sf, 1, None), ef, color="C0", alpha=0.35, lw=1.2,
                    label="embedding (fine trace)")

        ax.plot(x, effE, "o-", color="C0", lw=2, ms=5, label="embedding E (checkpoints)")
        ax.plot(x, effU, "s--", color="C3", lw=2, ms=5, label="unembedding U (checkpoints)")
        ax.axvline(max(Tg, 1), color="0.4", ls=":", lw=1.5)
        ax.text(max(Tg, 1), ax.get_ylim()[1] * 0.96, "  grok", color="0.3",
                fontsize=8, va="top")
        ax.axhline(floorE, color="C0", ls=":", lw=0.8, alpha=0.6)
        ax.axhline(floorU, color="C3", ls=":", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("training step (log)")
        ax.set_ylabel("effective rank")
        acc = float(np.nanmedian(z["acc_final"])) if np.isfinite(z["acc_final"]).any() else np.nan
        ax.set_title(rf"$\rho={rho:.2f}$  (median acc$={acc:.3f}$)"
                     "\nboth loci collapse around grok to a shared floor "
                     rf"($\approx{0.5*(floorE+floorU):.1f}$)", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out + ".pdf", bbox_inches="tight")
    plt.savefig(args.out + ".png", dpi=140, bbox_inches="tight")
    print("wrote", args.out + ".pdf / .png")


if __name__ == "__main__":
    main()
