#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
transformer_sweep_audit.py
---------------------------------------------------------------------------
Reads the real transformer clamp+free sweep (sample_data/transformer_sweep_p59/)
and reproduces the two results the paper reports from it:

  (1) The at-grok transient generalizes to the transformer (embedding E and
      unembedding U), is metric-agnostic and multi-locus  -> fig_transformer_transient.pdf
  (2) The MLP norm-budget depth law does NOT replicate on the transformer
      (a deliberate negative result / ablation)            -> fig_depthlaw_nonreplication.pdf

All numbers are computed from the npz trajectories; nothing is hard-coded.
Cells that do not reach test accuracy GROK_THR, or whose effective-rank floor
is still falling at the step cap (censored), are excluded from the converged
floor so the negative result is not a censoring artifact.

Usage:
  python audit/transformer_sweep_audit.py            # prints summary + writes figs to paper/
  python audit/transformer_sweep_audit.py --out DIR
"""
import argparse
import glob
import os

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__version__ = "1.1.0"  # 1.1.0: + lag_recurs() (section 7 lag now reproducible); makedirs(--out)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF = os.path.join(HERE, "sample_data", "transformer_sweep_p59")
MLP = os.path.join(HERE, "sample_data", "mlp_modadd_p59", "metrics")

GROK_THR = 0.90
EPS = 0.10
TAIL = 20000          # window over which "still falling" = censored is judged
SLOPE_CENS = -0.3     # eff-rank per 10k steps below which we call a cell censored


def load(p):
    m = np.load(p)
    return {k: m[k] for k in m.files}


def grok_step(m):
    g = np.where(m["test_acc"] >= GROK_THR)[0]
    return int(m["step"][g[0]]) if len(g) else -1


def conv(m, col, fw=0.2):
    s = m[col]
    w = max(5, int(fw * len(s)))
    return float(np.nanmedian(s[-w:]))


def at_grok(m, col, g):
    st, v = m["step"], m[col]
    idx = np.where(~np.isnan(v))[0]
    return float(v[idx[np.argmin(np.abs(st[idx] - g))]]) if len(idx) else np.nan


def censored(m):
    st = m["step"]
    tail = st >= (st[-1] - TAIL)
    x, y = st[tail].astype(float), m["E_erank_rv"][tail]
    ok = ~np.isnan(y)
    return (np.polyfit(x[ok], y[ok], 1)[0] * 1e4) < SLOPE_CENS if ok.sum() >= 3 else False


def tf_cells(proto):
    out = {}
    for p in sorted(glob.glob(os.path.join(TF, f"transformer__{proto}__*.npz"))):
        m = load(p)
        b, s = float(m["meta_budget"]), int(m["meta_seed"])
        out.setdefault(b, []).append((s, m))
    return out


def clean(m):
    g = grok_step(m)
    return g > 0 and np.nanmax(m["test_acc"]) >= GROK_THR and not censored(m)


def per_seed_tf_lag(m, fw=0.2, eps=EPS):
    """Canonical per-seed compression clock (identical definition to
    analyze_compression_clock_v1_5.py) applied to the transformer embedding
    effective rank. T_grok = first step with test_acc >= GROK_THR; floor =
    median E_erank_rv over the last `fw` of logged steps (same window used for
    the converged floor elsewhere in this script); T_compress = first step
    >= T_grok at which E <= floor + eps*drop. Returns ("ok", T_grok, lag),
    ("cens", T_grok, nan) if the threshold is never reached, or None if the
    cell does not grok / has too small a drop."""
    st = m["step"].astype(float)
    e = m["E_erank_rv"].astype(float)
    acc = m["test_acc"]
    g = np.where(acc >= GROK_THR)[0]
    if not len(g):
        return None
    Tg = float(st[g[0]])
    eff_g = float(e[g[0]])
    w = max(5, int(fw * len(e)))
    floor = float(np.nanmedian(e[-w:]))
    drop = eff_g - floor
    if not (np.isfinite(drop) and drop >= 1.0):
        return None
    thr = floor + eps * drop
    post = np.where((st >= Tg) & np.isfinite(e))[0]
    hit = post[e[post] <= thr]
    if not len(hit):
        return ("cens", Tg, np.nan)
    return ("ok", Tg, float(st[hit[0]]) - Tg)


def lag_recurs():
    """Section 7 'The lag recurs'. Reports the post-grok compression lag on the
    transformer and whether it orders with the norm budget, from the SAME clean
    clamp cells used for the transient and the depth law. Per-seed canonical
    clock, aggregated to a per-cell median, then summarised over cells. We print
    n and p explicitly because the un-normalised transformer is seed-fragile, so
    the ordering statistic is low-power and must not be over-read."""
    C = tf_cells("clamp")
    cell_gap, cell_tg, n_seeds = {}, {}, 0
    for b in sorted(C):
        gaps, tgs = [], []
        for s, m in C[b]:
            if not clean(m):
                continue
            r = per_seed_tf_lag(m)
            if r and r[0] == "ok":
                gaps.append(r[2]); tgs.append(r[1]); n_seeds += 1
        if gaps:
            cell_gap[b] = float(np.median(gaps))
            cell_tg[b] = float(np.median(tgs))
    rhos = sorted(cell_gap)
    g = np.array([cell_gap[r] for r in rhos])
    t = np.array([cell_tg[r] for r in rhos])
    med_lag = float(np.median(g))
    med_ratio = float(np.median(g / t))
    rs, p = spearmanr(rhos, g)
    print("  --- lag recurs: post-grok compression lag on the transformer (clean clamp cells) ---")
    print(f"    median lag = {med_lag:.0f} steps  (median lag/T_grok = {med_ratio:.2f}; "
          f"{len(rhos)} cells, {n_seeds} clean seeds)")
    sig = "significant" if p < 0.05 else "NOT significant"
    print(f"    Spearman(rho, lag) = {rs:+.2f} (p={p:.2f}, n={len(rhos)}) -> {sig}: "
          f"lag is large but not cleanly budget-ordered")
    return dict(med_lag=med_lag, med_ratio=med_ratio, rho_S=rs, p=p, n=len(rhos), n_seeds=n_seeds)


# --------------------------------------------------------------------------- #
# Summary numbers
# --------------------------------------------------------------------------- #
def summary():
    print("=" * 74)
    print("TRANSFORMER SWEEP AUDIT  (generalizing + non-censored cells; thr=%.2f)" % GROK_THR)
    print("=" * 74)
    for proto in ("clamp", "free"):
        C = tf_cells(proto)
        for loc in ("E", "U"):
            ratios = []
            for b in C:
                for s, m in C[b]:
                    if clean(m):
                        ratios.append(at_grok(m, f"{loc}_erank_rv", grok_step(m)) / conv(m, f"{loc}_erank_rv"))
            r = np.array(ratios)
            print(f"  transient {proto:5} {loc}: median {np.nanmedian(r):.2f}x  "
                  f"range [{np.nanmin(r):.2f},{np.nanmax(r):.2f}]  n={len(r)}  frac>1={np.mean(r>1):.2f}")
    # depth law (clean clamp cells)
    print("  --- depth law: Spearman(rho, converged floor), clean clamp cells ---")
    for loc in ("E", "U", "act"):
        B, F = [], []
        for b in tf_cells("clamp"):
            for s, m in tf_cells("clamp")[b]:
                if clean(m):
                    B.append(b); F.append(conv(m, f"{loc}_erank_rv"))
        rs, p = spearmanr(B, F)
        print(f"    transformer {loc}: rho_S={rs:+.2f} (p={p:.2f}, n={len(B)})")
    # MLP contrast
    bm, fm = mlp_floor_vs_rho()
    rs, p = spearmanr(bm, fm)
    print(f"    MLP (embedding, paper data): rho_S={rs:+.2f}  <- the depth law that does NOT replicate")
    # lag recurs (section 7)
    lag_recurs()


def mlp_floor_vs_rho():
    B, F = [], []
    for p in sorted(glob.glob(os.path.join(MLP, "longtrain_add_p59_rho*.npz"))):
        m = load(p)
        rho = float(m["rho"])
        eff = np.asarray(m["eff"], float)
        if eff.ndim > 1:                       # (seeds, steps) -> median over seeds
            eff = np.nanmedian(eff, axis=0)
        floor = float(np.nanmedian(eff[-max(5, int(0.1 * len(eff))):]))
        if rho <= 1.0:                         # paper gates the rho=1.00 boundary/never-compress cell
            continue
        B.append(rho); F.append(floor)
    order = np.argsort(B)
    return list(np.array(B)[order]), list(np.array(F)[order])


# --------------------------------------------------------------------------- #
# Figure 1: the transformer transient (E and U)
# --------------------------------------------------------------------------- #
def fig_transient(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, loc in zip(axes, ("E", "U")):
        C = tf_cells("clamp")
        rhos = sorted(C)
        ag_med, cv_med = [], []
        for b in rhos:
            ag_pts, cv_pts = [], []
            for s, m in C[b]:
                if clean(m):
                    ag_pts.append(at_grok(m, f"{loc}_erank_rv", grok_step(m)))
                    cv_pts.append(conv(m, f"{loc}_erank_rv"))
            if ag_pts:
                ax.scatter([b] * len(ag_pts), ag_pts, c="#c0392b", s=22, alpha=.55, zorder=3)
                ax.scatter([b] * len(cv_pts), cv_pts, c="#2471a3", s=22, alpha=.55, zorder=3)
                ag_med.append((b, np.median(ag_pts))); cv_med.append((b, np.median(cv_pts)))
        if ag_med:
            ax.plot(*zip(*ag_med), "-o", c="#c0392b", lw=2, label="read at grokking")
            ax.plot(*zip(*cv_med), "-s", c="#2471a3", lw=2, label="converged floor")
            for (b, a), (_, c) in zip(ag_med, cv_med):
                ax.fill_between([b - .012, b + .012], [c, c], [a, a], color="#c0392b", alpha=.10)
        ax.set_title(f"{'Embedding' if loc=='E' else 'Unembedding'} ($W_{loc}$)")
        ax.set_xlabel(r"norm budget $\rho$"); ax.set_ylabel("effective rank (Roy-Vetterli)")
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=.25)
    fig.suptitle("At-grok effective rank exceeds the converged floor on the transformer "
                 "(clamp sweep, generalizing non-censored seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .96])
    out = os.path.join(outdir, "fig_transformer_transient.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 2: depth-law non-replication (MLP -1.0  vs  transformer ~0)
# --------------------------------------------------------------------------- #
def fig_depthlaw(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    # MLP panel
    bm, fm = mlp_floor_vs_rho()
    rs_m, _ = spearmanr(bm, fm)
    axes[0].plot(bm, fm, "-o", c="#117a65", lw=2)
    axes[0].set_title(f"MLP embedding (paper)\nSpearman$(\\rho,\\,$floor$)={rs_m:+.2f}$ — depth law")
    axes[0].set_xlabel(r"norm budget $\rho$"); axes[0].set_ylabel("converged effective-rank floor")
    axes[0].grid(alpha=.25)
    # transformer panel (clean clamp cells, embedding)
    C = tf_cells("clamp")
    B, F = [], []
    med = []
    for b in sorted(C):
        pts = [conv(m, "E_erank_rv") for s, m in C[b] if clean(m)]
        for v in pts:
            B.append(b); F.append(v)
        if pts:
            axes[1].scatter([b] * len(pts), pts, c="#884ea0", s=24, alpha=.55, zorder=3)
            med.append((b, np.median(pts)))
    rs_t, p_t = spearmanr(B, F)
    if med:
        axes[1].plot(*zip(*med), "-o", c="#884ea0", lw=2)
    axes[1].set_title(f"Transformer embedding (this work)\nSpearman$(\\rho,\\,$floor$)={rs_t:+.2f}$ (p={p_t:.2f}) — no depth law")
    axes[1].set_xlabel(r"norm budget $\rho$"); axes[1].set_ylabel("converged effective-rank floor")
    axes[1].grid(alpha=.25)
    fig.suptitle("The norm-budget depth law is MLP-specific: it does not replicate on the transformer",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .95])
    out = os.path.join(outdir, "fig_depthlaw_nonreplication.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "paper"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary()
    fig_transient(a.out)
    fig_depthlaw(a.out)


if __name__ == "__main__":
    main()
