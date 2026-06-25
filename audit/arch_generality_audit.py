#!/usr/bin/env python3
r"""
arch_generality_audit.py   (paper section 8, Figure 6)
=====================================================================
Regenerates the "one harness, three architectures" result and Figure 6
from the RAW per-cell trajectories of an arch-generality run, with NO
re-training. It re-derives every per-seed quantity (T_grok, at-grok rank,
converged floor, T_compress, lag, frac-pre) independently from each cell's
metrics.jsonl, then aggregates per arm and renders the dashboard.

It is the section-8 counterpart of analyze_compression_clock_v1_5.py: the
clock definitions are identical (grok_thr on test accuracy, floor over the
last floor_frac of the logged trajectory, compression threshold
floor + eps*(at_grok - floor)).

INPUT  (an arch-generality run directory):
    <run>/arms/<arm>/<cell>/metrics.jsonl     # one JSON object per logged step
    <run>/arms/<arm>/<cell>/DONE              # optional: {"grokked_at": <step>}
    <run>/results/cells.csv                   # optional: harness clock, cross-checked
  where <arm> in {mlp, tf_lnfree, tf_ln, tf_canonical, tf_rms, ...}
  and  <cell> is "wd<value>_s<seed>"  (free decay)
                or "rho<value>_s<seed>" / under an arm dir ending "_clamp" (clamp).

  Required metrics.jsonl keys: step, test_acc, global_norm, and the primary
  rank key (default "embedding.sqnorm"). Any key that FALLS as the
  representation compresses can be substituted via --rank-key.

OUTPUT:
    <out>/fig_one_harness_three_arch.{pdf,png}   # Figure 6 (panels present in data)
    <out>/arch_section8_stats.csv                # per-arm transient / lag / frac-pre
    <out>/arch_section8.prov.json                # provenance

USAGE:
    python audit/arch_generality_audit.py --arch-dir <run> --out reproduced/
    python audit/arch_generality_audit.py --arch-dir <run> --out reproduced/ --rank-key embedding.rv

Panels A (free-decay rank trajectories) and B (low-budget MLP second collapse)
reproduce from a free-decay run. Panels C (LayerNorm free-vs-clamp) and D
(clamp grok-rate per rho) require clamp cells in the same run; if absent, the
panels are annotated "clamp cells not in this run" rather than fabricated.
"""
import argparse, glob, json, os, re, csv
import numpy as np

__version__ = "1.0.0"

# arm directory name -> (display label, plot color, plot order)
ARM_META = {
    "mlp":           ("MLP",                "tab:blue",   1),
    "tf_lnfree":     ("TF canonical (LN-free)", "tab:green",  0),
    "tf_canonical":  ("TF canonical (LN-free)", "tab:green",  0),
    "tf_ln":         ("TF LayerNorm",        "tab:red",    2),
    "tf_rms":        ("TF RMSNorm-emb",      "tab:purple", 3),
}

# Reference values reported in the paper (printed for the reader's convenience
# ONLY; never used in any computation). frac-pre is the robust, well-powered
# quantity; the absolute lag/T_grok is seed-sensitive (paper uses 5 seeds).
PAPER_REF = {
    "tf_lnfree": dict(transient=1.48, lag=0.23, fracpre=0.88),
    "mlp":       dict(transient=1.5,  lag=0.66, fracpre=0.63),
    "tf_ln":     dict(transient=3.0,  lag=1.84, fracpre=0.26),
}


def load_cell(path):
    """Read one metrics.jsonl -> dict of np arrays over logged steps."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
           for k in rows[0].keys() if k != "op"}
    return out


def cell_iter(arm_dir):
    """Yield (cell_name, wd_or_rho, seed, is_clamp, metrics_path) for each cell."""
    for d in sorted(glob.glob(os.path.join(arm_dir, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        mfree = re.match(r"wd([\d.]+)_s(\d+)$", name)
        mclamp = re.match(r"rho([\d.]+)_s(\d+)$", name)
        mp = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(mp):
            continue
        if mfree:
            yield name, float(mfree.group(1)), int(mfree.group(2)), False, mp
        elif mclamp:
            yield name, float(mclamp.group(1)), int(mclamp.group(2)), True, mp


def clock_one(m, rank_key, grok_thr, floor_frac, eps):
    """Re-derive the per-seed clock from a single cell's trajectory.

    Returns dict or None if the cell never groks / has no usable rank trace.
    """
    step = m["step"]
    acc = m.get("test_acc")
    rank = m.get(rank_key)
    if acc is None or rank is None or not np.isfinite(rank).any():
        return None
    grok_idx = np.where(acc >= grok_thr)[0]
    if len(grok_idx) == 0:
        return None  # did not grok
    gi = grok_idx[0]
    t_grok = step[gi]
    r_init = rank[0]
    r_grok = rank[gi]
    nfloor = max(1, int(round(floor_frac * len(step))))
    floor = float(np.mean(rank[-nfloor:]))
    drop = r_grok - floor
    # T_compress: first post-grok step within eps of the floor
    thr = floor + eps * max(drop, 0.0)
    post = np.arange(gi, len(step))
    hit = post[rank[post] <= thr]
    t_comp = step[hit[0]] if len(hit) else np.nan
    lag = (t_comp - t_grok) if np.isfinite(t_comp) else np.nan
    denom = (r_init - floor)
    fracpre = ((r_init - r_grok) / denom) if denom != 0 else np.nan
    return dict(t_grok=float(t_grok), at_grok=float(r_grok), floor=floor,
                t_comp=float(t_comp), lag=float(lag),
                fracpre=float(fracpre), transient=float(r_grok / floor) if floor > 0 else np.nan,
                r_init=float(r_init))


def analyze(arch_dir, rank_key, grok_thr, floor_frac, eps):
    arms_root = os.path.join(arch_dir, "arms")
    arm_dirs = sorted(d for d in glob.glob(os.path.join(arms_root, "*")) if os.path.isdir(d))
    per_arm = {}      # arm -> list of per-seed clock dicts (free decay, grokked)
    clamp_cells = {}  # arm -> list of (rho, seed, grokked_bool, clock_or_none)
    traj = {}         # arm -> list of (wd, seed, is_clamp, metrics dict)
    for ad in arm_dirs:
        arm = os.path.basename(ad)
        for name, val, seed, is_clamp, mp in cell_iter(ad):
            m = load_cell(mp)
            traj.setdefault(arm, []).append((val, seed, is_clamp, m))
            ck = clock_one(m, rank_key, grok_thr, floor_frac, eps)
            if is_clamp:
                grokked = (m.get("test_acc") is not None and np.nanmax(m["test_acc"]) >= grok_thr)
                clamp_cells.setdefault(arm, []).append((val, seed, bool(grokked), ck))
            else:
                if ck is not None:
                    per_arm.setdefault(arm, []).append(ck)
    return per_arm, clamp_cells, traj


def summarize(per_arm):
    """Per-arm medians for the three section-8 quantities."""
    rows = []
    for arm, cks in per_arm.items():
        tr = np.array([c["transient"] for c in cks])
        lg = np.array([c["lag"] / c["t_grok"] for c in cks if np.isfinite(c["lag"])])
        fp = np.array([c["fracpre"] for c in cks if np.isfinite(c["fracpre"])])
        rows.append(dict(arm=arm, n=len(cks),
                         transient_med=float(np.median(tr)) if len(tr) else np.nan,
                         lag_over_tg_med=float(np.median(lg)) if len(lg) else np.nan,
                         fracpre_med=float(np.median(fp)) if len(fp) else np.nan))
    rows.sort(key=lambda r: ARM_META.get(r["arm"], ("", "", 99))[2])
    return rows


def render_figure(per_arm, clamp_cells, traj, rank_key, out_base, summary_rows):
    """Clean, fully-reproducible Figure 6 (3 panels, no clamp / no fabricated panel):
      (A) free-decay rank trajectories per arm with the grok step marked;
      (B) frac-pre per arm (how much compression is done BY grok) — the ordering;
      (C) low-budget MLP cell: the converged floor is itself non-stationary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("Architecture generality: the transient is general, the lag is normalization-mediated (sec. 8)",
                 fontsize=12)

    # ---- (A) free-decay rank trajectories, representative curve per arm.
    # Representative cell = grokked free-decay seed whose frac-pre is closest to
    # that arm's median frac-pre (deterministic, documented choice).
    axA.set_title("(A) Free decay: embedding rank vs step (dot = grok)")
    for arm in sorted(per_arm, key=lambda a: ARM_META.get(a, ("", "", 99))[2]):
        label, color, _ = ARM_META.get(arm, (arm, None, 99))
        med = np.median([c["fracpre"] for c in per_arm[arm]])
        best, bestgap = None, 1e9
        for (wd, seed, is_clamp, m) in traj.get(arm, []):
            if is_clamp:
                continue
            ck = clock_one(m, rank_key, 0.9, 0.1, 0.1)
            if ck is None:
                continue
            gap = abs(ck["fracpre"] - med)
            if gap < bestgap:
                bestgap, best = gap, (m, ck)
        if best is None:
            continue
        m, ck = best
        axA.plot(m["step"], m[rank_key], color=color, lw=1.5, label=label)
        gi = int(np.argmin(np.abs(m["step"] - ck["t_grok"])))
        axA.plot(m["step"][gi], m[rank_key][gi], "o", color=color, ms=7,
                 markeredgecolor="k", zorder=5)
    axA.set_xlabel("step"); axA.set_ylabel("embedding sq-norm eff. rank")
    axA.legend(fontsize=8)

    # ---- (B) frac-pre bars: the reproducible quantity that orders the arms.
    axB.set_title("(B) Compression done BY grok (frac-pre)")
    rows = [r for r in summary_rows if np.isfinite(r["fracpre_med"])]
    rows.sort(key=lambda r: ARM_META.get(r["arm"], ("", "", 99))[2])
    xs = np.arange(len(rows))
    cols = [ARM_META.get(r["arm"], (r["arm"], "0.5", 99))[1] for r in rows]
    axB.bar(xs, [r["fracpre_med"] for r in rows], color=cols)
    for i, r in enumerate(rows):
        axB.text(i, r["fracpre_med"] + 0.015, f"{r['fracpre_med']:.2f}", ha="center", fontsize=9)
    axB.set_xticks(xs)
    axB.set_xticklabels([ARM_META.get(r["arm"], (r["arm"],))[0].replace(" ", "\n") for r in rows],
                        fontsize=7)
    axB.set_ylabel("frac-pre  (1 = all compression before grok)")
    axB.set_ylim(0, 1.05)
    axB.text(0.5, 0.93, "high frac-pre -> small lag", transform=axB.transAxes,
             ha="center", fontsize=8, color="0.3")

    # ---- (C) low-budget MLP cell: the floor is itself non-stationary.
    axC.set_title("(C) Low-budget MLP: the floor is also non-stationary")
    mlp_free = [(wd, s, m) for (wd, s, c, m) in traj.get("mlp", []) if not c]
    if mlp_free:
        wd_min = min(wd for wd, _, _ in mlp_free)
        m = [m for (wd, s, m) in mlp_free if wd == wd_min][0]
        axC.plot(m["step"], m[rank_key], color="tab:blue", lw=1.6)
        axC.set_xlabel("step"); axC.set_ylabel("emb rank", color="tab:blue")
        ax2 = axC.twinx()
        ax2.plot(m["step"], m["global_norm"], color="0.4", ls="--", lw=1.3)
        ax2.set_ylabel("|W| global norm", color="0.4")
        axC.text(0.5, 0.92, f"MLP wd={wd_min:g}: late 2nd collapse", transform=axC.transAxes,
                 ha="center", fontsize=8)
    else:
        axC.text(0.5, 0.5, "no MLP free-decay cell", ha="center", va="center",
                 transform=axC.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_base + ".pdf"); fig.savefig(out_base + ".png", dpi=130)
    plt.close(fig)
    return os.path.exists(out_base + ".pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-dir", required=True, help="arch-generality run dir (has arms/)")
    ap.add_argument("--out", default="reproduced", help="output dir")
    ap.add_argument("--rank-key", default="embedding.sqnorm",
                    help="metrics.jsonl key for the rank metric (any quantity that falls on compression)")
    ap.add_argument("--grok-thr", type=float, default=0.9)
    ap.add_argument("--floor-frac", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per_arm, clamp_cells, traj = analyze(args.arch_dir, args.rank_key,
                                         args.grok_thr, args.floor_frac, args.eps)
    if not per_arm:
        print("[arch] no grokked free-decay cells found under", os.path.join(args.arch_dir, "arms"))
        return
    rows = summarize(per_arm)

    print("=" * 86)
    print(f"ARCH-GENERALITY AUDIT (sec. 8)  rank_key={args.rank_key}  "
          f"grok_thr={args.grok_thr} floor_frac={args.floor_frac} eps={args.eps}")
    print("  re-derived independently from each cell's metrics.jsonl (no re-training)")
    print("=" * 86)
    print(f"{'arm':24s} {'n':>3s} {'transient(med)':>15s} {'lag/Tg(med)':>12s} {'frac-pre(med)':>14s}")
    for r in rows:
        print(f"{ARM_META.get(r['arm'],(r['arm'],))[0]:24s} {r['n']:3d} "
              f"{r['transient_med']:15.2f} {r['lag_over_tg_med']:12.2f} {r['fracpre_med']:14.2f}")
    print("-" * 86)
    print("paper reference (printed only; not used in any computation):")
    for arm in ("tf_lnfree", "mlp", "tf_ln"):
        ref = PAPER_REF[arm]
        print(f"  {ARM_META[arm][0]:24s} transient~{ref['transient']}  "
              f"lag/Tg~{ref['lag']}  frac-pre~{ref['fracpre']}")
    print("  NOTE: frac-pre and the transient are the robust, well-powered quantities.")
    print("        absolute lag/Tg is seed-sensitive; the paper's headline uses 5 seeds/cell.")
    clamp_present = any(clamp_cells.get(a) for a in clamp_cells)
    if not clamp_present:
        print("  [scope] no clamp cells in this run -> Figure 6 panels C/D need the matched-clamp arm.")
    print("=" * 86)

    # write stats csv
    csv_path = os.path.join(args.out, "arch_section8_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_cells", "transient_median", "lag_over_Tgrok_median", "fracpre_median"])
        for r in rows:
            w.writerow([r["arm"], r["n"], f"{r['transient_med']:.4f}",
                        f"{r['lag_over_tg_med']:.4f}", f"{r['fracpre_med']:.4f}"])
    print("wrote", csv_path)

    out_base = os.path.join(args.out, "fig_one_harness_three_arch")
    if render_figure(per_arm, clamp_cells, traj, args.rank_key, out_base, rows):
        print("wrote", out_base + ".pdf / .png")

    prov = dict(script="arch_generality_audit.py", version=__version__,
                arch_dir=os.path.abspath(args.arch_dir), rank_key=args.rank_key,
                grok_thr=args.grok_thr, floor_frac=args.floor_frac, eps=args.eps,
                clamp_cells_present=clamp_present, stats=rows)
    with open(os.path.join(args.out, "arch_section8.prov.json"), "w") as f:
        json.dump(prov, f, indent=2)


if __name__ == "__main__":
    main()
