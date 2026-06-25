#!/usr/bin/env python3
r"""
analyze_scale_invariance.py   (paper section 9: pre-registered scale-invariance control)
=====================================================================
Evaluates the PRE-REGISTERED scale-invariance control and renders its diagnostic figure.
The hypothesis under test (frozen before the run, see
specs/PREREGISTRATION_scale_inv.json): embedding scale invariance is what
defers compression in the LayerNorm transformer. The control arm `tf_rms`
(RMS-normalization on the embedding input only) shares LayerNorm's scale
invariance and nothing else.

Three predictions, with the acceptance rule "accept iff PD1 AND PD2":
  PD1 (deferral)    : concentration completed by grok  < PD1_THR (0.40)
  PD2 (norm-driven) : post-grok Spearman(|W|, rank)    < PD2_THR (-0.30)
  PD3 (Fourier)     : top-k embedding DFT freqs carry  > PD3_THR (0.50) of power

"Concentration completed by grok" == frac-pre
== (rank_init - rank_grok) / (rank_init - rank_floor), the same quantity
arch_generality_audit.py reports. PD2 is computed from each cell's logged
|W| (global_norm) and rank trajectory over post-grok steps. PD3 needs the
converged embedding weights (final checkpoint) and is computed only if torch
and a checkpoint are available; otherwise it is skipped, not fabricated.

INPUT  (a run directory that INCLUDES a tf_rms control arm):
    <run>/arms/<arm>/<cell>/metrics.jsonl
    <run>/arms/<arm>/<cell>/ckpt.pt          # optional, for PD3 (needs torch)
  arms expected for the full figure:
    tf_lnfree (canonical), mlp, tf_rms (control), tf_ln (LayerNorm)

OUTPUT:
    <out>/fig_scale_invariance_control.{pdf,png}   # Figure 7
    <out>/scale_invariance_PD.csv                  # the PD table + verdict
    <out>/scale_invariance.prov.json

USAGE:
    python audit/analyze_scale_invariance.py --arch-dir <run> --out reproduced/
"""
import argparse, glob, json, os, re, csv
import numpy as np

__version__ = "1.0.0"

ARM_ORDER = ["tf_lnfree", "mlp", "tf_rms", "tf_ln"]
ARM_LABEL = {"tf_lnfree": "TF canonical", "mlp": "MLP",
             "tf_rms": "TF RMSNorm-emb (control)", "tf_ln": "TF LayerNorm"}
ARM_COLOR = {"tf_lnfree": "tab:green", "mlp": "tab:blue",
             "tf_rms": "tab:purple", "tf_ln": "tab:red"}

DEFAULT_THR = dict(PD1_THR=0.40, PD2_THR=-0.30, PD3_THR=0.50, top_k=5)


def load_cell(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
            for k in rows[0].keys() if k != "op"}


def cell_dirs(arm_dir):
    for d in sorted(glob.glob(os.path.join(arm_dir, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metrics.jsonl")):
            yield d


def per_cell_pd(m, rank_key, grok_thr, floor_frac):
    """Return (fracpre, post_grok_spearman_norm_rank) for one cell, or None."""
    from scipy.stats import spearmanr
    acc, rank, norm, step = m.get("test_acc"), m.get(rank_key), m.get("global_norm"), m["step"]
    if acc is None or rank is None or not np.isfinite(rank).any():
        return None
    gi = np.where(acc >= grok_thr)[0]
    if len(gi) == 0:
        return None
    gi = gi[0]
    nfloor = max(1, int(round(floor_frac * len(step))))
    floor = float(np.mean(rank[-nfloor:]))
    r_init, r_grok = rank[0], rank[gi]
    denom = r_init - floor
    fracpre = (r_init - r_grok) / denom if denom != 0 else np.nan
    sp = np.nan
    if norm is not None and len(step) - gi >= 3:
        post = slice(gi, len(step))
        if np.isfinite(norm[post]).all() and np.ptp(norm[post]) > 0:
            sp = spearmanr(norm[post], rank[post]).correlation
    return fracpre, sp


def embedding_dft_power(ckpt_path, top_k):
    """Top-k normalized embedding DFT power, or None if torch/ckpt unavailable."""
    try:
        import torch
    except Exception:
        return None
    if not os.path.exists(ckpt_path):
        return None
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        emb = None
        for k, v in state.items():
            if "embed" in k.lower() and hasattr(v, "ndim") and v.ndim == 2:
                emb = v.detach().cpu().numpy()
                break
        if emb is None:
            return None
        # power per Fourier frequency, summed over embedding dims, normalized
        F = np.abs(np.fft.rfft(emb, axis=0)) ** 2
        p = F.sum(axis=1)
        p = p / p.sum()
        return float(np.sort(p)[::-1][:top_k].sum())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch-dir", required=True)
    ap.add_argument("--out", default="reproduced")
    ap.add_argument("--rank-key", default="embedding.sqnorm")
    ap.add_argument("--grok-thr", type=float, default=0.9)
    ap.add_argument("--floor-frac", type=float, default=0.1)
    ap.add_argument("--prereg", default=None, help="optional PREREGISTRATION_scale_inv.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    thr = dict(DEFAULT_THR)
    if args.prereg and os.path.exists(args.prereg):
        pj = json.load(open(args.prereg))
        thr.update({k: pj["thresholds"][k] for k in DEFAULT_THR if k in pj.get("thresholds", {})})

    arms_root = os.path.join(args.arch_dir, "arms")
    present = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(arms_root, "*")))
               if os.path.isdir(d)]

    # aggregate PD quantities per arm
    table = {}
    traj = {}
    for arm in present:
        ad = os.path.join(arms_root, arm)
        fps, sps, dfts = [], [], []
        for d in cell_dirs(ad):
            m = load_cell(os.path.join(d, "metrics.jsonl"))
            traj.setdefault(arm, []).append(m)
            r = per_cell_pd(m, args.rank_key, args.grok_thr, args.floor_frac)
            if r is None:
                continue
            fps.append(r[0]); sps.append(r[1])
            dp = embedding_dft_power(os.path.join(d, "ckpt.pt"), thr["top_k"])
            if dp is not None:
                dfts.append(dp)
        if fps:
            table[arm] = dict(
                fracpre=float(np.nanmedian(fps)),
                post_sp=float(np.nanmedian(sps)) if np.isfinite(sps).any() else np.nan,
                dft=float(np.nanmedian(dfts)) if dfts else np.nan,
                n=len(fps))

    print("=" * 84)
    print("SCALE-INVARIANCE CONTROL (sec. 9)  pre-registered: accept iff PD1 AND PD2")
    print(f"  PD1 deferral  : frac-pre < {thr['PD1_THR']}")
    print(f"  PD2 norm-drv  : post-grok Spearman(|W|,rank) < {thr['PD2_THR']}")
    print(f"  PD3 Fourier   : top-{thr['top_k']} embedding DFT power > {thr['PD3_THR']}")
    print("=" * 84)
    print(f"{'arm':28s} {'n':>3s} {'frac-pre':>9s} {'postSp(|W|,r)':>13s} {'DFT top-k':>10s}")
    for arm in ARM_ORDER:
        if arm in table:
            t = table[arm]
            dft = f"{t['dft']:.2f}" if np.isfinite(t['dft']) else "n/a"
            print(f"{ARM_LABEL[arm]:28s} {t['n']:3d} {t['fracpre']:9.2f} {t['post_sp']:13.2f} {dft:>10s}")
    for arm in present:
        if arm not in ARM_ORDER and arm in table:
            t = table[arm]
            print(f"{arm:28s} {t['n']:3d} {t['fracpre']:9.2f} {t['post_sp']:13.2f}")
    print("-" * 84)

    # pre-registered verdict on the control arm
    if "tf_rms" in table:
        t = table["tf_rms"]
        pd1 = t["fracpre"] < thr["PD1_THR"]
        pd2 = (np.isfinite(t["post_sp"]) and t["post_sp"] < thr["PD2_THR"])
        pd3 = (np.isfinite(t["dft"]) and t["dft"] > thr["PD3_THR"])
        accept = pd1 and pd2
        print(f"  control tf_rms: PD1={'PASS' if pd1 else 'FAIL'} (frac-pre={t['fracpre']:.2f}), "
              f"PD2={'PASS' if pd2 else 'FAIL'} (postSp={t['post_sp']:.2f}), "
              f"PD3={'PASS' if pd3 else 'FAIL' if np.isfinite(t['dft']) else 'N/A'}")
        print(f"  PRE-REGISTERED VERDICT: scale-invariance account "
              f"{'ACCEPTED' if accept else 'REJECTED'} (accept iff PD1 and PD2)")
        verdict = "ACCEPTED" if accept else "REJECTED"
    else:
        print("  [scope] control arm 'tf_rms' NOT in this run.")
        print("          PD1/PD2 cannot be evaluated -> section 9 verdict is not reproducible yet.")
        print("          Run the scale_inv control (RMSNorm-embedding-only TF) into this run dir.")
        verdict = "CONTROL_ARM_MISSING"
    print("=" * 84)

    # write PD csv
    pd_path = os.path.join(args.out, "scale_invariance_PD.csv")
    with open(pd_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_cells", "fracpre_median", "postgrok_spearman_norm_rank",
                    "dft_topk_power", "verdict"])
        for arm in present:
            if arm in table:
                t = table[arm]
                w.writerow([arm, t["n"], f"{t['fracpre']:.4f}",
                            f"{t['post_sp']:.4f}" if np.isfinite(t["post_sp"]) else "",
                            f"{t['dft']:.4f}" if np.isfinite(t["dft"]) else "",
                            verdict if arm == "tf_rms" else ""])
    print("wrote", pd_path)

    # ---- Figure 7
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("Pre-registered control: does scale invariance defer compression? (sec. 9)",
                 fontsize=12)

    # (A) concentration done by grok (frac-pre) per arm
    arms_plot = [a for a in ARM_ORDER if a in table]
    vals = [table[a]["fracpre"] for a in arms_plot]
    axA.bar(range(len(arms_plot)), vals, color=[ARM_COLOR[a] for a in arms_plot])
    axA.axhline(thr["PD1_THR"], ls="--", color="k", lw=1)
    axA.text(len(arms_plot) - 0.5, thr["PD1_THR"] + 0.02, f"PD1 thr {thr['PD1_THR']}",
             ha="right", fontsize=8)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axA.set_xticks(range(len(arms_plot)))
    axA.set_xticklabels([ARM_LABEL[a].replace(" ", "\n") for a in arms_plot], fontsize=7)
    axA.set_ylabel("concentration done BY grok (frac-pre)")
    axA.set_title("(A) Scale invariance does NOT defer\n(control groups with canonical?)")

    # (B) |W| global-norm trajectories per arm (representative cell)
    axB.set_title("(B) Is the norm explosion LayerNorm-specific?")
    for arm in arms_plot:
        ms = traj.get(arm, [])
        if ms:
            m = ms[0]
            axB.plot(m["step"], m["global_norm"], color=ARM_COLOR[arm],
                     lw=1.3, label=ARM_LABEL[arm])
    axB.set_xlabel("step"); axB.set_ylabel("|W| (global norm)"); axB.legend(fontsize=7)

    # (C) embedding DFT power for the control (needs torch + ckpt)
    axC.set_title("(C) What it compresses ONTO: Fourier pairs")
    drew_c = False
    if "tf_rms" in traj:
        ck = os.path.join(args.arch_dir, "arms", "tf_rms",
                          os.path.basename(glob.glob(os.path.join(arms_root, "tf_rms", "*"))[0]),
                          "ckpt.pt") if glob.glob(os.path.join(arms_root, "tf_rms", "*")) else ""
        try:
            import torch  # noqa
            if ck and os.path.exists(ck):
                sd = torch.load(ck, map_location="cpu", weights_only=False)
                state = sd.get("model", sd) if isinstance(sd, dict) else sd
                emb = next((v.detach().cpu().numpy() for k, v in state.items()
                            if "embed" in k.lower() and getattr(v, "ndim", 0) == 2), None)
                if emb is not None:
                    p = np.abs(np.fft.rfft(emb, axis=0)) ** 2
                    p = p.sum(axis=1); p = p / p.sum()
                    axC.stem(np.arange(len(p)), p)
                    axC.set_xlabel("Fourier frequency k")
                    axC.set_ylabel("embedding DFT power (norm.)")
                    drew_c = True
        except Exception:
            pass
    if not drew_c:
        axC.text(0.5, 0.5, "PD3 panel needs torch + the\ntf_rms converged checkpoint",
                 ha="center", va="center", transform=axC.transAxes, color="0.35", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_base = os.path.join(args.out, "fig_scale_invariance_control")
    fig.savefig(out_base + ".pdf"); fig.savefig(out_base + ".png", dpi=130)
    plt.close(fig)
    print("wrote", out_base + ".pdf / .png")

    prov = dict(script="analyze_scale_invariance.py", version=__version__,
                arch_dir=os.path.abspath(args.arch_dir), thresholds=thr,
                control_arm_present="tf_rms" in table, verdict=verdict, table=table)
    with open(os.path.join(args.out, "scale_invariance.prov.json"), "w") as f:
        json.dump(prov, f, indent=2, default=float)


if __name__ == "__main__":
    main()
