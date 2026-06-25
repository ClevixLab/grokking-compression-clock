#!/usr/bin/env python3
r"""
compute_unembedding_rank.py
===========================================================================
Compute spectral-entropy effective rank of the embedding (E) and the
unembedding (U) weight matrices from logged weight checkpoints, per seed,
at each logged checkpoint step. This is the data the compression-clock
audit needs to check whether the at-grok transient appears on BOTH
representation loci (not only the embedding).

Input  : a directory of *_struct.npz files, each with keys
            E_ckpts    : (ckpts, seeds, vocab, d_model)
            U_ckpts    : (ckpts, seeds, d_model, vocab)
            ckpt_steps : (ckpts,)
            T_grok     : (seeds,)
            rho        : scalar
         and (optionally) a directory of metrics npz with test_acc/steps so
         the per-seed final accuracy can be recorded.

Output : one compact npz per cell, tfnoln_eu_rho{RHO}.npz, with keys
            rho, ckpt_steps, eff_E (ckpts,seeds), eff_U (ckpts,seeds),
            T_grok (seeds), acc_final (seeds)
         plus a printed at-grok-vs-converged summary for both loci.

The effective-rank definition matches the paper and the analyzer
(Roy & Vetterli 2007): for singular values s of a matrix,
    p_i = s_i^2 / sum_j s_j^2 ;  eff = exp(-sum_i p_i log p_i).

Usage:
  python compute_unembedding_rank.py \
      --struct_dir sample_data/transformer_noln_p59/struct \
      --metrics_dir sample_data/transformer_noln_p59/metrics_struct \
      --out_dir sample_data/transformer_noln_p59/struct_eu
"""
import os, glob, re, argparse
import numpy as np


def eff_rank(M):
    """Spectral-entropy effective rank of a 2-D matrix."""
    s = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    s2 = s ** 2
    tot = s2.sum()
    if tot <= 0:
        return np.nan
    p = s2 / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def rho_of(path):
    m = re.search(r"rho(\d+\.\d+)", path)
    return float(m.group(1)) if m else None


def acc_lookup(metrics_dir, rho):
    """Return (T_acc_steps, test_acc[seeds,T]) for this rho, or (None,None)."""
    if not metrics_dir or not os.path.isdir(metrics_dir):
        return None, None
    for f in glob.glob(os.path.join(metrics_dir, "*.npz")):
        rf = rho_of(f)
        if rf is not None and abs(rf - rho) < 1e-3:
            z = np.load(f, allow_pickle=True)
            acc = z["test_acc"] if "test_acc" in z.files else z.get("acc")
            steps = z["steps"]
            return np.asarray(steps, float), np.asarray(acc, float)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct_dir", required=True)
    ap.add_argument("--metrics_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grok_thr", type=float, default=0.90)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.struct_dir, "*_struct.npz")), key=lambda f: (rho_of(f) or 0))
    if not files:
        print("[error] no *_struct.npz in", args.struct_dir)
        return

    print("=" * 96)
    print("EMBEDDING vs UNEMBEDDING effective rank (from weight checkpoints)")
    print(f"grok_thr={args.grok_thr}; at-grok = nearest ckpt to per-seed T_grok; converged = last ckpt")
    print("=" * 96)
    hdr = (f"{'rho':>5} {'seeds':>6} {'acc_fin':>8} | {'E@grok':>7} {'E_conv':>7} {'E drop':>7} {'E tr?':>6} "
           f"| {'U@grok':>7} {'U_conv':>7} {'U drop':>7} {'U tr?':>6}")
    print(hdr); print("-" * 96)

    for f in files:
        z = np.load(f, allow_pickle=True)
        rho = float(z["rho"])
        E = z["E_ckpts"].astype(np.float64)   # (C,S,V,d)
        U = z["U_ckpts"].astype(np.float64)   # (C,S,d,V)
        ck = z["ckpt_steps"].astype(float)    # (C,)
        Tg = z["T_grok"].astype(int)          # (S,)
        C, S = E.shape[0], E.shape[1]

        effE = np.array([[eff_rank(E[c, s]) for s in range(S)] for c in range(C)])  # (C,S)
        effU = np.array([[eff_rank(U[c, s]) for s in range(S)] for c in range(C)])  # (C,S)

        acc_steps, acc = acc_lookup(args.metrics_dir, rho)
        acc_final = np.full(S, np.nan)
        if acc is not None:
            acc_final = acc[:, -1]

        # per-seed at-grok (nearest ckpt to T_grok if seed grokked) vs converged (last ckpt)
        Eg, Ec, Ug, Uc, ng = [], [], [], [], 0
        for s in range(S):
            grokked = (acc_final[s] >= args.grok_thr) if np.isfinite(acc_final[s]) else (Tg[s] > 0)
            if not grokked:
                continue
            ng += 1
            cj = int(np.argmin(np.abs(ck - Tg[s]))) if Tg[s] > 0 else 0
            Eg.append(effE[cj, s]); Ec.append(effE[-1, s])
            Ug.append(effU[cj, s]); Uc.append(effU[-1, s])

        def med(x):
            return float(np.median(x)) if x else np.nan
        Egm, Ecm, Ugm, Ucm = med(Eg), med(Ec), med(Ug), med(Uc)
        Ed = (Egm - Ecm) / Egm if Egm else np.nan
        Ud = (Ugm - Ucm) / Ugm if Ugm else np.nan
        Etr = "YES" if (np.isfinite(Ed) and Ed >= 0.25) else "no"
        Utr = "YES" if (np.isfinite(Ud) and Ud >= 0.25) else "no"
        afm = float(np.nanmedian(acc_final)) if np.isfinite(acc_final).any() else np.nan
        print(f"{rho:>5.2f} {ng:>6} {afm:>8.3f} | {Egm:>7.2f} {Ecm:>7.2f} {100*Ed:>6.1f}% {Etr:>6} "
              f"| {Ugm:>7.2f} {Ucm:>7.2f} {100*Ud:>6.1f}% {Utr:>6}")

        out = os.path.join(args.out_dir, f"tfnoln_eu_rho{rho:.2f}.npz")
        np.savez(out, rho=rho, ckpt_steps=ck, eff_E=effE, eff_U=effU,
                 T_grok=Tg, acc_final=acc_final)
    print("=" * 96)
    print(f"wrote per-cell E/U npz to {args.out_dir}")
    print("Reading: 'tr?' = transient (drop >= 25% from at-grok to converged). "
          "The 'both loci' claim holds on a cell only if BOTH E and U are YES and it generalizes.")


if __name__ == "__main__":
    main()
