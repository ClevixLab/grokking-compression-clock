#!/usr/bin/env python3
r"""
sensitivity_floor_ci.py
===========================================================================
Two robustness analyses for the compression-clock audit, on the full MLP grid:
  (1) SENSITIVITY: vary each audit parameter (eps, floor_frac, grok_thr,
      min_drop, boundary drop-threshold) one at a time and report how the
      lag, the rho-orderings, the converged-floor law, and the verdict move.
  (2) FLOOR-LAW BOOTSTRAP CI: resample seeds with replacement; report a CI on
      Spearman(rho, converged floor) and on each per-cell floor.
Clock definitions match analyze_compression_clock_v1_5.py exactly.
Outputs: <out>/sensitivity_table.csv, <out>/sensitivity_table.tex, <out>/floor_ci.csv
Usage:  python sensitivity_floor_ci.py --data sample_data --out_dir results
"""
import os, glob, csv, argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau

DEFAULT = dict(eps=0.10, floor_frac=0.10, min_drop=1.0, grok_thr=0.90, drop_thr=0.25)

def load_task(folder):
    cells = []
    for f in sorted(glob.glob(f"{folder}/metrics/*.npz")):
        z = np.load(f, allow_pickle=True)
        cells.append(dict(rho=float(z["rho"]), steps=z["steps"].astype(float),
                          eff=z["eff"].astype(float), acc=z["acc"].astype(float)))
    return sorted(cells, key=lambda c: c["rho"])

def per_seed(cell, eps, floor_frac, min_drop, grok_thr):
    steps, eff, acc = cell["steps"], cell["eff"], cell["acc"]
    S, T = eff.shape; nf = max(1, int(floor_frac * T)); rows = []
    for s in range(S):
        e, a = eff[s], acc[s]; good = np.where(a >= grok_thr)[0]
        if len(good) == 0: continue
        gi = good[0]; Tg = steps[gi]; eg = e[gi]
        floor = float(np.median(e[-nf:])); drop = eg - floor
        if not (np.isfinite(drop) and drop >= min_drop):
            rows.append(dict(Tg=Tg, eg=eg, floor=floor, drop=drop, Tc=np.nan, gap=np.nan, valid=False)); continue
        thr = floor + eps * drop; post = np.where(steps >= Tg)[0]
        ci = next((i for i in post if e[i] <= thr), None)
        if ci is None: ci = post[-1]
        rows.append(dict(Tg=Tg, eg=eg, floor=floor, drop=drop, Tc=steps[ci], gap=steps[ci]-Tg, valid=True))
    return rows

def cell_agg(cell, **kw):
    rows = per_seed(cell, kw["eps"], kw["floor_frac"], kw["min_drop"], kw["grok_thr"])
    if not rows: return None
    val = [r for r in rows if r["valid"]]
    med = lambda k, src: (float(np.median([r[k] for r in src if np.isfinite(r[k])]))
                          if any(np.isfinite(r[k]) for r in src) else np.nan)
    eg, floor, drop = med("eg", rows), med("floor", rows), med("drop", rows)
    return dict(rho=cell["rho"], Tg=med("Tg", rows), eg=eg, floor=floor, drop=drop,
                Tc=med("Tc", val), gap=med("gap", val), dropratio=(drop/eg if eg else np.nan))

def audit(cells, **kw):
    p = dict(DEFAULT); p.update(kw)
    rows = [cell_agg(c, **p) for c in cells]; rows = [r for r in rows if r is not None]
    floors = [r["floor"] for r in rows if np.isfinite(r["floor"])]; fmed = np.median(floors) if floors else np.nan
    for r in rows:
        r["boundary"] = bool((np.isfinite(r["dropratio"]) and r["dropratio"] < p["drop_thr"]) or
                             (np.isfinite(r["floor"]) and np.isfinite(fmed) and r["floor"] > 3*fmed))
    nb = [r for r in rows if not r["boundary"] and np.isfinite(r["gap"]) and np.isfinite(r["Tc"])]
    gated = [r["rho"] for r in rows if r["boundary"]]
    if len(nb) < 2:
        return dict(verdict="LOW-POWER", gated=gated, lag=np.nan, rel=np.nan, sp_tg=np.nan, sp_tc=np.nan, sp_fl=np.nan, n_nb=len(nb))
    rho = np.array([r["rho"] for r in nb]); gap = np.array([r["gap"] for r in nb])
    tg = np.array([r["Tg"] for r in nb]); tc = np.array([r["Tc"] for r in nb]); fl = np.array([r["floor"] for r in nb])
    rel = float(np.median(gap/tg)); sp_tg, sp_tc, sp_fl = (spearmanr(rho, x).correlation for x in (tg, tc, fl))
    big = rel > 0.50; opp = np.isfinite(sp_tg) and np.isfinite(sp_tc) and sp_tg*sp_tc < 0
    amd = nb[int(np.argmin(tg))]["rho"] != nb[int(np.argmin(tc))]["rho"]
    v = ("TWO CLOCKS" if big and opp else "PARTIALLY SEP + LAG" if big and amd else "ONE CLOCK + LAG" if big else "SINGLE/SMALL LAG")
    return dict(verdict=v, gated=gated, lag=float(np.median(gap)), rel=rel, sp_tg=float(sp_tg), sp_tc=float(sp_tc), sp_fl=float(sp_fl), n_nb=len(nb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data"); ap.add_argument("--out_dir", default="results")
    ap.add_argument("--bootstrap", type=int, default=10000); args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True); rng = np.random.default_rng(0)
    DATA = {"add": load_task(f"{args.data}/mlp_modadd_p59"), "mult": load_task(f"{args.data}/mlp_modmult_p59")}
    sweeps = {"eps": [0.05, 0.10, 0.20], "floor_frac": [0.05, 0.10, 0.20], "grok_thr": [0.90, 0.95, 0.99],
              "min_drop": [0.5, 1.0, 2.0], "drop_thr": [0.20, 0.25, 0.30]}
    label = {"eps": r"$\varepsilon$", "floor_frac": "floor frac", "grok_thr": "grok thr", "min_drop": "min drop", "drop_thr": "gate thr"}
    print("="*104); print("SENSITIVITY — full grid (add 9, mult 7), one parameter varied; defaults:", DEFAULT); print("="*104)
    hdr = f"{'param':>16} {'task':>5} {'gated':>10} {'lag':>7} {'lag/Tg':>7} {'sp(Tg)':>7} {'sp(Tc)':>7} {'sp(fl)':>7} {'verdict':>20}"
    out_rows = []
    for pn, vals in sweeps.items():
        print("-"*104); print(hdr); print("-"*104)
        for v in vals:
            for task in ["add", "mult"]:
                r = audit(DATA[task], **{pn: v}); g = ",".join(f"{x:.2f}" for x in r["gated"]) if r["gated"] else "—"
                print(f"{pn+'='+str(v):>16} {task:>5} {g:>10} {r['lag']:>7.0f} {r['rel']:>7.2f} {r['sp_tg']:>+7.2f} {r['sp_tc']:>+7.2f} {r['sp_fl']:>+7.2f} {r['verdict']:>20}")
                out_rows.append(dict(param=pn, value=v, task=task, **r))
    print("="*104)
    with open(f"{args.out_dir}/sensitivity_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["param","value","task","gated","lag","rel","sp_tg","sp_tc","sp_fl","verdict","n_nb"]); w.writeheader()
        for r in out_rows:
            rr = dict(r); rr["gated"] = ";".join(f"{x:.2f}" for x in r["gated"]); w.writerow(rr)
    with open(f"{args.out_dir}/sensitivity_table.tex", "w") as fh:
        fh.write("% auto-generated by sensitivity_floor_ci.py\n\\begin{tabular}{llcccccc}\n\\toprule\n")
        fh.write("parameter & task & gated & lag & lag/$T_g$ & $\\rho_S(T_g)$ & $\\rho_S(T_c)$ & $\\rho_S$(floor) \\\\\n\\midrule\n")
        items = list(sweeps.items())
        for j, (pn, vals) in enumerate(items):
            for v in vals:
                for task in ["add", "mult"]:
                    r = audit(DATA[task], **{pn: v}); g = ",".join(f"{x:.2f}" for x in r["gated"]) if r["gated"] else "---"
                    fh.write(f"{label[pn]}$={v}$ & {task} & {g} & {r['lag']:.0f} & {r['rel']:.2f} & ${r['sp_tg']:+.2f}$ & ${r['sp_tc']:+.2f}$ & ${r['sp_fl']:+.2f}$ \\\\\n")
            if j < len(items) - 1:
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {args.out_dir}/sensitivity_table.csv and .tex")
    print("\n" + "="*104); print(f"FLOOR LAW — bootstrap CI (resample seeds, B={args.bootstrap})"); print("="*104)
    floor_csv = [["task","rho","floor","ci_lo","ci_hi"]]
    for task in ["add", "mult"]:
        aud = audit(DATA[task]); gated = set(np.round(aud["gated"], 2))
        nb = [c for c in DATA[task] if round(c["rho"], 2) not in gated]; rhos = [c["rho"] for c in nb]
        pf = [np.median(c["eff"][:, -max(1, int(0.10*c["eff"].shape[1])):], axis=1) for c in nb]
        S = pf[0].shape[0]; pt = [float(np.median(x)) for x in pf]
        sp_pt = spearmanr(rhos, pt).correlation; kt = kendalltau(rhos, pt).correlation; sp_p = spearmanr(rhos, pt).pvalue
        B = args.bootstrap; sp_b = np.empty(B); fl_b = np.empty((B, len(nb)))
        for b in range(B):
            idx = rng.integers(0, S, S); v = [float(np.median(x[idx])) for x in pf]; fl_b[b] = v; sp_b[b] = spearmanr(rhos, v).correlation
        lo, hi = np.nanpercentile(sp_b, [2.5, 97.5]); pneg = np.mean(sp_b <= -0.999)
        print(f"\n[{task}] non-boundary n={len(nb)}: Spearman(rho,floor)={sp_pt:+.3f} (p={sp_p:.2g}), Kendall tau={kt:+.3f}")
        print(f"  bootstrap Spearman 95% CI [{lo:+.3f}, {hi:+.3f}]; P(=-1.0)={pneg:.3f}")
        for i, c in enumerate(nb):
            clo, chi = np.nanpercentile(fl_b[:, i], [2.5, 97.5])
            print(f"    rho={c['rho']:.2f}: floor={pt[i]:6.3f}  95% CI [{clo:6.3f}, {chi:6.3f}]")
            floor_csv.append([task, f"{c['rho']:.2f}", f"{pt[i]:.4f}", f"{clo:.4f}", f"{chi:.4f}"])
    print("="*104)
    with open(f"{args.out_dir}/floor_ci.csv", "w", newline="") as fh: csv.writer(fh).writerows(floor_csv)
    print(f"wrote {args.out_dir}/floor_ci.csv")

if __name__ == "__main__":
    main()
