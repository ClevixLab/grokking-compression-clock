#!/usr/bin/env python3
r"""
make_dashboard_figure_v1.py
====================================================================
Builds the compression-clock dashboard figure from REAL longtrain_*rho*.npz data.
No numbers are hard-coded; everything is read from the npz trajectories you pass in.

Panels:
  A  eff_rank(step) trajectories per rho, with T_grok and T_compress marked
  B  at-grok vs final eff_rank scatter (the "at-grok is not converged" panel)
  C  compression-clock bars: T_grok and T_compress per rho (the lag)
  D  boundary-gate demo: drop/eff_grok per rho, threshold line (cells below = boundary)

Usage (real data):
  python make_dashboard_figure_v1.py --dir /path/to/your_run_dir --task "mod-mult" --out figures/dashboard_modmult.pdf
  python make_dashboard_figure_v1.py --dir /path/to/your_run_dir --task "mod-add"  --out figures/dashboard_modadd.pdf
"""
import argparse, glob, os, sys
import numpy as np
__version__ = "1.0.0"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 0.10
FLOOR_FRAC = 0.10
GROK_THR = 0.90

def eff_floor(eff, frac=FLOOR_FRAC):
    n = max(1, int(frac * eff.shape[-1]))
    return np.median(eff[..., -n:], axis=-1)

def _pick(z, *names):
    """Return the first present key among names (handles eff vs eff_rank_E, acc vs test_acc)."""
    for n in names:
        if n in z.files:
            return np.asarray(z[n], float)
    raise KeyError(f"none of {names} found in npz; keys present: {list(z.files)}")

def per_cell(z):
    steps = np.asarray(z["steps"], float)
    eff = _pick(z, "eff", "eff_rank_E", "effrank", "eff_rank")   # (seeds, T)
    acc = _pick(z, "acc", "test_acc", "test_accuracy")
    rho = float(z["rho"])
    em, am = np.median(eff, axis=0), np.median(acc, axis=0)
    gi = int(np.argmax(am >= GROK_THR)) if (am >= GROK_THR).any() else -1
    Tg = steps[gi] if gi >= 0 else np.nan
    eff_g = em[gi] if gi >= 0 else np.nan
    floor = np.median(em[-max(1, int(FLOOR_FRAC*len(em))):])
    thr = floor + EPS * (eff_g - floor)
    ci = None
    if gi >= 0:
        for i in range(gi, len(steps)):
            if em[i] <= thr:
                ci = i; break
    Tc = steps[ci] if ci is not None else np.nan
    drop = eff_g - floor
    return dict(rho=rho, steps=steps, em=em, am=am, Tg=Tg, Tc=Tc,
                eff_g=eff_g, floor=floor, drop=drop,
                drop_ratio=(drop/eff_g if eff_g else np.nan))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--task", default="task")
    ap.add_argument("--out", default="dashboard.pdf")
    ap.add_argument("--boundary_ratio", type=float, default=0.25)
    ap.add_argument("--glob", default=None,
                    help="filename glob inside <dir>/metrics (default: try longtrain_*rho*.npz then *rho*.npz then *.npz)")
    args = ap.parse_args()

    mdir = os.path.join(args.dir, "metrics")
    if not os.path.isdir(mdir):
        mdir = args.dir  # allow pointing straight at a folder of npz
    patterns = [args.glob] if args.glob else ["longtrain_*rho*.npz", "*rho*.npz", "*.npz"]
    files = []
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(mdir, pat)))
        if files:
            print(f"[info] matched {len(files)} files with pattern '{pat}' in {mdir}")
            break
    if not files:
        print(f"[error] no npz found in {mdir} (tried {patterns}).")
        print("        Point --dir at the run folder (it should contain a 'metrics' subfolder),")
        print("        or pass --glob '<pattern>'. Each npz needs keys: rho, steps, eff(or eff_rank_E), acc(or test_acc).")
        sys.exit(1)
    cells = [per_cell(np.load(f, allow_pickle=True)) for f in files]
    cells.sort(key=lambda c: c["rho"])
    rhos = [c["rho"] for c in cells]

    plt.rcParams.update({"font.size": 9, "axes.spineright": False} if False else {"font.size": 9})
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.6))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(cells)))

    # Panel A: trajectories
    a = ax[0, 0]
    for c, col in zip(cells, cmap):
        a.plot(c["steps"], c["em"], color=col, lw=1.4, label=f"ρ={c['rho']:.2f}")
        if np.isfinite(c["Tg"]): a.axvline(c["Tg"], color=col, ls=":", lw=0.7, alpha=0.5)
    a.set_xscale("log"); a.set_xlabel("step"); a.set_ylabel("effective rank (E)")
    a.set_title("A  eff_rank trajectories (dotted = $T_{grok}$)", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False)

    # Panel B: at-grok vs final
    b = ax[0, 1]
    eg = [c["eff_g"] for c in cells]; ef = [c["floor"] for c in cells]
    b.scatter(rhos, eg, c="#d1495b", label="at grok", zorder=3)
    b.scatter(rhos, ef, c="#30638e", label="at convergence", zorder=3)
    for c in cells:
        b.plot([c["rho"], c["rho"]], [c["floor"], c["eff_g"]], color="#bbb", lw=1, zorder=1)
    b.set_xlabel("norm budget ρ"); b.set_ylabel("effective rank (E)")
    b.set_title("B  at-grok is not converged", fontsize=9, loc="left")
    b.legend(fontsize=7, frameon=False)

    # Panel C: clock bars
    c_ax = ax[1, 0]
    x = np.arange(len(cells)); w = 0.38
    Tg = [c["Tg"] for c in cells]; Tc = [c["Tc"] for c in cells]
    c_ax.bar(x - w/2, Tg, w, label="$T_{grok}$", color="#edae49")
    c_ax.bar(x + w/2, Tc, w, label="$T_{compress}$", color="#00798c")
    c_ax.set_xticks(x); c_ax.set_xticklabels([f"{r:.2f}" for r in rhos])
    c_ax.set_xlabel("norm budget ρ"); c_ax.set_ylabel("step")
    c_ax.set_title("C  compression lags grokking", fontsize=9, loc="left")
    c_ax.legend(fontsize=7, frameon=False)

    # Panel D: boundary gate
    d = ax[1, 1]
    dr = [c["drop_ratio"] for c in cells]
    colors = ["#d1495b" if (np.isfinite(r) and r < args.boundary_ratio) else "#2e933c" for r in dr]
    d.bar(x, dr, color=colors)
    d.axhline(args.boundary_ratio, color="k", ls="--", lw=0.8)
    d.text(0.02, args.boundary_ratio+0.02, f"boundary threshold ({args.boundary_ratio})", fontsize=7, transform=d.get_yaxis_transform())
    d.set_xticks(x); d.set_xticklabels([f"{r:.2f}" for r in rhos])
    d.set_xlabel("norm budget ρ"); d.set_ylabel("drop / eff_grok")
    d.set_title("D  boundary gate (red = excluded)", fontsize=9, loc="left")

    fig.suptitle(f"Compression-clock audit — {args.task}", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight"); fig.savefig(args.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", args.out)

if __name__ == "__main__":
    main()
