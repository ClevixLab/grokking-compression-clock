r"""
analyze_compression_clock_generic_v1.py   (v1.0.0)
====================================================================
Run the compression-clock audit on THIRD-PARTY grokking runs, not just our own
npz convention. The goal is that anyone with saved checkpoints OR a logged
metrics table can audit whether their at-grok reading of effective rank is a
transient, with no need to re-train.

Two input modes (pick one):

  MODE A  --checkpoints  : a directory of weight checkpoints, one per logged step.
      Each checkpoint is a .npz/.npy/.pt-exported-.npz holding the matrix to audit
      (default key: "embedding"; override with --matrix-key). Filenames must encode
      the step, e.g. ckpt_step1000.npz, step_002000.npy. We compute the
      spectral-entropy effective rank of that matrix at each step and pair it with
      an accuracy log (--acc-csv: columns step,test_acc) to form one cell.

  MODE B  --metrics-csv  : a CSV that already has the trajectory, with columns
      step, test_acc, eff_rank   (column names overridable). One file = one cell.
      Use --rho to tag the norm-budget of the cell; pass several CSVs with
      --metrics-csv a.csv:1.0 b.csv:1.1 ...  to build a dose-response.

Both modes write a normalized longtrain_*rho*.npz into an output dir and then call
the canonical analyzer (analyze_compression_clock_v1_5.py) on it, so the verdict,
boundary gate, censoring flag, and ordering guard are identical to the paper's.

Examples:
  # Audit someone's checkpoints (embedding matrix), one budget:
  python analyze_compression_clock_generic_v1.py \
     --checkpoints ./their_run/ckpts --matrix-key W_E \
     --acc-csv ./their_run/acc.csv --rho 1.0 \
     --out ./audited --clock-script ../audit/analyze_compression_clock_v1_5.py

  # Audit a logged metrics table with several budgets:
  python analyze_compression_clock_generic_v1.py \
     --metrics-csv add_wd1.csv:1.0 add_wd2.csv:1.2 \
     --eff-col effective_rank --acc-col test_acc --step-col step \
     --out ./audited --clock-script ../audit/analyze_compression_clock_v1_5.py

This adapter intentionally does no training and fabricates no data: it only
re-expresses an existing run in the analyzer's input schema. If a run is too
short to reach the floor, the analyzer will (correctly) report censoring.
"""
__version__ = "1.0.0"
import numpy as np, os, sys, re, glob, argparse, importlib.util, csv

def eff_rank_from_matrix(W):
    """Spectral-entropy effective rank of a 2-D matrix (same definition as the paper)."""
    W = np.asarray(W, float)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    s = np.linalg.svd(W, compute_uv=False)
    s2 = s ** 2; t = s2.sum()
    if t <= 0:
        return 0.0
    p = s2 / t; nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))

def _step_from_name(fn):
    m = re.findall(r"(\d+)", os.path.basename(fn))
    return int(m[-1]) if m else None

def _load_matrix(path, key):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    z = np.load(path, allow_pickle=True)
    if key in z:
        return z[key]
    # fall back to the only / largest 2-D array present
    cands = [(k, np.asarray(z[k])) for k in z.files]
    twod = [(k, a) for k, a in cands if a.ndim == 2]
    if not twod:
        raise SystemExit(f"[error] no 2-D matrix in {path}; keys={list(z.files)}. Use --matrix-key.")
    twod.sort(key=lambda kv: -kv[1].size)
    return twod[0][1]

def _read_csv(path, step_col, acc_col, eff_col=None):
    steps, accs, effs = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = {c.lower(): c for c in (r.fieldnames or [])}
        def col(name):
            if name in (r.fieldnames or []):
                return name
            if name.lower() in cols:
                return cols[name.lower()]
            raise SystemExit(f"[error] column '{name}' not in {path}; has {r.fieldnames}")
        sc, ac = col(step_col), col(acc_col)
        ec = col(eff_col) if eff_col else None
        for row in r:
            steps.append(float(row[sc])); accs.append(float(row[ac]))
            if ec:
                effs.append(float(row[ec]))
    return np.array(steps), np.array(accs), (np.array(effs) if eff_col else None)

def write_cell(out, rho, steps, eff, acc):
    """eff/acc are 1-D here (one seed); store as (1,T) so the analyzer treats it as a 1-seed cell."""
    os.makedirs(os.path.join(out, "metrics"), exist_ok=True)
    np.savez(os.path.join(out, "metrics", f"longtrain_generic_p0_rho{rho:.2f}.npz"),
             rho=float(rho), p=0, op="generic", wc=0.0,
             steps=np.asarray(steps, float),
             eff=np.asarray(eff, float).reshape(1, -1),
             acc=np.asarray(acc, float).reshape(1, -1),
             T_grok_per_seed=np.array([np.nan]))

def build_from_checkpoints(out, ckpt_dir, matrix_key, acc_csv, rho, step_col, acc_col):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.np[yz]")), key=lambda f: (_step_from_name(f) or 0))
    if not files:
        raise SystemExit(f"[error] no .npy/.npz checkpoints in {ckpt_dir}")
    csteps, effs = [], []
    for f in files:
        st = _step_from_name(f)
        if st is None:
            continue
        csteps.append(st); effs.append(eff_rank_from_matrix(_load_matrix(f, matrix_key)))
    csteps = np.array(csteps, float); effs = np.array(effs, float)
    asteps, accs, _ = _read_csv(acc_csv, step_col, acc_col)
    # align accuracy onto checkpoint steps by nearest-step lookup
    acc_on = np.interp(csteps, asteps, accs)
    write_cell(out, rho, csteps, effs, acc_on)
    print(f"[ok] checkpoint cell rho={rho}: {len(csteps)} steps, "
          f"eff {effs[0]:.1f}->{effs[-1]:.1f}, acc {accs.min():.2f}->{accs.max():.2f}")

def build_from_metrics(out, spec, step_col, acc_col, eff_col):
    path, rho = (spec.split(":") + [None])[:2]
    if rho is None:
        raise SystemExit(f"[error] tag each --metrics-csv as path:rho (got '{spec}')")
    steps, accs, effs = _read_csv(path, step_col, acc_col, eff_col)
    if effs is None:
        raise SystemExit("[error] --metrics-csv mode needs --eff-col")
    write_cell(out, float(rho), steps, effs, accs)
    print(f"[ok] metrics cell rho={rho}: {len(steps)} steps from {os.path.basename(path)}")

def main():
    ap = argparse.ArgumentParser(description="Run the compression-clock audit on third-party runs.")
    ap.add_argument("--out", required=True, help="output dir for normalized cells + verdict")
    ap.add_argument("--clock-script", required=True, help="path to analyze_compression_clock_v1_5.py")
    ap.add_argument("--checkpoints", help="MODE A: dir of weight checkpoints")
    ap.add_argument("--matrix-key", default="embedding", help="key of the matrix to audit in each ckpt")
    ap.add_argument("--acc-csv", help="MODE A: csv with step + test_acc")
    ap.add_argument("--rho", type=float, help="MODE A: norm-budget tag for the single cell")
    ap.add_argument("--metrics-csv", nargs="+", help="MODE B: one or more path:rho CSVs")
    ap.add_argument("--step-col", default="step")
    ap.add_argument("--acc-col", default="test_acc")
    ap.add_argument("--eff-col", default=None, help="MODE B: effective-rank column name")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    if args.checkpoints:
        if not (args.acc_csv and args.rho is not None):
            raise SystemExit("[error] --checkpoints needs --acc-csv and --rho")
        build_from_checkpoints(args.out, args.checkpoints, args.matrix_key,
                               args.acc_csv, args.rho, args.step_col, args.acc_col)
    elif args.metrics_csv:
        for spec in args.metrics_csv:
            build_from_metrics(args.out, spec, args.step_col, args.acc_col, args.eff_col)
    else:
        raise SystemExit("[error] pick a mode: --checkpoints or --metrics-csv")

    # call the canonical analyzer on the normalized dir
    spec = importlib.util.spec_from_file_location("cc", args.clock_script)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print("\n--- running canonical compression-clock analyzer on audited cells ---")
    sys.argv = ["cc", "--dir", args.out]
    m.main()

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    main()
