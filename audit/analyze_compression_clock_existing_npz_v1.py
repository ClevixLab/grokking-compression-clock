r"""
analyze_compression_clock_existing_npz_v1.py   (v1.0.0)
====================================================================
Runs the compression-clock audit on EXISTING logged trajectories that follow the
externally-logged npz convention (keys: steps, test_acc, eff_rank_E, rho, arm), i.e. files named
  metrics/*.npz   (NOT the longtrain_*rho*.npz convention of the fresh MLP runs).

Two independent jobs, kept separate on purpose:

  JOB 1 - COMPRESSION CLOCK (dense trajectory).
    Uses eff_rank_E(seeds,T) + test_acc(seeds,T) + steps(T). This is the only dense signal,
    so the clock is computed on the EMBEDDING locus. Reuses the exact analyzer logic (boundary
    gate, censoring, opposite-order guard) by importing it.

  JOB 2 - LOCUS CHECK (2-point, E vs U).
    Structural ckpts (E_ckpts, U_ckpts) are log-spaced in time and, for slow high-rho cells,
    give essentially only an at-grok-ish point and a final point. So we do NOT build a clock
    from them; we only compare compression of E vs U between the near-grok ckpt and the final
    ckpt. Pre-registered reading:
      - E not compressed AND U not compressed (high rho) -> genuine censoring/incomplete, not
        a locus error; the clock's boundary/censor flag is correct.
      - E not compressed BUT U compressed            -> wrong locus on the transformer:
        compression lives in the unembedding. (Reported as SUGGESTIVE for high rho, where only
        1-2 post-grok ckpts exist; more reliable at low rho / ctrl where post-grok ckpts exist.)
      - both compressed at low rho/ctrl              -> MLP story replicates on a new architecture.

Usage:
  python analyze_compression_clock_existing_npz_v1.py \
     --dir /path/to/your_logged_run \
     --clock_script audit/analyze_compression_clock_v1_5.py
"""
import numpy as np, glob, os, sys, json, argparse, importlib.util, tempfile, shutil

__version__ = "1.0.0"

def eff_rank_from_matrix(W):
    """spectral-entropy effective rank of a 2-D matrix (same definition as the MLP runs)."""
    W = np.asarray(W, float)
    s = np.linalg.svd(W, compute_uv=False); s2 = s**2; t = s2.sum()
    if t <= 0: return 0.0
    p = s2/t; nz = p[p > 0]
    return float(np.exp(-(nz*np.log(nz)).sum()))

def load_clock_module(path):
    spec = importlib.util.spec_from_file_location("cc_v13", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def job1_clock(metrics_dir, clock_mod, workdir):
    """Re-emit existing metrics into the longtrain_*rho*.npz schema the analyzer expects,
    then call its load_rows + verdict path. We shim by writing temp npz with keys eff,acc,steps,rho."""
    os.makedirs(os.path.join(workdir, "metrics"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(metrics_dir, "metrics", "*.npz")))
    n = 0; ctrl = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "eff_rank_E" not in z.files or "test_acc" not in z.files or "rho" not in z.files:
            continue
        arm = str(z["arm"]) if "arm" in z.files else "clamp"
        rho = float(z["rho"])
        eff = np.asarray(z["eff_rank_E"], float)     # (seeds,T)
        acc = np.asarray(z["test_acc"], float)       # (seeds,T)
        steps = np.asarray(z["steps"], float)        # (T,)
        Tg = np.asarray(z["T_grok_per_seed"], float) if "T_grok_per_seed" in z.files else np.full(eff.shape[0], -1.0)
        # The clock dose-response is over the CLAMP arm only. The free control ('none') shares
        # rho=1.00 with a clamp cell and would corrupt the rho ordering; keep it aside.
        if arm != "clamp":
            ctrl = dict(rho=rho, eff=eff, acc=acc, steps=steps, Tg=Tg, arm=arm)
            continue
        out = os.path.join(workdir, "metrics", f"longtrain_tfnoln_p59_rho{rho:.2f}.npz")
        np.savez(out, rho=rho, p=59, op="tfnoln",
                 wc=float(z["wc_used"]) if "wc_used" in z.files else 0.0,
                 steps=steps, eff=eff, acc=acc, T_grok_per_seed=Tg)
        n += 1
    return n, ctrl

def job2_locus(metrics_dir):
    """2-point E-vs-U compression comparison from structural ckpts."""
    rows = []
    for f in sorted(glob.glob(os.path.join(metrics_dir, "Temp", "*struct.npz"))):
        z = np.load(f, allow_pickle=True)
        if "E_ckpts" not in z.files or "U_ckpts" not in z.files:
            continue
        E = z["E_ckpts"]; U = z["U_ckpts"]      # (C, seeds, .., ..)
        cs = np.asarray(z["ckpt_steps"], float)
        tg = np.median(np.asarray(z["T_grok"], float)) if "T_grok" in z.files else np.nan
        rho = float(z["rho"]); arm = str(z["arm"]) if "arm" in z.files else "clamp"
        C, S = E.shape[0], E.shape[1]
        # near-grok ckpt = last ckpt with step <= ~1.5*T_grok (fallback: middle); final = last ckpt
        if np.isfinite(tg):
            cand = np.where(cs <= 1.5*tg)[0]
            gi = cand[-1] if len(cand) else max(0, C-2)
        else:
            gi = max(0, C-2)
        fi = C-1
        def med_eff(M, ci):
            return float(np.median([eff_rank_from_matrix(M[ci, s]) for s in range(S)]))
        eE_g, eE_f = med_eff(E, gi), med_eff(E, fi)
        eU_g, eU_f = med_eff(U, gi), med_eff(U, fi)
        rows.append(dict(rho=rho, arm=arm, tg=tg,
                         step_g=float(cs[gi]), step_f=float(cs[fi]),
                         nckpt_postgrok=int((cs > tg).sum()) if np.isfinite(tg) else -1,
                         effE_g=eE_g, effE_f=eE_f, dropE=eE_g-eE_f,
                         effU_g=eU_g, effU_f=eU_f, dropU=eU_g-eU_f))
    rows.sort(key=lambda r: (r["arm"], r["rho"]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clock_script", "--clock_v13", dest="clock_script", required=True,
                    help="path to the current analyzer, e.g. analyze_compression_clock_v1_5.py")
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="ccexist_")
    try:
        clock_mod = load_clock_module(args.clock_script)
        n, ctrl = job1_clock(args.dir, clock_mod, workdir)
        print("="*92)
        print(f"COMPRESSION-CLOCK on existing npz  [adapter v{__version__}]   (locus = embedding E, dense traj)")
        print(f"source: {args.dir}\n re-emitted {n} CLAMP cells -> running analyzer logic (free control reported separately below)\n"+"="*92)
        # call the analyzer main by pointing it at workdir via argv
        sys.argv = ["cc_v13", "--dir", workdir]
        clock_mod.main()
        if ctrl is not None:
            steps = ctrl["steps"]; eff = np.median(ctrl["eff"], axis=0); acc = np.median(ctrl["acc"], axis=0)
            tg = np.median(ctrl["Tg"]) if np.isfinite(ctrl["Tg"]).any() else np.nan
            gi = int(np.argmax(acc >= 0.9)) if (acc >= 0.9).any() else -1
            print(f"\n[free control, arm={ctrl['arm']}, rho={ctrl['rho']:.2f}] (NOT in the dose-response above)")
            print(f"   T_grok≈{steps[gi]:.0f}  eff@grok≈{eff[gi]:.1f}  eff@final≈{eff[-1]:.1f}  "
                  f"drop≈{eff[gi]-eff[-1]:+.1f}  acc@final≈{acc[-1]:.2f}")
    finally:
        pass

    print("\n" + "="*92)
    print("LOCUS CHECK (2-point, E vs U from structural ckpts) — SUGGESTIVE at high rho (few post-grok ckpts)")
    print("="*92)
    rows = job2_locus(args.dir)
    print(f"{'arm':6}{'rho':>6}{'T_grok':>9}{'#post':>6}{'effE_g':>8}{'effE_f':>8}{'dropE':>7}{'effU_g':>8}{'effU_f':>8}{'dropU':>7}  read")
    for r in rows:
        # pre-registered reading per cell
        E_comp = r["dropE"] > 2.0
        U_comp = r["dropU"] > 2.0
        if not E_comp and U_comp:   read = "WRONG-LOCUS? U compresses, E doesn't"
        elif not E_comp and not U_comp: read = "neither compresses (censor/incomplete)"
        elif E_comp and U_comp:     read = "both compress (story holds)"
        else:                       read = "E compresses, U doesn't"
        rel = "" if r["nckpt_postgrok"] >= 2 else "  [<2 post-grok ckpts: weak]"
        print(f"{r['arm']:6}{r['rho']:>6.2f}{r['tg']:>9.0f}{r['nckpt_postgrok']:>6}"
              f"{r['effE_g']:>8.1f}{r['effE_f']:>8.1f}{r['dropE']:>7.1f}"
              f"{r['effU_g']:>8.1f}{r['effU_f']:>8.1f}{r['dropU']:>7.1f}  {read}{rel}")
    print("\nReading guide (pre-registered):")
    print("  E flat + U flat at high rho  -> genuine censoring (grok very late, little post-grok budget); clock's flag is correct.")
    print("  E flat + U compresses        -> embedding is the WRONG locus on the transformer; compression is in the unembedding.")
    print("  both compress at low rho/ctrl-> MLP compression story replicates on this new architecture.")
    print("  high-rho rows are SUGGESTIVE only (1-2 post-grok ckpts); trust low-rho / ctrl rows more.")
    out = os.path.join(args.dir, "locus_check_EvU_v1.json")
    json.dump(rows, open(out, "w"), indent=1, default=float)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    main()
