#!/usr/bin/env python3
r"""
reproduce_all.py
====================================================================
One-command reproduction of the paper's analysis from PROCESSED trajectories
(no raw training required).

It will, for whichever task folders are present under sample_data/:
  1. run the adversarial test suite,
  2. run the analyzer on each task and collect the summary row,
  3. write a combined summary CSV (the paper's Table 2),
  4. rebuild the dashboard figure for each task.

Task folders expected (any subset; missing ones are skipped with a notice):
  sample_data/mlp_modadd_p59/metrics/longtrain_*rho*.npz
  sample_data/mlp_modmult_p59/metrics/longtrain_*rho*.npz
  sample_data/transformer_noln_p59/metrics/longtrain_*rho*.npz

Usage:
  python reproduce_all.py
  python reproduce_all.py --skip-tests --skip-figures
"""
import argparse, glob, os, subprocess, sys, csv

__version__ = "1.4.0"
HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZER = os.path.join(HERE, "audit", "analyze_compression_clock_v1_5.py")
TESTS = os.path.join(HERE, "tests", "test_compression_clock_adversarial_v1.py")
PROPS = os.path.join(HERE, "tests", "test_compression_clock_properties_v1.py")
FIGURE = os.path.join(HERE, "audit", "make_dashboard_figure_v1.py")
SAMPLE = os.path.join(HERE, "sample_data")
OUTDIR = os.path.join(HERE, "reproduced")

TASKS = [
    ("mlp_modadd_p59", "mod-add (MLP, p=59)"),
    ("mlp_modmult_p59", "mod-mult (MLP, p=59)"),
    ("transformer_noln_p59", "transformer no-LN (p=59)"),
]

def run(cmd, capture=True, timeout=300):
    """Run a subprocess. capture=False streams output directly (used for figures so large
    matplotlib stdout/stderr does not fill the pipe buffer and deadlock). Always bounded by timeout."""
    print("  $", " ".join(str(c) for c in cmd))
    try:
        if capture:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        else:
            r = subprocess.run(cmd, text=True, timeout=timeout)
        status = "OK" if r.returncode == 0 else f"FAILED (rc={r.returncode})"
        print("    ->", status)
        return r
    except subprocess.TimeoutExpired:
        print(f"    -> TIMEOUT after {timeout}s")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--arch-dir", default=None,
                    help="arch-generality run dir (arms/...) for section 8 (Fig 6). "
                         "Defaults to the bundled sample_data/arch_generality; pass a path to override.")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # Default the section-8 data to the copy bundled with the repo, so a bare
    # `python reproduce_all.py` reproduces every figure including Figure 6.
    if args.arch_dir is None:
        bundled = os.path.join(HERE, "sample_data", "arch_generality")
        if os.path.isdir(os.path.join(bundled, "arms")):
            args.arch_dir = bundled

    if not args.skip_tests:
        print("\n[1/6] adversarial test suite")
        r = run([sys.executable, TESTS, "--clock_script", ANALYZER])
        if r is None:
            print("   [warn] test suite timed out")
        else:
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
            print("   ->", tail)
            if "ALL CASES BEHAVED" not in r.stdout:
                print("   [warn] test suite did not report all-pass; see output above")
        print("     property-based invariant tests")
        rp = run([sys.executable, PROPS, "--clock_script", ANALYZER])
        if rp is None:
            print("   [warn] property tests timed out")
        else:
            tailp = rp.stdout.strip().splitlines()[-1] if rp.stdout.strip() else "(no output)"
            print("   ->", tailp)
            if "ALL PROPERTIES HELD" not in rp.stdout:
                print("   [warn] property tests did not all hold; see output above")

    print("\n[2/6] analyzer per task")
    summary_rows = []
    for folder, label in TASKS:
        d = os.path.join(SAMPLE, folder)
        if not glob.glob(os.path.join(d, "metrics", "longtrain_*rho*.npz")):
            print(f"   [skip] {folder}: no metrics/longtrain_*rho*.npz found")
            continue
        print(f"   analyzing {label}")
        r = run([sys.executable, ANALYZER, "--dir", d])
        if r is None:
            print("      [warn] analyzer timed out; skipping this task in summary")
            continue
        # the analyzer writes compression_clock_v1_5.csv into d; collect verdict line
        verdict = next((ln for ln in r.stdout.splitlines() if ln.startswith("VERDICT")), "VERDICT: (not found)")
        print("     ", verdict.strip())
        csv_path = os.path.join(d, "compression_clock_v1_5.csv")
        summary_rows.append((label, folder, verdict.strip(), csv_path if os.path.exists(csv_path) else ""))

    print("\n[3/6] writing combined summary")
    summ = os.path.join(OUTDIR, "summary_table.csv")
    with open(summ, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "folder", "verdict", "per_cell_csv"])
        w.writerows(summary_rows)
    print("   ->", summ)

    if not args.skip_figures:
        print("\n[4/6] rebuilding dashboards")
        for folder, label in TASKS:
            d = os.path.join(SAMPLE, folder)
            if not glob.glob(os.path.join(d, "metrics", "longtrain_*rho*.npz")):
                continue
            out = os.path.join(OUTDIR, f"dashboard_{folder}.pdf")
            run([sys.executable, FIGURE, "--dir", d, "--task", label, "--out", out], capture=False)
            print("   ->", out)

    print("\n[5/6] robustness analyses (sensitivity, floor-CI, corrected reading, unembedding rank)")
    SENS = os.path.join(HERE, "audit", "sensitivity_floor_ci.py")
    CORR = os.path.join(HERE, "audit", "corrected_reading.py")
    URANK = os.path.join(HERE, "audit", "compute_unembedding_rank.py")
    if glob.glob(os.path.join(SAMPLE, "mlp_modadd_p59", "metrics", "*.npz")):
        run([sys.executable, SENS, "--data", SAMPLE, "--out_dir", OUTDIR])
    if os.path.isdir(os.path.join(SAMPLE, "mlp_grid_small")):
        run([sys.executable, CORR, "--small", os.path.join(SAMPLE, "mlp_grid_small"), "--full", SAMPLE])
    struct_dir = os.path.join(SAMPLE, "transformer_noln_p59", "struct")
    if glob.glob(os.path.join(struct_dir, "*_struct.npz")):
        run([sys.executable, URANK, "--struct_dir", struct_dir,
             "--metrics_dir", os.path.join(SAMPLE, "transformer_noln_p59", "metrics_struct"),
             "--out_dir", os.path.join(SAMPLE, "transformer_noln_p59", "struct_eu")])
    TRANS = os.path.join(HERE, "audit", "transient_replication.py")
    if os.path.isdir(os.path.join(SAMPLE, "mlp_freewd_p59", "configs")):
        print("     transient replication (free-wd + parity; depth-law direction caveat)")
        run([sys.executable, TRANS, "--data", os.path.join(SAMPLE, "mlp_freewd_p59")], capture=False)
        print("     metric-agnostic transient (eff rank vs participation ratio vs stable rank)")
        MAT = os.path.join(HERE, "audit", "metric_agnostic_transient.py")
        run([sys.executable, MAT], capture=False)

    if not args.skip_figures:
        print("\n[6/6] supplementary figures (floor-CI, transformer transient + depth-law ablation)")
        TFSWEEP = os.path.join(HERE, "audit", "transformer_sweep_audit.py")
        if glob.glob(os.path.join(SAMPLE, "transformer_sweep_p59", "transformer__*.npz")):
            print("     transformer dose-response: transient + depth-law non-replication")
            run([sys.executable, TFSWEEP, "--out", OUTDIR], capture=False)
        FLOORFIG = os.path.join(HERE, "audit", "make_floor_ci_figure.py")
        EUFIG = os.path.join(HERE, "audit", "make_transformer_eu_figure.py")
        ATGROKFIG = os.path.join(HERE, "audit", "make_atgrok_vs_converged_figure.py")
        if glob.glob(os.path.join(SAMPLE, "mlp_modadd_p59", "metrics", "*.npz")):
            run([sys.executable, FLOORFIG, "--data", SAMPLE, "--out", os.path.join(OUTDIR, "floor_law_ci")], capture=False)
            run([sys.executable, ATGROKFIG, "--data", SAMPLE, "--out", os.path.join(OUTDIR, "fig_atgrok_vs_converged")], capture=False)
        if glob.glob(os.path.join(SAMPLE, "transformer_noln_p59", "struct_eu", "*.npz")):
            run([sys.executable, EUFIG, "--eu_dir", os.path.join(SAMPLE, "transformer_noln_p59", "struct_eu"),
                 "--metrics_dir", os.path.join(SAMPLE, "transformer_noln_p59", "metrics_struct"),
                 "--out", os.path.join(OUTDIR, "fig_transformer_eu")], capture=False)

    print("\n[7/7] sections 8-9: arch-generality harness (Figs 6-7)")
    if args.arch_dir and os.path.isdir(os.path.join(args.arch_dir, "arms")):
        ARCH = os.path.join(HERE, "audit", "arch_generality_audit.py")
        SCALE = os.path.join(HERE, "audit", "analyze_scale_invariance.py")
        PREREG = os.path.join(HERE, "specs", "PREREGISTRATION_scale_inv.json")
        print("   section 8: one harness, three architectures (transient / lag / frac-pre + Fig 6)")
        run([sys.executable, ARCH, "--arch-dir", args.arch_dir, "--out", OUTDIR], capture=False)
        print("   section 9: pre-registered scale-invariance control (PD1/PD2/PD3 + Fig 7)")
        run([sys.executable, SCALE, "--arch-dir", args.arch_dir, "--out", OUTDIR,
             "--prereg", PREREG], capture=False)
    else:
        print("   [skip] no --arch-dir with arms/ given; sections 8-9 (Figs 6-7) not reproduced.")
        print("          run:  python reproduce_all.py --arch-dir <arch_run_dir>")

    print("\nDone. Reproduced artifacts are in:", OUTDIR)
    if not summary_rows:
        print("NOTE: no task folders had data. Add processed npz under sample_data/<task>/metrics/")

if __name__ == "__main__":
    main()
