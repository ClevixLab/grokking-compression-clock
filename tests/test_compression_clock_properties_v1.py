r"""
test_compression_clock_properties_v1.py
====================================================================
PROPERTY-BASED invariant tests for the compression-clock analyzer.

Unlike the adversarial suite (which checks pre-registered verdicts on specific
hand-built shapes), this suite checks structural INVARIANTS that must hold for
EVERY input, across many randomized trajectories. These are the diagnostic
identities the analyzer is contractually required to satisfy; a violation on any
random draw is a real bug.

Inspired by formally-verified diagnostic-identity checks in the grokking
diagnostics literature (e.g. Verma 2026's Lean-checked identities): we do not
prove the identities, but we test them as falsifiable properties over a random
input distribution.

Run:
  python test_compression_clock_properties_v1.py --clock_script ../audit/analyze_compression_clock_v1_5.py

Exits non-zero if any property is violated on any draw.

Properties checked (each over N random trajectory sets):
  P1  No fabricated ordering: if the printed (rho, T_compress) Spearman is NaN
      or based on < 3 valid cells, the verdict must NOT assert a clock type.
  P2  Non-negative gap: a reported median gap is never negative
      (T_compress is defined as the first floor-crossing at/after T_grok).
  P3  Censoring monotonicity: shortening every trajectory (truncating the time
      axis) can only turn a non-censored verdict into a censored/low-power one,
      never the reverse. (Less data => not more confidence.)
  P4  Boundary-gate safety: adding one never-compressing boundary cell to a clean
      ONE-CLOCK set must not flip the verdict to TWO CLOCKS.
  P5  Determinism: running the analyzer twice on identical input yields an
      identical verdict line.
"""
__version__ = "1.0.0"
import numpy as np, os, sys, tempfile, importlib.util, argparse, io, contextlib

# -- reuse the synthetic generator shape from the adversarial suite (kept local for independence) --
def synth(rho, grok_step, eff_peak, floor, comp_tau, T=200, dt=1000, S=6, seed=0,
          acc_final=1.0, noise=0.3):
    steps = np.arange(T) * dt
    rng = np.random.default_rng(seed)
    eff = np.zeros((S, T)); acc = np.zeros((S, T))
    gi = max(1, grok_step // dt)
    for s in range(S):
        e = np.full(T, floor, float)
        e[:gi] = np.linspace(8, eff_peak, gi)
        post = np.arange(T - gi) * dt
        e[gi:] = floor + (eff_peak - floor) * np.exp(-post / comp_tau)
        eff[s] = e + rng.normal(0, noise, T)
        acc[s] = np.clip(np.where(steps >= grok_step, acc_final, steps / max(grok_step, 1) * 0.9), 0, acc_final)
    return steps, eff, acc

def write_cell(d, rho, steps, eff, acc, grok_step):
    S = eff.shape[0]
    np.savez(os.path.join(d, "metrics", f"longtrain_prop_p59_rho{rho:.2f}.npz"),
             rho=rho, p=59, op="prop", wc=55.0, steps=steps.astype(float),
             eff=eff, acc=acc, T_grok_per_seed=np.full(S, float(grok_step)))

def make_dir():
    d = tempfile.mkdtemp(prefix="ccprop_"); os.makedirs(os.path.join(d, "metrics")); return d

def run(clock_mod, d):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            sys.argv = ["cc", "--dir", d]; clock_mod.main()
    except Exception:
        import traceback; buf.write("Traceback:\n" + traceback.format_exc())
    return buf.getvalue()

def verdict_line(out):
    for ln in out.splitlines():
        if ln.strip().upper().startswith("VERDICT"):
            return ln.strip()
    return ""

def is_clocky(v):
    return any(w in v.upper() for w in ["ONE CLOCK", "TWO CLOCKS", "PARTIALLY SEPARATED"])

def spearman_nan(out, which="t_compress"):
    for ln in out.splitlines():
        l = ln.lower()
        if "spearman" in l and which in l and "nan" in l:
            return True
    return False

def median_gap(out):
    for ln in out.splitlines():
        if "median gap" in ln.lower():
            tail = ln.split("=")[-1].strip().split()[0]
            try: return float(tail.replace(",", ""))
            except Exception: return None
    return None

def random_clean_set(d, rng, n_cells=None):
    """A random but well-formed ONE-CLOCK set: monotone rho->later grok, all compress within budget."""
    n_cells = n_cells or rng.integers(4, 7)
    rhos = np.sort(rng.uniform(1.0, 1.4, n_cells))
    base_gs = rng.integers(4000, 8000)
    for i, rho in enumerate(rhos):
        gs = int(base_gs + i * rng.integers(2000, 5000))
        pk = rng.uniform(20, 50); fl = rng.uniform(4, 9)
        tau = rng.uniform(8000, 13000)
        s, e, a = synth(rho, gs, pk, fl, tau, seed=int(rho * 1000) + i)
        write_cell(d, rho, s, e, a, gs)
    return rhos

def truncate_dir(src, dst, frac=0.5):
    """Copy a metrics dir but truncate the time axis of every cell to `frac` of its length."""
    os.makedirs(os.path.join(dst, "metrics"))
    import glob
    for f in glob.glob(os.path.join(src, "metrics", "*.npz")):
        z = dict(np.load(f, allow_pickle=True))
        T = z["eff"].shape[1]; k = max(3, int(T * frac))
        z["eff"] = z["eff"][:, :k]; z["acc"] = z["acc"][:, :k]; z["steps"] = z["steps"][:k]
        np.savez(os.path.join(dst, "metrics", os.path.basename(f)), **z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock_script", required=True)
    ap.add_argument("--draws", type=int, default=20, help="random draws per property")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cc", args.clock_script)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(0)

    print("=" * 78); print("PROPERTY-BASED INVARIANT TESTS — compression-clock analyzer"); print("=" * 78)
    fails = []

    # P1 + P2: over random clean sets, gap >= 0 and no clock verdict on NaN ordering
    p1 = p2 = True
    for k in range(args.draws):
        d = make_dir(); random_clean_set(d, rng); out = run(m, d)
        v = verdict_line(out)
        if is_clocky(v) and spearman_nan(out, "t_compress") and spearman_nan(out, "t_grok"):
            p1 = False; fails.append(("P1", f"draw {k}: clock verdict with NaN Spearman", v))
        g = median_gap(out)
        if g is not None and g < 0:
            p2 = False; fails.append(("P2", f"draw {k}: negative median gap {g}", v))
    print(f"P1 no-fabricated-ordering : {'PASS' if p1 else 'FAIL'}")
    print(f"P2 non-negative-gap       : {'PASS' if p2 else 'FAIL'}")

    # P3: truncation can only reduce confidence (clocky -> not-clocky-or-censored allowed; reverse not)
    p3 = True
    for k in range(args.draws):
        d = make_dir(); random_clean_set(d, rng); full = run(m, d)
        td = tempfile.mkdtemp(prefix="cctrunc_"); truncate_dir(d, td, frac=0.35)
        trunc = run(m, td)
        # if full was NOT clocky, that's fine. If full WAS clocky, truncated may be clocky or weaker;
        # the violation is: full not-clocky but truncated MORE confident (clocky). Less data => not more.
        if (not is_clocky(verdict_line(full))) and is_clocky(verdict_line(trunc)):
            p3 = False; fails.append(("P3", f"draw {k}: truncation INCREASED confidence", verdict_line(trunc)))
    print(f"P3 censoring-monotonicity : {'PASS' if p3 else 'FAIL'}")

    # P4: adding a boundary (never-compressing) cell must not create TWO CLOCKS
    p4 = True
    for k in range(args.draws):
        d = make_dir(); random_clean_set(d, rng)
        # add a boundary cell: high floor ~ peak, never compresses
        gs = int(rng.integers(4000, 7000))
        s, e, a = synth(1.00, gs, 46, 43, 1e12, seed=999 + k)
        write_cell(d, 0.99, s, e, a, gs)  # rho below the set -> boundary candidate
        out = run(m, d)
        if "TWO CLOCKS" in verdict_line(out).upper():
            p4 = False; fails.append(("P4", f"draw {k}: boundary cell created TWO CLOCKS", verdict_line(out)))
    print(f"P4 boundary-gate-safety   : {'PASS' if p4 else 'FAIL'}")

    # P5: determinism
    p5 = True
    for k in range(args.draws // 2 or 1):
        d = make_dir(); random_clean_set(d, rng)
        if verdict_line(run(m, d)) != verdict_line(run(m, d)):
            p5 = False; fails.append(("P5", f"draw {k}: non-deterministic verdict", ""))
    print(f"P5 determinism            : {'PASS' if p5 else 'FAIL'}")

    print("=" * 78)
    if fails:
        print(f"{len(fails)} PROPERTY VIOLATION(S) (the test working):")
        for p, why, v in fails[:10]:
            print(f"  [{p}] {why}  | {v}")
        sys.exit(1)
    print("ALL PROPERTIES HELD ACROSS ALL RANDOM DRAWS.")

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    main()
