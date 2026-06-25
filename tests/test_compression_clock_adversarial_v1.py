r"""
test_compression_clock_adversarial_v1.py
====================================================================
ADVERSARIAL test suite for the compression-clock analyzer (current: analyze_compression_clock_v1_5.py).

Philosophy: each case TRIES TO BREAK the analyzer, not to confirm it. Several cases (6-9)
are ones we SUSPECT the analyzer might mishandle. Expected behavior is pre-registered in the
CASES table below BEFORE running. A case that fails reveals a real bug (good - we fix it),
not a reason to weaken the test.

Run:
  python test_compression_clock_adversarial_v1.py --clock_script analyze_compression_clock_v1_5.py
Exits non-zero if any case deviates from its pre-registered expectation.
"""
__version__ = "1.1.0"   # tightened: clock verdict must be backed by finite Spearman
import numpy as np, os, sys, glob, tempfile, importlib.util, argparse, io, contextlib

def synth(rho, grok_step, eff_peak, floor, comp_tau, T=200, dt=1000, S=6, seed=0,
          acc_final=1.0, post_bump=0.0, comp_before_grok=False, noise=0.3):
    """Make one (seeds,T) eff-rank + acc trajectory. Knobs let us build adversarial shapes."""
    steps = np.arange(T)*dt
    rng = np.random.default_rng(seed)
    eff = np.zeros((S, T)); acc = np.zeros((S, T))
    gi = max(1, grok_step//dt)
    for s in range(S):
        e = np.full(T, floor, float)
        e[:gi] = np.linspace(8, eff_peak, gi)
        post = np.arange(T-gi)*dt
        if comp_before_grok:
            # compression happens DURING memorization, before acc rises (violates "precedes")
            e[:gi] = floor + (eff_peak-floor)*np.exp(-np.arange(gi)*dt/comp_tau)
            e[gi:] = floor
        else:
            e[gi:] = floor + (eff_peak-floor)*np.exp(-post/comp_tau)
        if post_bump > 0:  # non-monotone: eff rises again late
            half = gi + (T-gi)//2
            e[half:] += post_bump * np.sin(np.linspace(0, np.pi, T-half))
        eff[s] = e + rng.normal(0, noise, T)
        acc[s] = np.clip(np.where(steps >= grok_step, acc_final, steps/max(grok_step,1)*0.9), 0, acc_final)
    return steps, eff, acc

def write_cell(d, tag, rho, steps, eff, acc, grok_step):
    S = eff.shape[0]
    np.savez(os.path.join(d, "metrics", f"longtrain_test_p59_rho{rho:.2f}.npz"),
             rho=rho, p=59, op="test", wc=55.0, steps=steps.astype(float),
             eff=eff, acc=acc, T_grok_per_seed=np.full(S, float(grok_step)))

def build_case(name):
    d = tempfile.mkdtemp(prefix=f"cc_{name}_"); os.makedirs(os.path.join(d, "metrics"))
    if name == "clean_one_clock":
        for rho, gs, pk, fl, tau in [(1.00,5000,48,8,9000),(1.10,7000,40,7,10000),
                                     (1.20,12000,25,6,11000),(1.30,16000,15,5,12000)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100)); write_cell(d,name,rho,s,e,a,gs)
    elif name == "all_censored":
        # compression timescale >> budget: nothing reaches floor within T
        for rho, gs in [(1.00,5000),(1.10,7000),(1.20,9000)]:
            s,e,a = synth(rho, gs, 45, 40, 1e12, seed=int(rho*100)); write_cell(d,name,rho,s,e,a,gs)
    elif name == "high_floor_boundary":
        # one boundary cell (high floor, tiny drop) + others compress -> must NOT become two-clock
        for rho, gs, pk, fl, tau, af in [(1.00,5000,46,42,1e12,0.99),  # boundary, never compresses
                                         (1.10,7000,48,8,9000,1.0),
                                         (1.20,12000,25,6,10000,1.0),
                                         (1.30,16000,15,5,11000,1.0)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100), acc_final=af); write_cell(d,name,rho,s,e,a,gs)
    elif name == "true_two_clock":
        # grok speed and compression speed ordered OPPOSITE across rho
        for rho, gs, pk, fl, tau in [(1.00,5000,48,8,40000),(1.10,6000,46,8,30000),
                                     (1.20,12000,26,7,10000),(1.30,16000,15,5,4000)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100)); write_cell(d,name,rho,s,e,a,gs)
    elif name == "single_cell":
        s,e,a = synth(1.10, 7000, 45, 8, 9000, seed=1); write_cell(d,name,1.10,s,e,a,7000)
    elif name == "non_monotone":
        for rho, gs, pk, fl, tau in [(1.00,5000,48,8,9000),(1.10,7000,40,7,10000),(1.20,12000,25,6,11000)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100), post_bump=12.0); write_cell(d,name,rho,s,e,a,gs)
    elif name == "compress_before_grok":
        for rho, gs, pk, fl, tau in [(1.00,9000,48,8,3000),(1.10,11000,40,7,3000),(1.20,14000,25,6,3000)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100), comp_before_grok=True); write_cell(d,name,rho,s,e,a,gs)
    elif name == "ties":
        # two cells engineered to give identical T_compress
        for rho, gs, pk, fl, tau in [(1.00,7000,40,8,9000),(1.10,7000,40,8,9000),(1.20,12000,25,6,11000)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100)); write_cell(d,name,rho,s,e,a,gs)
    elif name == "noisy":
        for rho, gs, pk, fl, tau in [(1.00,5000,48,8,9000),(1.10,7000,40,7,10000),
                                     (1.20,12000,25,6,11000),(1.30,16000,15,5,12000)]:
            s,e,a = synth(rho, gs, pk, fl, tau, seed=int(rho*100), noise=3.0); write_cell(d,name,rho,s,e,a,gs)
    return d

def verdict_line(out):
    for line in out.splitlines():
        if line.strip().startswith("VERDICT:"):
            return line.strip()
    return ""

# pre-registered expectations: (substring that MUST appear in VERDICT line, substring that must NOT)
CASES = {
    "clean_one_clock":      ("ONE CLOCK",            "TWO CLOCKS"),
    "all_censored":         ("censor",               None),        # checked in full output (not verdict line)
    "high_floor_boundary":  ("ONE CLOCK",            "TWO CLOCKS"),# 4 cells, gate drops rho=1.00 -> 3 valid -> ONE CLOCK
    "true_two_clock":       ("TWO CLOCKS",           None),
    "single_cell":          ("",                     "Traceback"), # must not crash
    "non_monotone":         ("",                     "Traceback"),
    "compress_before_grok": ("",                     "Traceback"),
    "ties":                 ("",                     "Traceback"), # tied T_grok+T_comp: must not crash; ordering may be defined or not
    "noisy":                ("ONE CLOCK",            "Traceback"), # enough cells, monotone -> ONE CLOCK
}

def run_case(clock_mod, d):
    buf = io.StringIO()
    rc = 0
    try:
        with contextlib.redirect_stdout(buf):
            sys.argv = ["cc_v13", "--dir", d]
            clock_mod.main()
    except Exception as e:
        import traceback
        buf.write("Traceback:\n" + traceback.format_exc())
        rc = 1
    return buf.getvalue(), rc

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clock_script", "--clock_v13", dest="clock_script", required=True,
                    help="path to the analyzer under test (e.g. analyze_compression_clock_v1_5.py)")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cc", args.clock_script)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    print("="*86); print("ADVERSARIAL UNIT TESTS — compression-clock analyzer"); print("="*86)
    print(f"{'case':24}{'must contain':18}{'must NOT':14}{'result':10}")
    failures = []
    for name, (must, mustnot) in CASES.items():
        d = build_case(name)
        out, rc = run_case(m, d)
        vline = verdict_line(out)
        ok = True; why = ""
        if "Traceback" in out:
            ok = False; why = "crashed"
        # clock-type assertions are checked on the VERDICT line; 'censor' is checked in full output
        hay = out if (must and must.lower() == "censor") else vline
        if must and must.lower() not in hay.lower():
            ok = False; why = f"verdict missing '{must}' (got: {vline})"
        if mustnot and mustnot != "Traceback" and mustnot.lower() in vline.lower():
            ok = False; why = f"verdict contains '{mustnot}'"
        # extra semantic check: no negative median gap fabricated
        for line in out.splitlines():
            if "median gap" in line.lower():
                tail = line.split("=")[-1]
                if tail.strip().startswith("-"):
                    ok = False; why = "negative gap"
        # v1.5 semantic guard: a CLOCK-TYPE verdict must be backed by a FINITE Spearman.
        # Claiming "ONE CLOCK"/"TWO CLOCKS" while Spearman printed nan is a false-confidence bug.
        clocky = any(w in vline.upper() for w in ["ONE CLOCK", "TWO CLOCKS", "PARTIALLY SEPARATED"])
        if clocky:
            sp_nan = any(("spearman" in l.lower() and "nan" in l.lower()) for l in out.splitlines())
            if sp_nan:
                ok = False; why = "clock verdict on NaN Spearman (false confidence)"
        print(f"{name:24}{str(must)[:16]:18}{str(mustnot)[:12]:14}{'PASS' if ok else 'FAIL '+why}")
        if not ok:
            failures.append((name, why, out))
    print("="*86)
    if failures:
        print(f"{len(failures)} CASE(S) REVEALED A PROBLEM (this is the test working):")
        for name, why, out in failures:
            print(f"\n--- {name}: {why} ---")
            print("\n".join(out.splitlines()[-12:]))
        sys.exit(1)
    print("ALL CASES BEHAVED AS PRE-REGISTERED.")

if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    main()
