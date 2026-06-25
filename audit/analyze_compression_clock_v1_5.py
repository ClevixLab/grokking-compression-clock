r"""
analyze_compression_clock_v1_5.py   (v1.5.0)
====================================================================
Tests the "two clocks" hypothesis on long-train data:

  Is low-rank compression a separate dynamical event from grokking onset,
  or just grokking plus a small/fixed lag?

This is v1.5, which consolidates the fixes introduced across v1.1--v1.5:
  1. TWO-CLOCK verdict requires opposite ordering of T_grok and T_compress
     across rho, not merely different argmin locations.
  2. Compression events can be censored if the run ends before the threshold
     is reached; censoring is reported and gates the verdict.
  3. Compression amplitude is checked; tiny drops are marked undefined.
  4. Shared endpoint is assessed by floor CV as well as absolute span.
  5. (v1.5) A clock-type verdict must be backed by a finite order statistic;
     claiming a clock on a NaN Spearman is treated as a false-confidence bug.

Expected long-train npz schema:
  metrics/longtrain_*rho*.npz
  keys:
    rho     : scalar
    steps   : (T,)
    eff     : (seeds,T)  effective-rank trajectory
    acc     : (seeds,T)  test-accuracy trajectory

Robust fallbacks:
  eff_rank_E may replace eff; test_acc may replace acc.

Definitions per seed:
  T_grok      : first step with test_acc >= grok_thr
  eff_grok    : eff_rank at T_grok
  eff_floor   : final plateau = median eff_rank over last floor_frac of logged steps
  drop        : eff_grok - eff_floor
  T_compress  : first step after grok where eff_rank <= eff_floor + eps*drop
                eps=0.10 means 90% of the way from eff_grok to eff_floor
  gap         : T_compress - T_grok
  comp_rate   : drop/gap
  censored    : True if compression threshold is not reached before final logged step
  valid_comp  : False if drop < min_drop or no compression event is defined

Usage:
  python analyze_compression_clock_v1_5.py --dir /path/to/run_dir

Optional:
  --eps 0.10 --floor_frac 0.10 --min_drop 1.0 --grok_thr 0.90
"""

import os, sys, glob, json, argparse, datetime
import numpy as np

__version__ = "1.5.0"   # v1.5: merge of v1.4 reporting (low-power verdict, duplicate-rho warning, no-grok accounting) + v1.3.1 fixes (spearman>=3 not >=4; no clock claim on undefined ordering)


def _load(f):
    return np.load(f, allow_pickle=True)


def _key(z, *names):
    for n in names:
        if n in z.files:
            return np.asarray(z[n])
    raise KeyError(f"none of keys found: {names}; available={list(z.files)}")


def _rank(a):
    a = np.asarray(a, float)
    o = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c), dtype=float)
    np.add.at(ss, inv, r)
    return (ss / c)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:          # v1.5: 3 monotone cells give a valid rank corr (was <4, mislabeled
        return np.nan        # valid 3-cell sweeps as undetermined)
    rx, ry = _rank(x[m]), _rank(y[m])
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def med_dict(rows, k):
    v = [x[k] for x in rows if np.isfinite(x.get(k, np.nan))]
    return float(np.median(v)) if v else np.nan


def frac(rows, pred):
    if not rows:
        return np.nan
    return float(sum(1 for r in rows if pred(r)) / len(rows))


def per_seed_clocks(z, eps=0.10, floor_frac=0.10, min_drop=1.0, grok_thr=0.90):
    steps = _key(z, "steps").astype(float)
    eff = _key(z, "eff", "eff_rank_E").astype(float)
    acc = _key(z, "acc", "test_acc").astype(float)

    if eff.ndim != 2 or acc.ndim != 2:
        raise ValueError(f"expected eff and acc as (seeds,T); got {eff.shape}, {acc.shape}")
    if eff.shape != acc.shape:
        raise ValueError(f"eff and acc shapes differ: {eff.shape} vs {acc.shape}")
    if eff.shape[1] != len(steps):
        raise ValueError(f"steps length {len(steps)} does not match trajectory length {eff.shape[1]}")

    S, T = eff.shape
    nfloor = max(1, int(floor_frac * T))
    out = []

    for s in range(S):
        e = eff[s]
        a = acc[s]
        finite = np.isfinite(e) & np.isfinite(a)
        if not finite.any():
            continue

        good = np.where((a >= grok_thr) & finite)[0]
        if len(good) == 0:
            continue

        gi = int(good[0])
        Tg = float(steps[gi])
        eff_g = float(e[gi])

        tail = e[-nfloor:]
        tail = tail[np.isfinite(tail)]
        if len(tail) == 0 or not np.isfinite(eff_g):
            continue

        floor = float(np.median(tail))
        drop = float(eff_g - floor)
        valid_amp = bool(np.isfinite(drop) and drop >= min_drop)

        if not valid_amp:
            out.append(dict(seed=s, Tg=Tg, eff_g=eff_g, floor=floor, drop=drop,
                            Tc=np.nan, eff_c=np.nan, gap=np.nan, rel_gap=np.nan,
                            rate=np.nan, censored=False, valid_comp=False))
            continue

        thr = floor + eps * drop
        post = np.where((steps >= Tg) & np.isfinite(e))[0]
        ci = None
        for i in post:
            if e[i] <= thr:
                ci = int(i)
                break

        censored = ci is None
        if censored:
            ci = int(post[-1])
            Tc = float(steps[ci])
            eff_c = float(e[ci])
        else:
            Tc = float(steps[ci])
            eff_c = float(e[ci])

        gap = float(Tc - Tg)
        rel_gap = float(gap / Tg) if Tg > 0 and np.isfinite(gap) else np.nan
        rate = float(drop / gap) if gap > 0 and np.isfinite(drop) else np.nan

        out.append(dict(seed=s, Tg=Tg, eff_g=eff_g, floor=floor, drop=drop,
                        Tc=Tc, eff_c=eff_c, gap=gap, rel_gap=rel_gap,
                        rate=rate, censored=bool(censored), valid_comp=True))
    return out


def load_rows(files, args):
    rows = []
    for f in files:
        z = _load(f)
        rho = float(z["rho"])
        cs = per_seed_clocks(z, eps=args.eps, floor_frac=args.floor_frac,
                             min_drop=args.min_drop, grok_thr=args.grok_thr)
        if not cs:
            # Keep an explicit row instead of silently dropping no-grok / unusable cells.
            rows.append(dict(
                file=f, rho=rho, n=0, n_valid=0, n_censored=0, censor_frac=np.nan,
                amp_invalid_frac=np.nan, Tg=np.nan, eff_g=np.nan, floor=np.nan, drop=np.nan,
                Tc=np.nan, gap=np.nan, rel_gap=np.nan, rate=np.nan, raw=[],
                no_grok_file=True, boundary=True,
            ))
            continue
        valid = [c for c in cs if c.get("valid_comp", False)]
        r = dict(
            file=f,
            rho=rho,
            n=len(cs),
            n_valid=len(valid),
            n_censored=sum(1 for c in valid if c.get("censored", False)),
            censor_frac=frac(valid, lambda x: x.get("censored", False)) if valid else np.nan,
            amp_invalid_frac=frac(cs, lambda x: not x.get("valid_comp", False)),
            Tg=med_dict(cs, "Tg"),
            eff_g=med_dict(cs, "eff_g"),
            floor=med_dict(cs, "floor"),
            drop=med_dict(cs, "drop"),
            Tc=med_dict(valid, "Tc"),
            gap=med_dict(valid, "gap"),
            rel_gap=med_dict(valid, "rel_gap"),
            rate=med_dict(valid, "rate"),
            raw=cs,
            no_grok_file=False,
        )
        rows.append(r)
    rows.sort(key=lambda r: r["rho"])
    return rows


def fmt(x, nd=2):
    if x is None or not np.isfinite(x):
        return "nan"
    if nd == 0:
        return f"{x:.0f}"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--floor_frac", type=float, default=0.10)
    ap.add_argument("--min_drop", type=float, default=1.0)
    ap.add_argument("--grok_thr", type=float, default=0.90)
    ap.add_argument("--censor_warn", type=float, default=0.25,
                    help="if overall censored fraction exceeds this, verdict becomes inconclusive")
    ap.add_argument("--big_gap_rel", type=float, default=0.50,
                    help="median(gap/T_grok) threshold for saying compression is temporally separated")
    ap.add_argument("--floor_cv_shared", type=float, default=0.15,
                    help="floor CV below this is treated as shared low-rank endpoint")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "metrics", "longtrain_*rho*.npz")))
    if not files:
        print("[error] no longtrain npz found in", args.dir)
        return

    rows = load_rows(files, args)
    if not rows:
        print("[error] no usable rows")
        return

    # v1.4 reporting guards: duplicate rho values can arise when a free/control arm shares
    # the same nominal rho as a clamp arm. The raw analyzer has no arm metadata, so it warns
    # rather than silently treating duplicate rho as an ordinary dose-response axis.
    rho_vals = [r["rho"] for r in rows if np.isfinite(r.get("rho", np.nan))]
    dup_rhos = sorted(set([x for x in rho_vals if rho_vals.count(x) > 1]))

    print("=" * 96)
    print(f"COMPRESSION CLOCK  [v{__version__}]")
    print(f"  eps={args.eps} -> compression threshold = floor + {args.eps:.2f}*(eff_grok-floor)")
    print(f"  floor_frac={args.floor_frac}, min_drop={args.min_drop}, grok_thr={args.grok_thr}")
    if dup_rhos:
        print(f"  [warning] duplicate rho values detected: {dup_rhos}. If these are ctrl/clamp collisions, use an adapter that separates arms before dose-response analysis.")
    n_no_grok_files = sum(1 for r in rows if r.get("no_grok_file", False))
    if n_no_grok_files:
        print(f"  [warning] {n_no_grok_files} file(s) had no seed reaching grok_thr or no usable trajectories; kept as explicit no-grok rows and excluded from clock verdict.")
    print("=" * 96)
    print(f"{'rho':>6}{'n':>4}{'valid':>7}{'cens%':>8}{'T_grok':>9}{'eff_grok':>10}"
          f"{'floor':>9}{'drop':>9}{'T_comp':>10}{'gap':>9}{'gap/Tg':>9}{'rate':>10}")

    for r in rows:
        cens_pct = 100*r['censor_frac'] if np.isfinite(r['censor_frac']) else np.nan
        print(f"{r['rho']:>6.2f}{r['n']:>4}{r['n_valid']:>7}{cens_pct:>7.1f}%"
              f"{fmt(r['Tg'],0):>9}{fmt(r['eff_g'],2):>10}{fmt(r['floor'],2):>9}"
              f"{fmt(r['drop'],2):>9}{fmt(r['Tc'],0):>10}{fmt(r['gap'],0):>9}"
              f"{fmt(r['rel_gap'],2):>9}{fmt(r['rate'],5):>10}")

    # ---- BOUNDARY-CELL GATE (added in v1.3) -------------------------------------------------
    # A cell that never actually compressed produces a spuriously high floor and a
    # meaningless T_compress (the trajectory sits at its high plateau, so it "reaches"
    # floor+eps immediately). Such a cell is a rho_min/boundary case (incomplete grok),
    # NOT a second clock. Flag and EXCLUDE it before the clock verdict.
    #   criterion 1: compression ratio drop/eff_grok is tiny (barely fell)
    #   criterion 2: floor is a high outlier vs the other cells' floors
    comp_ratio = {r["rho"]: (r["drop"]/r["eff_g"] if (r["eff_g"] and np.isfinite(r["drop"])) else np.nan)
                  for r in rows}
    finite_floors = [r["floor"] for r in rows if np.isfinite(r["floor"])]
    floor_median_all = float(np.median(finite_floors)) if finite_floors else np.nan
    boundary = []
    for r in rows:
        cr = comp_ratio.get(r["rho"], np.nan)
        high_floor = (np.isfinite(r["floor"]) and np.isfinite(floor_median_all)
                      and r["floor"] > 3.0*floor_median_all)   # floor >> typical compressed floor
        tiny_drop  = (np.isfinite(cr) and cr < 0.25)            # fell <25% of its at-grok rank
        if high_floor or tiny_drop:
            boundary.append(r["rho"]); r["boundary"] = True
        else:
            r["boundary"] = False
    if boundary:
        print("\n[BOUNDARY GATE] cells excluded as non-compressed / incomplete-grok (rho_min boundary):")
        for r in rows:
            if r.get("boundary"):
                print(f"   rho={r['rho']:.2f}: floor={fmt(r['floor'],1)} (vs typ {fmt(floor_median_all,1)}), "
                      f"drop/eff_grok={fmt(comp_ratio[r['rho']],2)}, acc may be <1.0 -> NOT a clock event")
        print("   These are reported as a capacity/rho_min boundary, NOT as a second compression clock.")

    valid_rows = [r for r in rows if r["n_valid"] > 0 and np.isfinite(r["Tc"]) and np.isfinite(r["gap"])
                  and not r.get("boundary", False)]
    rhos = np.asarray([r["rho"] for r in valid_rows], float)
    gaps = np.asarray([r["gap"] for r in valid_rows], float)
    rel_gaps = np.asarray([r["rel_gap"] for r in valid_rows], float)
    Tg = np.asarray([r["Tg"] for r in valid_rows], float)
    Tc = np.asarray([r["Tc"] for r in valid_rows], float)
    floors = np.asarray([r["floor"] for r in valid_rows], float)
    rates = np.asarray([r["rate"] for r in valid_rows], float)

    print("\n[Q0 data quality]")
    total_valid = sum(r["n_valid"] for r in rows)
    total_cens = sum(r["n_censored"] for r in rows)
    overall_censor = total_cens / total_valid if total_valid else np.nan
    print(f"   valid compression seeds = {total_valid}; censored = {total_cens} ({100*overall_censor:.1f}%)")
    for r in rows:
        if r.get("no_grok_file", False):
            print(f"   rho={r['rho']:.2f}: NO-GROK/UNUSABLE file (no seed reached grok_thr); excluded from clock verdict")
    for r in rows:
        if r["amp_invalid_frac"] > 0:
            print(f"   rho={r['rho']:.2f}: {100*r['amp_invalid_frac']:.1f}% seeds have drop < min_drop")

    print("\n[Q1 separability] gap = T_compress - T_grok")
    if len(valid_rows) < 2:
        gap_med = gap_span = np.nan
        print("   insufficient valid rows")
    else:
        gap_med = float(np.nanmedian(gaps))
        gap_span = float(np.nanmax(gaps) - np.nanmin(gaps))
        rel_med = float(np.nanmedian(rel_gaps))
        print(f"   median gap = {gap_med:.0f} steps ; gap spread across rho = {gap_span:.0f}")
        print(f"   median gap/T_grok = {rel_med:.2f}  "
              f"({'large separate phase' if rel_med > args.big_gap_rel else 'small/modest lag'})")

    print("\n[Q2 ordering] do grokking and compression order rho differently?")
    if len(valid_rows) < 2:
        rho_fast_grok = rho_early_comp = rho_fast_rate = np.nan
        sp_g = sp_c = sp_gap = np.nan
        print("   insufficient valid rows")
    else:
        rho_fast_grok = valid_rows[int(np.nanargmin(Tg))]["rho"]
        rho_early_comp = valid_rows[int(np.nanargmin(Tc))]["rho"]
        rho_fast_rate = valid_rows[int(np.nanargmax(rates))]["rho"] if np.isfinite(rates).any() else np.nan
        sp_g = spearman(rhos, Tg)
        sp_c = spearman(rhos, Tc)
        sp_gap = spearman(rhos, gaps)
        print(f"   fastest grok onset (min T_grok):       rho={rho_fast_grok}")
        print(f"   earliest compression (min T_comp):    rho={rho_early_comp}")
        print(f"   fastest compression rate:             rho={rho_fast_rate}")
        print(f"   Spearman(rho,T_grok)={sp_g:+.3f}")
        print(f"   Spearman(rho,T_compress)={sp_c:+.3f}")
        print(f"   Spearman(rho,gap)={sp_gap:+.3f}")
        print(f"   opposite-order condition: {'YES' if np.isfinite(sp_g) and np.isfinite(sp_c) and sp_g*sp_c < 0 else 'NO'}")

    print("\n[Q3 endpoint] eff_floor across rho")
    if len(floors) < 2 or not np.isfinite(floors).all():
        floor_span = floor_cv = np.nan
        print("   insufficient floor data")
    else:
        floor_span = float(np.nanmax(floors) - np.nanmin(floors))
        floor_cv = float(np.nanstd(floors) / max(1e-12, np.nanmean(floors)))
        print(f"   floor range = [{np.nanmin(floors):.2f}, {np.nanmax(floors):.2f}] "
              f"(span {floor_span:.2f}); CV={floor_cv:.3f}")
        print(f"   endpoint verdict: {'SHARED low-rank endpoint' if floor_cv < args.floor_cv_shared else 'endpoint varies with rho'}")

    big_gap = len(valid_rows) >= 2 and np.isfinite(rel_gaps).any() and np.nanmedian(rel_gaps) > args.big_gap_rel
    high_censor = np.isfinite(overall_censor) and overall_censor > args.censor_warn
    ordering_known = np.isfinite(sp_g) and np.isfinite(sp_c)
    opposite_order = ordering_known and (sp_g * sp_c < 0)
    shifted_argmin = np.isfinite(rho_fast_grok) and np.isfinite(rho_early_comp) and (rho_fast_grok != rho_early_comp)

    print("\n" + "=" * 96)
    if len(valid_rows) < 2:
        print("VERDICT: LOW-POWER / DESCRIPTIVE ONLY.")
        print("   Fewer than two non-boundary rows have a valid compression event after gating; do not infer clock ordering or lag class.")
    elif high_censor:
        print("VERDICT: INCONCLUSIVE due to censoring.")
        print(f"   {100*overall_censor:.1f}% of valid compression seeds did not reach the compression threshold.")
    elif big_gap and not ordering_known:
        # v1.5: gap is real but the rho-ordering could not be established (Spearman undefined:
        # tied/degenerate values). Do NOT assert a shared clock from ordering we don't have.
        print("VERDICT: LARGE LAG, ORDERING UNDETERMINED.")
        print("   Compression lags grokking by a large amount, but the rho-ordering of the two times")
        print("   could not be established (Spearman undefined: tied values). Report the lag only.")
    elif big_gap and opposite_order:
        print("VERDICT: TWO CLOCKS.")
        print("   Grokking onset and rank compression are temporally separated AND ordered differently")
        print("   across norm budgets (opposite Spearman signs).")
    elif big_gap and shifted_argmin:
        print("VERDICT: PARTIALLY SEPARATED CLOCKS.")
        print("   Compression lags grokking by a large amount and the argmin rho differs, but")
        print("   the global Spearman ordering is not opposite. Do not claim fully independent clocks.")
    elif big_gap:
        print("VERDICT: ONE CLOCK + LARGE LAG.")
        print("   Compression is delayed after grokking, and the rho-ordering of T_grok and T_compress")
        print("   has the same sign (consistent with a shared clock).")
    else:
        print("VERDICT: SINGLE CLOCK / SMALL LAG.")
        print("   Compression tracks grokking with only a small/modest lag. Do not claim two clocks.")
    print("=" * 96)

    out = os.path.join(args.dir, "compression_clock_v1_5.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("rho,n,n_valid,n_censored,censor_frac,T_grok,eff_grok,eff_floor,drop,T_compress,gap,rel_gap,comp_rate,amp_invalid_frac,no_grok_file,boundary\n")
        for r in rows:
            fh.write(f"{r['rho']:.2f},{r['n']},{r['n_valid']},{r['n_censored']},"
                     f"{r['censor_frac']:.6g},{r['Tg']:.0f},{r['eff_g']:.6g},"
                     f"{r['floor']:.6g},{r['drop']:.6g},{r['Tc']:.0f},"
                     f"{r['gap']:.0f},{r['rel_gap']:.6g},{r['rate']:.8g},"
                     f"{r['amp_invalid_frac']:.6g},{int(r.get('no_grok_file', False))},{int(r.get('boundary', False))}\n")

    seed_out = os.path.join(args.dir, "compression_clock_v1_5_per_seed.csv")
    with open(seed_out, "w", encoding="utf-8") as fh:
        fh.write("rho,seed,T_grok,eff_grok,eff_floor,drop,T_compress,gap,rel_gap,comp_rate,censored,valid_comp\n")
        for r in rows:
            for c in r["raw"]:
                fh.write(f"{r['rho']:.2f},{c['seed']},{c['Tg']:.0f},{c['eff_g']:.6g},"
                         f"{c['floor']:.6g},{c['drop']:.6g},")
                if np.isfinite(c["Tc"]):
                    fh.write(f"{c['Tc']:.0f},{c['gap']:.0f},{c['rel_gap']:.6g},{c['rate']:.8g},"
                             f"{int(c['censored'])},{int(c['valid_comp'])}\n")
                else:
                    fh.write(f"nan,nan,nan,nan,{int(c['censored'])},{int(c['valid_comp'])}\n")

    prov = dict(
        script="analyze_compression_clock_v1_5.py",
        script_version=__version__,
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        dir=args.dir,
        params=dict(eps=args.eps, floor_frac=args.floor_frac, min_drop=args.min_drop,
                    grok_thr=args.grok_thr, censor_warn=args.censor_warn,
                    big_gap_rel=args.big_gap_rel, floor_cv_shared=args.floor_cv_shared),
        files=[os.path.relpath(f, args.dir) for f in files],
        rows=[{k: v for k, v in r.items() if k not in ("raw", "file")} for r in rows],
        summary=dict(
            total_valid=total_valid,
            total_censored=total_cens,
            overall_censor_frac=overall_censor,
            gap_med=float(gap_med) if np.isfinite(gap_med) else None,
            gap_span=float(gap_span) if np.isfinite(gap_span) else None,
            floor_span=float(floor_span) if np.isfinite(floor_span) else None,
            floor_cv=float(floor_cv) if np.isfinite(floor_cv) else None,
            rho_fast_grok=float(rho_fast_grok) if np.isfinite(rho_fast_grok) else None,
            rho_early_comp=float(rho_early_comp) if np.isfinite(rho_early_comp) else None,
            spearman_rho_Tg=float(sp_g) if np.isfinite(sp_g) else None,
            spearman_rho_Tcomp=float(sp_c) if np.isfinite(sp_c) else None,
            spearman_rho_gap=float(sp_gap) if np.isfinite(sp_gap) else None,
        )
    )
    json.dump(prov, open(os.path.join(args.dir, "compression_clock_v1_5.prov.json"), "w"), indent=1, default=float)

    print(f"\nwrote {out}")
    print(f"wrote {seed_out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
