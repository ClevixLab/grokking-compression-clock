# Grokking Compression-Clock Audit

A small, **tested** diagnostic for grokking studies that measure representation structure
(effective rank, spectral entropy, etc.) around the generalization transition.

**The problem it solves.** It is common to measure a representation metric *at* the grokking step
and treat it as a property of the converged generalizing circuit. This can mislead: on modular
arithmetic the embedding effective rank *at grokking* is a **transient** that long training drives
down to a low-rank floor. This audit separates **generalization onset** (`T_grok`) from
**representation compression** (`T_compress`), handles censoring and boundary cells, and refuses to
claim an ordering it cannot support.

It is a **measurement discipline + tooling**, not a new mechanism of grokking. See
`paper/compression_clock_paper.pdf`.

---

## Reproducing the paper

```bash
python reproduce_all.py
```

This runs the test suite, runs the analyzer on all three bundled tasks (mod-add, mod-mult, transformer),
writes the combined summary table (`reproduced/summary_table.csv`), and rebuilds each dashboard — all
from **processed trajectories**, no raw training required.

All of the paper's analysis reproduces out of the box: the adversarial test suite passes, the two MLP
tasks return `PARTIALLY SEPARATED CLOCKS` with the Table 2 numbers (mod-add median gap 18,000,
lag/T_grok 1.00, ρS(ρ,T_grok)=+0.85, ρS(ρ,T_compress)=+0.29; mod-mult median gap 17,000,
lag/T_grok 1.04, ρS(ρ,T_grok)=+0.90, ρS(ρ,T_compress)=+0.18), the mod-add boundary gate excludes
ρ=1.00, and the transformer task correctly returns low-power on its censored cells. The repo reproduces
**analysis from processed trajectories**, not training from scratch.

### What `reproduce_all.py` regenerates

Each table and figure in the paper is regenerated into `reproduced/`. `paper/` holds the canonical
committed figures (the ones in the PDF); running the pipeline **never modifies `paper/`** — it writes
fresh copies to `reproduced/` so you can diff them against the committed versions.

| Paper artifact | Regenerated file (`reproduced/`) | Script |
|---|---|---|
| Figure 1 — mod-add dashboard | `dashboard_mlp_modadd_p59.pdf` | `audit/make_dashboard_figure_v1.py` |
| Figure 2 — at-grok vs. converged | `fig_atgrok_vs_converged.pdf` | `audit/make_atgrok_vs_converged_figure.py` |
| Figure 3 — mod-mult dashboard | `dashboard_mlp_modmult_p59.pdf` | `audit/make_dashboard_figure_v1.py` |
| Figure 4 — transformer transient | `fig_transformer_transient.pdf` | `audit/transformer_sweep_audit.py` |
| Figure 5 — depth-law non-replication | `fig_depthlaw_nonreplication.pdf` | `audit/transformer_sweep_audit.py` |
| Figure 6 — one harness, three architectures | `fig_one_harness_three_arch.pdf` | `audit/arch_generality_audit.py` |
| Table 2 — compression-lag clocks | `summary_table.csv` | `audit/analyze_compression_clock_v1_5.py` |
| Table 3 — sensitivity to free parameters | `sensitivity_table.csv`, `sensitivity_table.tex` | `audit/sensitivity_floor_ci.py` |
| §8 per-arm stats (transient / lag / frac-pre) | `arch_section8_stats.csv` | `audit/arch_generality_audit.py` |

Table 1 is a qualitative positioning table written directly in the paper source (not data-generated).
The adversarial and property test suites in `tests/` are run first and must pass before any number is
trusted.

### Section 8 (Figure 6): the arch-generality harness

Section 8 (one harness, three architectures) is reproduced from the per-cell `metrics.jsonl` of an
arch-generality run. That run ships **slimmed and bundled** at `sample_data/arch_generality/`
(trajectories only, no checkpoints), so `python reproduce_all.py` regenerates Figure 6 automatically —
no extra arguments. To run the step on its own, or on your own run:

```bash
python audit/arch_generality_audit.py \
    --arch-dir sample_data/arch_generality --out reproduced/
```

It re-derives every per-seed quantity (T_grok, at-grok rank, floor, T_compress, lag, frac-pre) from the
raw trajectories with the same clock as the core analyzer, and renders Figure 6:

| Figure 6 panel | what it shows | reproduces from bundled data |
|---|---|---|
| A — free-decay rank trajectories (grok marked) | the transient is architecture-general | ✓ |
| B — frac-pre per arm (0.87 / 0.66 / 0.25) | the lag is normalization-mediated | ✓ |
| C — low-budget MLP second collapse | the converged floor is itself non-stationary | ✓ |

**Section 9 (pre-registered control, future test).** The paper leaves the *mechanism* of the deferral
open and pre-registers a scale-invariance control (an RMS-normalized-embedding arm, `tf_rms`) rather than
asserting it. The pre-registration (`specs/PREREGISTRATION_scale_inv.json`) and its analyzer
(`audit/analyze_scale_invariance.py`) ship so the test can be run when the `tf_rms` arm is added:

```bash
python audit/analyze_scale_invariance.py \
    --arch-dir sample_data/arch_generality --out reproduced/ \
    --prereg specs/PREREGISTRATION_scale_inv.json
```

On the bundled data this correctly reports `CONTROL_ARM_MISSING` (no `tf_rms` arm), and its reproducible
by-product — the post-grok norm–rank coupling, +0.70 canonical vs. −0.80 LayerNorm — matches the paper.
Adding the `tf_rms` arm (one free-decay run with RMS-normalized embeddings, written into the same
run layout) completes the control; no code change is needed.

## Audit your own runs (no re-training)

The audit is not specific to our data. If you have a grokking run of your own, you can check whether your
at-grok reading of effective rank is a transient:

```bash
# You already log effective rank + accuracy per step (one CSV per norm budget):
python audit/analyze_compression_clock_generic_v1.py \
  --metrics-csv run_wd1.csv:1.0 run_wd2.csv:1.2 \
  --eff-col effective_rank --acc-col test_acc --step-col step \
  --out ./audited --clock-script audit/analyze_compression_clock_v1_5.py

# Or you only saved weight checkpoints — it computes effective rank for you:
python audit/analyze_compression_clock_generic_v1.py \
  --checkpoints ./ckpts --matrix-key embedding \
  --acc-csv ./acc.csv --rho 1.0 \
  --out ./audited --clock-script audit/analyze_compression_clock_v1_5.py
```

The adapter only re-expresses your run in the analyzer's schema — it never trains or fabricates data. If
your run is too short to reach the compressed floor, the audit reports censoring rather than guessing.

## What's here

| script | role | what it does |
|---|---|---|
| `audit/analyze_compression_clock_v1_5.py` | **analyzer** (canonical) | All clock logic. Computes `T_grok`, `T_compress`, lag, floor, censoring per norm budget; applies the boundary gate; emits the verdict. Writes CSV + per-seed CSV + provenance JSON. |
| `audit/analyze_compression_clock_existing_npz_v1.py` | **adapter** | Runs the clock on externally-logged runs (externally-logged `metrics/*.npz`); clamp-arm-only dose-response, free control reported separately, 2-point E-vs-U locus check. |
| `tests/test_compression_clock_adversarial_v1.py` | **tests** | Nine pre-registered adversarial cases; checks each verdict *and its reason* (clock verdict must be backed by a finite order statistic). |
| `audit/make_dashboard_figure_v1.py` | **figure** | 4-panel dashboard (trajectories; at-grok vs converged; clock lag; boundary gate) from the run's npz; no hard-coded numbers. |
| `audit/analyze_compression_clock_generic_v1.py` | **third-party adapter** | Audit *anyone's* run: point it at weight checkpoints (computes effective rank per step) or a logged metrics CSV, and it runs the canonical analyzer. No re-training needed. |
| `tests/test_compression_clock_properties_v1.py` | **property tests** | Structural invariants checked over many random draws (no fabricated ordering, non-negative gap, censoring monotonicity, boundary-gate safety, determinism). |
| `reproduce_all.py` | **driver** | Runs tests + analyzer on every task in `sample_data/`, writes the summary table, rebuilds dashboards, then runs the robustness analyses and supplementary figures below. From processed trajectories. |
| `audit/sensitivity_floor_ci.py` | **robustness** | Varies each audit threshold one at a time (sensitivity) and bootstraps the seeds for a CI on the converged-floor law. Emits `sensitivity_table.{csv,tex}` (paper Table 3) and `floor_ci.csv`. |
| `audit/compute_unembedding_rank.py` | **E/U rank** | Computes embedding and unembedding effective rank from logged weight checkpoints (`*_struct.npz`), per seed per checkpoint. Backs the "transient on both loci" claim. |
| `audit/make_floor_ci_figure.py` | **figure** | Converged-floor-vs-budget with bootstrap-over-seeds 95% CI bands, both tasks (paper Fig 2). |
| `audit/make_transformer_eu_figure.py` | **figure** | Embedding vs unembedding effective-rank trajectories for the checkpointed transformer cells (paper Fig 5). |
| `audit/corrected_reading.py` | **robustness** | Reproduces the self-correction: `Spearman(rho, T_compress)` on the small grid (`+0.95`/`+0.40`) vs the full grid (`+0.29`/`+0.18`). |
| `audit/make_atgrok_vs_converged_figure.py` | **figure** | At-grok vs converged effective rank against the norm budget, both tasks (paper Fig 1). Shows the at-grok snapshot overstates the converged floor 3–5× and cannot separate the non-compressing boundary cell from compressing neighbours. |
| `audit/transient_replication.py` | **robustness** | Confirms the central transient (at-grok > converged) under a *free weight-decay* sweep and on *parity* (a third task); also reports, honestly, that the depth law's *direction* is clamp-protocol-specific. |
| `sample_data/transformer_noln_p59/` | **data** | Four real transformer clamp cells (embedding rank), deliberately imperfect (high-budget cells right-censored) so the audit correctly returns low-power; plus `struct/` (E+U weight checkpoints for the two generalizing cells), `metrics_struct/`, and `struct_eu/` (computed E/U rank trajectories). |
| `sample_data/mlp_grid_small/` | **data** | The earlier small budget grid (5 add + 4 mult cells) used by `corrected_reading.py` to reproduce the pre-enlargement ordering. |
| `sample_data/mlp_freewd_p59/` | **data** | Free weight-decay sweep at p=59 on three tasks (modular addition, multiplication, parity), 6 decay strengths × 3 seeds, used by `transient_replication.py`. |
| `figures/` | output | Dashboard figures (mod-add, mod-mult, transformer) + `floor_law_ci` + `fig_transformer_eu`, in pdf + png. |
| `paper/` | writeup | `compression_clock_paper.{tex,pdf}` + the embedded figures + `sensitivity_table.tex`. |
| `CHANGELOG.md` | ledger | Released component versions and the reproducibility-relevant changes between them. |

```
audit/        analyzer + adapters + figure generators + robustness scripts
tests/        adversarial + property test suites
sample_data/  modular-arithmetic MLP grids (full + small) and transformer clamp cells
              (incl. E/U weight checkpoints), all runnable examples
figures/      dashboards + floor-law CI + transformer E/U figures
paper/        the write-up + embedded figures + generated sensitivity table
CHANGELOG.md  component versions + release notes
```

## Install

```bash
pip install -r requirements.txt   # numpy, matplotlib, pandas
```

Python 3.9+.

---

## Expected input format

The analyzer reads `metrics/longtrain_*rho*.npz`, one file per norm-budget cell, with keys:

| key   | shape          | meaning                                  |
|-------|----------------|------------------------------------------|
| `rho` | scalar         | norm budget (ρ = ‖W‖ / ‖W‖_c)            |
| `steps` | `(T,)`       | logged training steps                    |
| `eff` | `(seeds, T)`   | effective rank (or any complexity metric) per seed over steps |
| `acc` | `(seeds, T)`   | test accuracy per seed over steps        |

`T_grok_per_seed` is optional. The metric in `eff` need not be effective rank — any scalar that
*falls* as the representation compresses works (spectral entropy, participation ratio, …).

If your runs use a different schema (e.g. keys `eff_rank_E`, `test_acc`, `arm` as in weight-norm
delay-law runs), use the **adapter** below instead.

---

## Quick start

### 1. Run the audit on long-train data

```bash
python audit/analyze_compression_clock_v1_5.py --dir /path/to/run_dir
```

where `run_dir/metrics/` holds the `longtrain_*rho*.npz` files. It prints a per-cell table
(`T_grok`, `T_compress`, gap, floor, censoring), a boundary-gate report, and one of the verdicts
below. It also writes `compression_clock_v1_5.csv`, a per-seed CSV, and a `.prov.json` provenance file.

### 2. Run on externally-logged runs

```bash
python audit/analyze_compression_clock_existing_npz_v1.py \
    --dir /path/to/your_run \
    --clock_script audit/analyze_compression_clock_v1_5.py
```

This adapter (a) re-emits the dense `eff_rank_E`/`test_acc` trajectory as a compression clock on the
**clamp** arm only (the free control is reported separately, never mixed into the dose-response), and
(b) does a 2-point **E-vs-U locus check** from structural checkpoints, to tell a genuine
not-yet-compressed cell from a wrong-locus measurement.

### 3. Make the dashboard figure

```bash
python audit/make_dashboard_figure_v1.py --dir /path/to/run_dir --task "mod-add" --out figures/dashboard_modadd.pdf
```

No numbers are hard-coded; the figure is built entirely from the npz trajectories.

### 4. Try it on the bundled sample data

```bash
python audit/analyze_compression_clock_v1_5.py --dir sample_data/transformer_noln_p59
```

The sample is **real** data: four clamp cells from an un-normalized transformer run. It is included
*because* it is imperfect — the high-budget cells grok very late and have little post-grok budget, so
after the boundary gate fewer than two cells remain compressed and the audit correctly returns
**`LOW-POWER / DESCRIPTIVE ONLY`** instead of inventing a clock. That is the intended behavior, and
the point of the tool: it declines when the data do not support a verdict.

---

## Verdicts

| verdict | meaning |
|---|---|
| `ONE CLOCK + LARGE LAG` | compression lags grokking, but `T_grok` and `T_compress` order the same way across ρ (a shared clock). |
| `TWO CLOCKS` | large lag **and** opposite ρ-ordering of the two times. Rarely warranted; the audit is conservative about it. |
| `PARTIALLY SEPARATED CLOCKS` | large lag, argmin-ρ differs, but global ordering is not opposite. |
| `LARGE LAG, ORDERING UNDETERMINED` | a real lag, but the ρ-ordering statistic is undefined (too few cells / ties). The audit reports the lag only and does **not** claim a clock count. |
| `SINGLE CLOCK / SMALL LAG` | compression tracks grokking closely. |
| `INCONCLUSIVE due to censoring` | too many seeds never reached the compression threshold within budget. |
| `LOW-POWER / DESCRIPTIVE ONLY` | fewer than two non-boundary cells survive gating. |

The **boundary gate** excludes cells that did not fully generalize (high floor, tiny drop) *before*
any ordering statistic is computed — this is what prevents a partially-generalizing floor cell from
manufacturing a false `TWO CLOCKS`.

---

## Tests

```bash
python tests/test_compression_clock_adversarial_v1.py --clock_script audit/analyze_compression_clock_v1_5.py
```

Nine adversarial synthetic cases, each with a **pre-registered** expected behavior. The suite checks
the *verdict line* (not just string presence) and, crucially, that any clock-type verdict is backed by
a **finite order statistic** — a clock claim printed next to a `NaN` Spearman is a failure. This guard
caught a real regression during development (see `CHANGELOG.md`).

---

## Known failure modes / scope

- **Needs post-grok coverage.** A reliable `T_compress` needs several checkpoints *after* `T_grok`.
  If cells grok very late relative to the budget, they are censored — the audit flags this rather than
  guessing. (This is exactly what the sample data shows.)
- **Locus on transformers.** On a transformer, the right locus for compression may be the unembedding,
  not the token embedding; use the adapter's E-vs-U locus check.
- **Not a mechanism.** The audit measures *that* compression lags grokking; it does not explain *why*.
- **Metric borrowed, not introduced.** Effective rank / spectral entropy are from prior work; the
  contribution is the audit and the discipline around the measurement.

## Roadmap / future development

Planned, low-cost extensions (contributions welcome):

- **Same-run multi-clock.** Add persistent-homology, LID, and Fourier-Gini clocks so rank, topology, and feature clocks sit on one timeline from a single run — the experiment that would settle whether rank compression and topological reorganization are one event or two.
- **Checkpoint-density guidance.** An ablation that subsamples post-grok checkpoints to report how many are needed for a reliable `T_compress`, producing a concrete logging recommendation.
- **Metric-agnostic clocks.** The clock needs only a quantity that falls as the representation compresses; spectral entropy, participation ratio, or stable rank can be dropped into the same input schema.
- **Per-locus transformer clocks.** Promote the 2-point E-vs-U locus check into a full per-locus clock, since the compression locus on a transformer may be the unembedding.
- **Forward/backward coupling.** Pair the audit with a forward transition detector so a pipeline can both predict a transition and certify whether a measurement taken at it has converged.

## Citation

If you use this audit or its findings, please cite the paper. A machine-readable `CITATION.cff` is
included (GitHub's "Cite this repository" reads it):

```bibtex
@article{truong2026atgrok,
  title  = {At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics},
  author = {Truong, Xuan Khanh and Luu, Duc Trung},
  year   = {2026}
  % journal/eprint = {arXiv:XXXX.XXXXX}  % add once assigned
}
```

The compiled paper is at `paper/compression_clock_paper.pdf`; its full source (with the bundled figures)
is in `paper/`.

## License

MIT — see `LICENSE`.
