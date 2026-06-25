# Changelog

Versions of the released components and the reproducibility-relevant changes between them.
The paper is the canonical description of the method; this file tracks the code and data.

## Component versions (current release)

| component | file | version |
|---|---|---|
| canonical clock analyzer | `audit/analyze_compression_clock_v1_5.py` | 1.5.0 |
| external-npz adapter | `audit/analyze_compression_clock_existing_npz_v1.py` | 1.0.0 |
| generic third-party adapter | `audit/analyze_compression_clock_generic_v1.py` | 1.0.0 |
| dashboard figure generator | `audit/make_dashboard_figure_v1.py` | 1.0.0 |
| transformer dose-response audit | `audit/transformer_sweep_audit.py` | 1.0.0 |
| architecture-generality audit (§8) | `audit/arch_generality_audit.py` | 1.0.0 |
| scale-invariance control (§9) | `audit/analyze_scale_invariance.py` | 1.0.0 |
| sensitivity + floor-CI | `audit/sensitivity_floor_ci.py` | 1.0.0 |
| unembedding-rank computation | `audit/compute_unembedding_rank.py` | 1.0.0 |
| property test suite | `tests/test_compression_clock_properties_v1.py` | 1.0.0 |
| adversarial test suite | `tests/test_compression_clock_adversarial_v1.py` | 1.1.0 |
| reproduce driver | `reproduce_all.py` | 1.5.0 |

## Release notes

### Current
- Added the architecture-generality audit (`arch_generality_audit.py`) that reproduces Figure 6 and the
  section-8 per-arm statistics (transient, lag/T_grok, frac-pre) from the bundled `sample_data/arch_generality`.
- Added the pre-registered scale-invariance control (`analyze_scale_invariance.py`) plus its
  pre-registration (`specs/PREREGISTRATION_scale_inv.json`); it reports `CONTROL_ARM_MISSING` on the
  current data by design (the `tf_rms` arm is a future run).
- `reproduce_all.py` now reproduces every figure with a single command (auto-detects the bundled section-8
  data) and writes all regenerated figures to `reproduced/`, never modifying `paper/`.

### Earlier
- Two-clock analyzer (`analyze_compression_clock_v1_5.py`): separates generalization onset (`T_grok`)
  from representation compression (`T_compress`), flags censoring, excludes boundary cells that did not
  fully generalize, and declines an ordering verdict when the order statistic is undefined.
- Adversarial + property test suites added; the adversarial suite caught and fixed false-confidence
  regressions in the analyzer before any number was reported.
- Sensitivity grid and bootstrap floor confidence intervals added.
- Claim scoping: the norm-budget depth law is reported as an MLP-specific, protocol-specific negative
  result on generality, confirmed by the transformer dose-response.
