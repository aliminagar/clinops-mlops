# ClinOps — Model Results

> **Synthetic data.** All numbers below come from a **synthetic** Synthea cohort
> (no real patients, no PHI). They are reproduced from a real training run, not
> hand-edited — but see the [integrity caveat](#integrity-caveat) before reading
> them as real-world performance.

## Run provenance

| | |
|---|---|
| Cohort | 5,626 synthetic patients (Synthea, `-p 5000 -s 42`) |
| Target | `readmission_30d` (30-day inpatient readmission) |
| Prevalence | **3.54%** positive (199 / 5,626) — heavy class imbalance |
| Split | stratified 60/20/20 → train 3,375 · val 1,125 · test 1,126 |
| Features | 12 (3 zero-variance dropped: `age_missing`, `cond_copd`, `systolic_bp_missing`) |
| Seed | 42 (deterministic) |
| Imbalance handling | class weighting only (no SMOTE/resampling) |
| Headline metric | **PR-AUC** (average precision) — appropriate for a rare positive |

## Three-model comparison

All three candidates train on the **same** features, split, and CV protocol —
apples-to-apples. Headline (threshold-free) metrics are reported once on the
held-out **test** fold, with 5-fold cross-validated PR-AUC on the **train** fold
(mean ± std):

| Model | CV PR-AUC (train) | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| **Logistic regression** (`class_weight=balanced`) — *champion* | **0.325 ± 0.068** | 0.395 | 0.900 |
| LightGBM (`scale_pos_weight≈27.4`) | 0.319 ± 0.064 | 0.416 | 0.882 |
| PyTorch MLP (`pos_weight≈27.4`) | 0.257 ± 0.122 | 0.387 | 0.854 |

Against the no-skill PR-AUC baseline (= prevalence, **0.035**), the champion's
0.395 test PR-AUC is a **~11.2× lift** (LightGBM's 0.416 is ~11.7×).

### Champion: logistic regression (registry v3)

Promotion is decided on **CV PR-AUC** (chosen before any test-fold peeking), where
logistic regression edged LightGBM (0.325 vs 0.319). It is registered as
**`clinops-readmission-classifier` v3** with the `champion` alias, and the
champion **changed from LightGBM (v2) → logistic regression (v3)** on this run.

Honest note: LightGBM actually scored *higher* on the held-out **test** PR-AUC
(0.416 vs 0.395). The two are within a standard deviation of each other on CV, and
we promote on CV — not on the test fold — to avoid selecting on the number we
report. So the promotion is correct by protocol even though the test ranking
differs.

The PyTorch challenger competed apples-to-apples and placed **third** (CV PR-AUC
0.257, with the highest variance ±0.122 on this small synthetic signal). It is
load-bearing in that it *could* win the promotion — not that it did here.

## Operating points — champion (test fold)

Three thresholds are reported side by side. Tuned thresholds are selected on the
**validation** fold only and applied **once** to test (no leakage). The
**deployed** operating point is **F2** (recall-weighted) — see
[why F2](#why-f2-is-the-default).

| Operating point | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Fixed @0.5 | 0.500 | 0.146 | 0.775 | 0.246 |
| Max-F1 | 1.000 | 1.000 | 0.225 | 0.367 |
| **F2 (deployed)** | **0.813** | 0.198 | **0.425** | 0.270 |

The deployed F2 threshold (~0.813) trades precision for recall versus Max-F1, as
intended for screening. For reference, the F2 **recall** of each candidate is
0.425 (logreg) · 0.525 (LightGBM) · 0.750 (PyTorch) — the more flexible models
reach more positives at the cost of precision.

## Why F2 is the default

For 30-day readmission **screening**, a false negative (a missed readmission the
care team never flags) is costlier than a false positive (an extra follow-up
call). The deployed threshold is therefore the one maximizing **F2** (recall
weighted twice as heavily as precision; `threshold_beta = 2` in
[`config.py`](../src/clinops/config.py)). The fixed-0.5 and max-F1 points are kept
alongside for context, never replaced. To deploy at F1 instead, set
`threshold_beta = 1.0`.

## Integrity caveat

These results are on **synthetic** data and should **not** be read as real-world
performance:

- **Synthetic separability is inflated.** Synthea generates patients from
  rule-based care pathways, so the signal is cleaner and more learnable than real
  EHR data. Published 30-day readmission models on real cohorts typically land
  around **0.65–0.75 ROC-AUC**; the **0.85–0.90 ROC-AUC** here reflects that
  synthetic inflation, not a better model.
- **Read PR-AUC against the base rate, not absolute.** At a 3.5% positive rate a
  trivial classifier scores ~0.035 PR-AUC, so the ~0.40 here is meaningful as a
  **~11× lift** — but a ~0.40 PR-AUC is not directly comparable to a balanced
  problem.
- **The identical Max-F1 point for logistic regression and LightGBM**
  (P=1.000, R=0.225, F1=0.367) is **not a copy-paste error.** It is a
  synthetic-data score-clustering artifact: both models push the rare positives
  to probability scores bunched near 1.0, so their F1-maximizing thresholds land
  on the same handful of highest-confidence cases. The PyTorch model, which
  spreads its scores differently, lands a distinct Max-F1 point (P=0.611,
  R=0.275); and every model's F2 point, reaching lower-confidence cases, diverges
  as expected.

Reproduce with `python -m clinops.training.train`; metrics are written to
[`reports/metrics.json`](../reports/metrics.json) and logged to MLflow.
