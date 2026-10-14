# ChurnCompass Run Report

**Execution gate:** PASS  
**Selected model:** logistic_regression  
**Locked-test records:** 1,409  
**Source data SHA-256:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`  
**Run timestamp (UTC):** 2026-10-14T10:55:30+00:00  
**Scope:** retrospective churn-status classification and contact-priority simulation.

## Locked-test model results

| Metric | Result | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.854 | 0.831–0.876 |
| Average precision | 0.693 | 0.641–0.734 |
| Test churn prevalence (AP reference) | 0.265 | — |
| Average precision / prevalence | 2.61× | — |
| Brier score (lower is better) | 0.132 | 0.123–0.143 |
| Log loss (lower is better) | 0.407 | — |

The probability threshold of 0.50 is reported only as a reference. CRM targeting uses ranking and contact-budget comparisons instead.

## Development-set model comparison

All rows below use development-only repeated stratified 5-fold cross-validation (two repeats). They are not locked-test results.

```text
              model  average_precision_mean  average_precision_sd  roc_auc_mean  roc_auc_sd
           lightgbm                   0.652                 0.040         0.844       0.014
      random_forest                   0.646                 0.042         0.842       0.016
logistic_regression                   0.644                 0.046         0.842       0.018
     prior_baseline                   0.265                 0.000         0.500       0.000
```

## CRM prioritization at a 20% contact budget

| Strategy | Share of test churners captured | Share of churner-linked monthly charges captured | Churn rate among selected customers |
|---|---:|---:|---:|
| Risk only | 51.1% | 55.0% | 67.7% |
| Risk × monthly charges | 51.1% | 58.6% | 67.7% |
| Month-to-month first, then shorter tenure | 41.2% | 35.0% | 54.6% |

Risk ranking's churn-capture difference versus the simple rule is **+9.9 percentage points** at the 20% budget.

## Acceptance checks

| check                      | status   | details                                                                                                                            |
|:---------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------|
| data_quality_checks        | PASS     | No failed structural data checks.                                                                                                  |
| holdout_evaluation_present | PASS     | Selected model evaluated after development-only selection.                                                                         |
| roc_auc_threshold          | PASS     | Project gate: ROC-AUC >= 0.78; observed 0.854.                                                                                     |
| average_precision_lift     | PASS     | Project gate: average precision / churn prevalence >= 1.5x; observed 2.61x.                                                        |
| top20_crm_capture_gain     | PASS     | Project gate: risk ranking captures >= 5.0 percentage points more churners than the simple rule at 20% capacity; observed +9.9 pp. |

## Study limitations

- The dataset is a cross-sectional snapshot with a churn-status label, not a time-stamped next-month prediction target.
- The dataset was used in an earlier exploratory feasibility review. The current holdout is protected from model tuning in this pipeline, but it is not independent external validation of the project or data source.
- The contact-budget analysis is a retrospective ranking simulation, not a randomized retention experiment.
- Monthly charges are an exposure proxy, not profit, customer lifetime value, saved revenue, or ROI.
- The dataset has no treatment records; the analysis cannot estimate causal retention lift.
- Feature importance and segment differences are descriptive associations, not causal effects.
- The scored queue includes `observed_churn` strictly for retrospective evaluation. It must not be used as an input in a live scoring workflow.
