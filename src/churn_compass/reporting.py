"""Build concise reports and auditable acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_calibration_summary(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return pd.DataFrame(
        {
            "calibration_bin": np.arange(1, len(observed) + 1),
            "mean_predicted_churn_probability": predicted,
            "observed_churn_rate": observed,
            "absolute_gap": np.abs(predicted - observed),
            "evaluation_scope": "Locked holdout; diagnostic only.",
        }
    )


def build_validation_checks(
    quality: pd.DataFrame,
    model_metrics: dict[str, Any],
    targeting: pd.DataFrame,
    minimum_auc: float = 0.78,
    minimum_ap_lift: float = 1.5,
    minimum_top20_churn_capture_gain_pp: float = 5.0,
) -> pd.DataFrame:
    risk_20 = targeting.loc[
        targeting["policy"].eq("risk_only") & targeting["contact_budget_pct"].eq(20.0),
        "share_test_churners_captured",
    ]
    rule_20 = targeting.loc[
        targeting["policy"].eq("month_to_month_then_tenure") & targeting["contact_budget_pct"].eq(20.0),
        "share_test_churners_captured",
    ]
    risk_capture = float(risk_20.iloc[0]) if len(risk_20) else float("nan")
    rule_capture = float(rule_20.iloc[0]) if len(rule_20) else float("nan")
    capture_gain_pp = (risk_capture - rule_capture) * 100
    ap_lift = float(model_metrics["average_precision"] / model_metrics["average_precision_baseline"])
    checks = [
        ("data_quality_checks", not (quality["status"] == "FAIL").any(), "No failed structural data checks."),
        ("holdout_evaluation_present", bool(model_metrics.get("test_evaluated_once")), "Selected model evaluated after development-only selection."),
        ("roc_auc_threshold", model_metrics["roc_auc"] >= minimum_auc, f"Project gate: ROC-AUC >= {minimum_auc:.2f}; observed {model_metrics['roc_auc']:.3f}."),
        ("average_precision_lift", ap_lift >= minimum_ap_lift, f"Project gate: average precision / churn prevalence >= {minimum_ap_lift:.1f}x; observed {ap_lift:.2f}x."),
        (
            "top20_crm_capture_gain",
            capture_gain_pp >= minimum_top20_churn_capture_gain_pp,
            f"Project gate: risk ranking captures >= {minimum_top20_churn_capture_gain_pp:.1f} percentage points more churners than the simple rule at 20% capacity; observed {capture_gain_pp:+.1f} pp.",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "PASS" if bool(passed) else "FAIL", "details": details}
            for name, passed, details in checks
        ]
    )


def build_run_report(
    metrics: dict[str, Any],
    model_comparison: pd.DataFrame,
    targeting: pd.DataFrame,
    checks: pd.DataFrame,
    run_metadata: dict[str, Any],
) -> str:
    risk20 = targeting[(targeting["policy"] == "risk_only") & (targeting["contact_budget_pct"] == 20.0)].iloc[0]
    rule20 = targeting[(targeting["policy"] == "month_to_month_then_tenure") & (targeting["contact_budget_pct"] == 20.0)].iloc[0]
    value20 = targeting[(targeting["policy"] == "value_weighted") & (targeting["contact_budget_pct"] == 20.0)].iloc[0]
    failed = checks.loc[checks["status"] == "FAIL", "check"].tolist()
    status = "PASS" if not failed else "FAIL"
    cv_rows = model_comparison[["model", "average_precision_mean", "average_precision_sd", "roc_auc_mean", "roc_auc_sd"]].to_string(
        index=False, float_format=lambda value: f"{value:.3f}"
    )
    gain_pp = risk20["churn_capture_gain_vs_simple_rule_pp"]
    return f"""# ChurnCompass Run Report

**Execution gate:** {status}  
**Selected model:** {metrics['selected_model']}  
**Locked-test records:** {metrics['test_rows']:,}  
**Source data SHA-256:** `{run_metadata['dataset_sha256']}`  
**Run timestamp (UTC):** {run_metadata['executed_at_utc']}  
**Scope:** retrospective churn-status classification and contact-priority simulation.

## Locked-test model results

| Metric | Result | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | {metrics['roc_auc']:.3f} | {metrics['roc_auc_ci_95_low']:.3f}–{metrics['roc_auc_ci_95_high']:.3f} |
| Average precision | {metrics['average_precision']:.3f} | {metrics['average_precision_ci_95_low']:.3f}–{metrics['average_precision_ci_95_high']:.3f} |
| Test churn prevalence (AP reference) | {metrics['average_precision_baseline']:.3f} | — |
| Average precision / prevalence | {metrics['average_precision'] / metrics['average_precision_baseline']:.2f}× | — |
| Brier score (lower is better) | {metrics['brier_score']:.3f} | {metrics['brier_score_ci_95_low']:.3f}–{metrics['brier_score_ci_95_high']:.3f} |
| Log loss (lower is better) | {metrics['log_loss']:.3f} | — |

The probability threshold of 0.50 is reported only as a reference. CRM targeting uses ranking and contact-budget comparisons instead.

## Development-set model comparison

All rows below use development-only repeated stratified 5-fold cross-validation (two repeats). They are not locked-test results.

```text
{cv_rows}
```

## CRM prioritization at a 20% contact budget

| Strategy | Share of test churners captured | Share of churner-linked monthly charges captured | Churn rate among selected customers |
|---|---:|---:|---:|
| Risk only | {risk20['share_test_churners_captured']:.1%} | {risk20['share_churner_linked_monthly_charges_captured']:.1%} | {risk20['selected_customer_churn_rate']:.1%} |
| Risk × monthly charges | {value20['share_test_churners_captured']:.1%} | {value20['share_churner_linked_monthly_charges_captured']:.1%} | {value20['selected_customer_churn_rate']:.1%} |
| Month-to-month first, then shorter tenure | {rule20['share_test_churners_captured']:.1%} | {rule20['share_churner_linked_monthly_charges_captured']:.1%} | {rule20['selected_customer_churn_rate']:.1%} |

Risk ranking's churn-capture difference versus the simple rule is **{gain_pp:+.1f} percentage points** at the 20% budget.

## Acceptance checks

{checks.to_markdown(index=False)}

## Study limitations

- The dataset is a cross-sectional snapshot with a churn-status label, not a time-stamped next-month prediction target.
- The dataset was used in an earlier exploratory feasibility review. The current holdout is protected from model tuning in this pipeline, but it is not independent external validation of the project or data source.
- The contact-budget analysis is a retrospective ranking simulation, not a randomized retention experiment.
- Monthly charges are an exposure proxy, not profit, customer lifetime value, saved revenue, or ROI.
- The dataset has no treatment records; the analysis cannot estimate causal retention lift.
- Feature importance and segment differences are descriptive associations, not causal effects.
- The scored queue includes ``observed_churn`` strictly for retrospective evaluation. It must not be used as an input in a live scoring workflow.

"""
