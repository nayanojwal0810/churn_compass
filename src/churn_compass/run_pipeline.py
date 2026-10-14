"""Run the full ChurnCompass analysis from the command line."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before importing numerical libraries to keep runtime predictable on laptops/CI.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env] = "1"

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from churn_compass.data import (
    DATASET_URL,
    EXPECTED_SHA256,
    ensure_dataset_available,
    load_and_validate_data,
)
from churn_compass.modeling import (
    RANDOM_STATE,
    compare_models,
    evaluate_locked_test,
    export_feature_importance,
    fit_primary_model,
    select_primary_model,
)
from churn_compass.prioritization import (
    build_segment_summary,
    evaluate_policies,
    rank_test_customers,
)
from churn_compass.reporting import (
    build_calibration_summary,
    build_run_report,
    build_validation_checks,
    write_json,
)

DEFAULT_DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Telco dataset, compare churn models, and evaluate CRM contact policies."
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the Telco CSV. The canonical file is downloaded if the default path is missing.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for reproducible CSV/JSON outputs and the run report.",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Locked test fraction (default: 0.20).")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Fixed split seed.")
    parser.add_argument(
        "--allow-alternative-data",
        action="store_true",
        help="Allow a schema-compatible non-canonical CSV at the default input path; schema checks still apply.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    executed_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.data_path)
    is_default_path = input_path.as_posix() == DEFAULT_DATA_PATH.as_posix()

    try:
        ensure_dataset_available(input_path, allow_download=is_default_path)
        data = load_and_validate_data(
            input_path,
            test_size=args.test_size,
            random_state=args.random_state,
            verify_canonical_checksum=is_default_path and not args.allow_alternative_data,
        )

        print("[1/5] Data validation and stratified holdout complete.", flush=True)
        print("[2/5] Comparing candidate models with stratified 5-fold CV...", flush=True)
        cv_summary = compare_models(data.X_development, data.y_development)
        selected_model_name, selection_reason = select_primary_model(cv_summary)
        print(f"[3/5] Selected {selected_model_name}; fitting on development records...", flush=True)
        selected_model = fit_primary_model(
            selected_model_name, data.X_development, data.y_development
        )

        # The holdout is scored after all model-selection decisions are complete.
        print("[4/5] Evaluating the selected model on the reserved holdout...", flush=True)
        test_metrics, test_probabilities = evaluate_locked_test(
            selected_model_name, selected_model, data.X_test, data.y_test
        )
        test_metrics.update(
            {
                "selection_reason": selection_reason,
                "development_rows": int(len(data.y_development)),
                "development_churn_prevalence": float(data.y_development.mean()),
                "split_random_state": int(args.random_state),
                "test_fraction": float(args.test_size),
                "dataset_sha256": data.dataset_sha256,
                "executed_at_utc": executed_at_utc,
            }
        )

        comparison = cv_summary.copy()
        comparison["selection_scope"] = "development records only; repeated stratified 5-fold cross-validation"
        comparison["selected_primary_model"] = comparison["model"].eq(selected_model_name)
        comparison["selection_reason"] = np.where(
            comparison["selected_primary_model"],
            selection_reason,
            "Not selected by the development-only rule.",
        )

        # Observed labels are exported solely for retrospective evaluation and are not features.
        test_rows = data.cleaned.loc[data.test_index].copy()
        scored_test = rank_test_customers(test_rows, test_probabilities)
        targeting = evaluate_policies(scored_test)
        segments = build_segment_summary(scored_test)
        calibration = build_calibration_summary(data.y_test, test_probabilities)
        feature_importance = export_feature_importance(selected_model_name, selected_model)
        validation = build_validation_checks(data.quality, test_metrics, targeting)

        print("[5/5] Exporting CRM comparisons, diagnostics, and validation report...", flush=True)
        output_frames = {
            "data_quality_checks.csv": data.quality,
            "model_comparison.csv": comparison,
            "targeting_policy_comparison.csv": targeting,
            "customer_priority_queue.csv": scored_test,
            "segment_performance.csv": segments,
            "model_feature_importance.csv": feature_importance,
            "calibration_summary.csv": calibration,
            "validation_checks.csv": validation,
        }
        for filename, frame in output_frames.items():
            frame.to_csv(output_dir / filename, index=False, float_format="%.8g")
        write_json(output_dir / "model_evaluation.json", test_metrics)

        validation_failed = validation["status"].eq("FAIL").any()
        metadata = {
            "project": "ChurnCompass",
            "status": "FAIL" if validation_failed else "PASS",
            "selected_model": selected_model_name,
            "selection_reason": selection_reason,
            "input_path": str(input_path),
            "executed_at_utc": executed_at_utc,
            "input_source_url": DATASET_URL if is_default_path else None,
            "expected_canonical_sha256": EXPECTED_SHA256 if is_default_path else None,
            "dataset_sha256": data.dataset_sha256,
            "raw_rows": int(len(data.cleaned)),
            "raw_columns": int(len(data.raw.columns)),
            "development_rows": int(len(data.X_development)),
            "test_rows": int(len(data.X_test)),
            "development_churn_prevalence": float(data.y_development.mean()),
            "test_churn_prevalence": float(data.y_test.mean()),
            "features_used": list(data.X_development.columns),
            "identifier_excluded_from_model": "customerID",
            "target_excluded_from_features": "Churn",
            "model_selection_scope": "development records only",
            "final_test_scope": "one fixed holdout evaluation after model selection; not independent external validation",
            "contact_budgets_pct": [10, 20, 30],
            "acceptance_thresholds": {
                "roc_auc_minimum": 0.78,
                "average_precision_to_prevalence_minimum": 1.5,
                "risk_only_churn_capture_gain_vs_simple_rule_at_20pct_minimum_pp": 5.0,
            },
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "lightgbm_version": lightgbm.__version__,
            "outputs": list(output_frames.keys()) + [
                "model_evaluation.json",
                "run_summary.json",
                "run_report.md",
            ],
            "limitations": [
                "The target describes churn status in a cross-sectional snapshot; it is not a next-month forecast.",
                "This dataset was used in an earlier exploratory feasibility review; the fixed holdout is not independent external validation.",
                "The contact-budget analysis is a retrospective ranking simulation, not a randomized retention experiment.",
                "MonthlyCharges is an exposure proxy, not profit, lifetime value, saved revenue, or ROI.",
                "The data contains no retention-treatment outcomes, so causal retention lift cannot be estimated.",
                "Observed churn in customer_priority_queue.csv is for retrospective evaluation only and must not enter live scoring features.",
            ],
        }
        metadata["runtime_seconds"] = round(time.perf_counter() - started, 2)
        write_json(output_dir / "run_summary.json", metadata)
        report = build_run_report(test_metrics, comparison, targeting, validation, metadata)
        (output_dir / "run_report.md").write_text(report, encoding="utf-8")

        print(f"Dataset: {len(data.cleaned):,} rows x {len(data.raw.columns)} columns")
        print(f"Dataset SHA-256: {data.dataset_sha256}")
        print(f"Selected model: {selected_model_name}")
        print(f"Selection reason: {selection_reason}")
        print(
            "Locked-test metrics: "
            f"ROC-AUC={test_metrics['roc_auc']:.3f} "
            f"(95% CI {test_metrics['roc_auc_ci_95_low']:.3f}-{test_metrics['roc_auc_ci_95_high']:.3f}), "
            f"AP={test_metrics['average_precision']:.3f} "
            f"(95% CI {test_metrics['average_precision_ci_95_low']:.3f}-{test_metrics['average_precision_ci_95_high']:.3f}), "
            f"Brier={test_metrics['brier_score']:.3f}"
        )
        risk20 = targeting[(targeting["policy"] == "risk_only") & (targeting["contact_budget_pct"] == 20.0)].iloc[0]
        value20 = targeting[(targeting["policy"] == "value_weighted") & (targeting["contact_budget_pct"] == 20.0)].iloc[0]
        rule20 = targeting[(targeting["policy"] == "month_to_month_then_tenure") & (targeting["contact_budget_pct"] == 20.0)].iloc[0]
        print("20% contact-budget comparison:")
        for row in (risk20, value20, rule20):
            print(
                f"  {row['policy']}: captures {row['share_test_churners_captured']:.1%} of test churners; "
                f"captures {row['share_churner_linked_monthly_charges_captured']:.1%} of churner-linked monthly charges"
            )
        print("Acceptance checks:")
        print(validation.to_string(index=False))
        print(f"Runtime: {metadata['runtime_seconds']:.1f} seconds")
        print(f"Outputs written to: {output_dir.resolve()}")
        return 2 if validation_failed else 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
