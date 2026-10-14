from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_compass.data import load_and_validate_data
from churn_compass.modeling import select_primary_model
from churn_compass.prioritization import evaluate_policies, rank_test_customers


def make_dataset(n: int = 1_000) -> pd.DataFrame:
    """Make a deterministic schema-compatible fixture without network access."""
    index = np.arange(n)
    tenure = index % 73
    tenure[:11] = 0  # Reproduce the canonical zero-tenure blank-charge edge case.
    monthly_charges = 25.0 + (index % 90) * 0.75
    total_charges = tenure * monthly_charges
    total_charges = total_charges.astype(float).astype(object)
    for row in range(min(11, n)):
        if tenure[row] == 0:
            total_charges[row] = " "
    return pd.DataFrame(
        {
            "customerID": [f"TEST-{row:05d}" for row in index],
            "gender": np.where(index % 2 == 0, "Female", "Male"),
            "SeniorCitizen": index % 2,
            "Partner": np.where(index % 3 == 0, "Yes", "No"),
            "Dependents": np.where(index % 4 == 0, "Yes", "No"),
            "tenure": tenure,
            "PhoneService": np.where(index % 5 == 0, "No", "Yes"),
            "MultipleLines": np.where(index % 5 == 0, "No phone service", np.where(index % 2 == 0, "Yes", "No")),
            "InternetService": np.select([index % 3 == 0, index % 3 == 1], ["DSL", "Fiber optic"], default="No"),
            "OnlineSecurity": np.where(index % 3 == 0, "Yes", "No internet service"),
            "OnlineBackup": np.where(index % 3 == 1, "Yes", "No internet service"),
            "DeviceProtection": np.where(index % 3 == 2, "Yes", "No internet service"),
            "TechSupport": np.where(index % 2 == 0, "Yes", "No"),
            "StreamingTV": np.where(index % 2 == 0, "Yes", "No"),
            "StreamingMovies": np.where(index % 3 == 1, "Yes", "No"),
            "Contract": np.select([index % 3 == 0, index % 3 == 1], ["Month-to-month", "One year"], default="Two year"),
            "PaperlessBilling": np.where(index % 2 == 0, "Yes", "No"),
            "PaymentMethod": np.select([index % 4 == 0, index % 4 == 1, index % 4 == 2], ["Electronic check", "Mailed check", "Bank transfer (automatic)"], default="Credit card (automatic)"),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": np.where((index % 4 == 0) | (tenure <= 3), "Yes", "No"),
        }
    )


def test_data_contract_and_split_are_reproducible(tmp_path):
    path = tmp_path / "customers.csv"
    make_dataset().to_csv(path, index=False)
    first = load_and_validate_data(path)
    second = load_and_validate_data(path)

    assert len(first.cleaned) == 1_000
    assert first.cleaned["customerID"].is_unique
    assert "customerID" not in first.X_development.columns
    assert "Churn" not in first.X_development.columns
    assert first.test_index.equals(second.test_index)
    assert set(first.y_test.unique()) == {0, 1}
    assert int(first.quality.loc[first.quality["check"] == "blank_total_charges_after_trim", "value"].iloc[0]) == 11
    assert first.dataset_sha256 == second.dataset_sha256


def test_rankings_and_policy_budgets_are_valid():
    sample = pd.DataFrame(
        {
            "customerID": [f"C{i:02}" for i in range(1, 11)],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "One year", "Month-to-month", "Two year", "One year", "Month-to-month", "Two year"],
            "tenure": [1, 20, 3, 50, 6, 2, 72, 13, 4, 36],
            "MonthlyCharges": [80, 40, 90, 65, 45, 82, 55, 60, 95, 35],
            "Churn": ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No"],
        }
    )
    scores = np.array([0.9, 0.1, 0.8, 0.05, 0.2, 0.85, 0.03, 0.25, 0.7, 0.04])
    ranked = rank_test_customers(sample, scores)
    assert ranked["risk_only_rank"].min() == 1
    assert ranked["risk_only_rank"].max() == len(sample)
    assert ranked["risk_only_rank"].is_unique
    assert "Churn" not in ranked.columns
    assert set(ranked["observed_churn"].unique()) == {0, 1}

    comparisons = evaluate_policies(ranked, budgets=(0.2,))
    assert set(comparisons["policy"]) == {
        "risk_only", "value_weighted", "month_to_month_then_tenure", "short_tenure_only", "high_monthly_charges_only"
    }
    assert (comparisons["customers_selected"] == 2).all()
    assert comparisons["share_test_churners_captured"].between(0, 1).all()
    assert np.isfinite(comparisons["churn_capture_gain_vs_simple_rule_pp"]).all()


def test_invalid_positive_tenure_total_charges_fails(tmp_path):
    data = make_dataset()
    data.loc[data["tenure"] > 0, "TotalCharges"] = data.loc[data["tenure"] > 0, "TotalCharges"].astype(str)
    data.loc[data["tenure"] == 1, "TotalCharges"] = " "
    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="positive tenure"):
        load_and_validate_data(path)


def test_missing_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_validate_data(tmp_path / "does_not_exist.csv")


def test_invalid_contact_budget_fails():
    sample = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "Contract": ["Month-to-month", "One year"],
            "tenure": [1, 20],
            "MonthlyCharges": [80.0, 40.0],
            "observed_churn": [1, 0],
            "risk_only_rank": [1, 2],
            "value_weighted_rank": [1, 2],
            "month_to_month_then_tenure_rank": [1, 2],
            "short_tenure_only_rank": [1, 2],
            "high_monthly_charges_only_rank": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="contact budget"):
        evaluate_policies(sample, budgets=(0.0,))


def test_model_selection_applies_predeclared_simplicity_rule():
    below_threshold = pd.DataFrame(
        {
            "model": ["logistic_regression", "random_forest", "lightgbm", "prior_baseline"],
            "average_precision_mean": [0.644, 0.647, 0.652, 0.265],
        }
    )
    selected, reason = select_primary_model(below_threshold, minimum_complex_model_gain=0.01)
    assert selected == "logistic_regression"
    assert "below the predeclared" in reason

    above_threshold = below_threshold.copy()
    above_threshold.loc[above_threshold["model"] == "lightgbm", "average_precision_mean"] = 0.660
    selected, reason = select_primary_model(above_threshold, minimum_complex_model_gain=0.01)
    assert selected == "lightgbm"
    assert "meeting the predeclared" in reason
