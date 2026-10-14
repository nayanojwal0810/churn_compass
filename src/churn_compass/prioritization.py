"""Compare customer-contact policies at fixed simulated capacity budgets."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

BUDGETS = (0.10, 0.20, 0.30)


def rank_test_customers(test_rows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Create retrospective customer scores and deterministic policy ranks.

    The original target column is replaced with ``observed_churn`` to reduce the
    chance that it is mistaken for a valid feature in a future scoring workflow.
    """
    if len(test_rows) != len(probabilities):
        raise ValueError("Probability count does not match test rows.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1.")
    if "Churn" not in test_rows.columns:
        raise ValueError("Retrospective ranking requires observed Churn labels for evaluation.")

    result = test_rows.drop(columns=["Churn"]).copy().reset_index(drop=True)
    result["observed_churn"] = test_rows["Churn"].eq("Yes").astype("int8").to_numpy()
    result["predicted_churn_probability"] = np.asarray(probabilities, dtype=float)
    result["predicted_monthly_charge_exposure"] = (
        result["predicted_churn_probability"] * result["MonthlyCharges"]
    )
    max_tenure = float(result["tenure"].max())
    result["month_to_month_rule_score"] = (
        result["Contract"].eq("Month-to-month").astype(int) * (max_tenure + 1.0)
        + (max_tenure - result["tenure"])
    )
    result["tenure_rule_score"] = max_tenure - result["tenure"]
    result["monthly_charge_rule_score"] = result["MonthlyCharges"].astype(float)

    score_columns = {
        "risk_only": "predicted_churn_probability",
        "value_weighted": "predicted_monthly_charge_exposure",
        "month_to_month_then_tenure": "month_to_month_rule_score",
        "short_tenure_only": "tenure_rule_score",
        "high_monthly_charges_only": "monthly_charge_rule_score",
    }
    for policy, score_column in score_columns.items():
        ordered_ids = result.sort_values(
            [score_column, "customerID"], ascending=[False, True], kind="mergesort"
        )["customerID"]
        ranks = pd.Series(np.arange(1, len(result) + 1), index=ordered_ids.to_numpy())
        result[f"{policy}_rank"] = result["customerID"].map(ranks).astype(int)

    result["risk_priority_tier"] = pd.cut(
        (result["risk_only_rank"] - 1) / max(len(result), 1),
        bins=[-0.000001, 0.10, 0.20, 0.30, 1.0],
        labels=["Top 10%", "Top 20%", "Top 30%", "Remaining 70%"],
        include_lowest=True,
        right=True,
    ).astype(str)
    result["evaluation_scope"] = "Locked holdout; observed churn is available for retrospective analysis only."
    return result


def evaluate_policies(scored_test: pd.DataFrame, budgets: Iterable[float] = BUDGETS) -> pd.DataFrame:
    """Compare all policies on identical customers and exact integer contact budgets."""
    policies = {
        "risk_only": "risk_only_rank",
        "value_weighted": "value_weighted_rank",
        "month_to_month_then_tenure": "month_to_month_then_tenure_rank",
        "short_tenure_only": "short_tenure_only_rank",
        "high_monthly_charges_only": "high_monthly_charges_only_rank",
    }
    n = len(scored_test)
    if n == 0:
        raise ValueError("Cannot evaluate policies on an empty test set.")
    total_churners = int(scored_test["observed_churn"].sum())
    total_churner_charges = float(
        scored_test.loc[scored_test["observed_churn"].eq(1), "MonthlyCharges"].sum()
    )
    prevalence = float(scored_test["observed_churn"].mean())
    rows: list[dict[str, float | int | str]] = []
    normalized_budgets = tuple(float(value) for value in budgets)
    if any(value <= 0 or value > 1 for value in normalized_budgets):
        raise ValueError("Each contact budget must be greater than 0 and at most 1.")

    for policy, rank_column in policies.items():
        ordered = scored_test.sort_values([rank_column, "customerID"], ascending=[True, True])
        for budget in normalized_budgets:
            n_selected = min(n, max(1, math.ceil(n * budget)))
            selected = ordered.head(n_selected)
            captured_churners = int(selected["observed_churn"].sum())
            selected_churn_rate = float(selected["observed_churn"].mean())
            captured_churner_charges = float(
                selected.loc[selected["observed_churn"].eq(1), "MonthlyCharges"].sum()
            )
            rows.append(
                {
                    "policy": policy,
                    "contact_budget_pct": budget * 100,
                    "test_customers": n,
                    "customers_selected": n_selected,
                    "selected_customer_churn_rate": selected_churn_rate,
                    "churners_captured": captured_churners,
                    "share_test_churners_captured": captured_churners / total_churners if total_churners else np.nan,
                    "lift_vs_random": selected_churn_rate / prevalence if prevalence else np.nan,
                    "share_churner_linked_monthly_charges_captured": (
                        captured_churner_charges / total_churner_charges if total_churner_charges else np.nan
                    ),
                    "selected_monthly_charges_total": float(selected["MonthlyCharges"].sum()),
                    "evaluation_scope": "Locked holdout; retrospective ranking simulation.",
                }
            )

    comparison = pd.DataFrame(rows)
    simple_rule = comparison.loc[
        comparison["policy"].eq("month_to_month_then_tenure"),
        ["contact_budget_pct", "share_test_churners_captured", "share_churner_linked_monthly_charges_captured"],
    ].rename(
        columns={
            "share_test_churners_captured": "simple_rule_churn_capture",
            "share_churner_linked_monthly_charges_captured": "simple_rule_charge_capture",
        }
    )
    comparison = comparison.merge(simple_rule, on="contact_budget_pct", how="left", validate="many_to_one")
    comparison["churn_capture_gain_vs_simple_rule_pp"] = (
        comparison["share_test_churners_captured"] - comparison["simple_rule_churn_capture"]
    ) * 100
    comparison["charge_capture_gain_vs_simple_rule_pp"] = (
        comparison["share_churner_linked_monthly_charges_captured"] - comparison["simple_rule_charge_capture"]
    ) * 100
    comparison = comparison.drop(columns=["simple_rule_churn_capture", "simple_rule_charge_capture"])
    return comparison.sort_values(
        ["contact_budget_pct", "share_test_churners_captured"], ascending=[True, False]
    ).reset_index(drop=True)


def build_segment_summary(scored_test: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed churn and model scores across key CRM segments."""
    segment_columns = ["Contract", "InternetService", "PaymentMethod", "tenure_band"]
    df = scored_test.copy()
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49-72 months"],
        include_lowest=True,
    ).astype("string")
    summaries: list[pd.DataFrame] = []
    for column in segment_columns:
        grouped = (
            df.groupby(column, dropna=False, observed=True)
            .agg(
                customers=("customerID", "size"),
                observed_churners=("observed_churn", "sum"),
                observed_churn_rate=("observed_churn", "mean"),
                mean_monthly_charges=("MonthlyCharges", "mean"),
                mean_predicted_churn_probability=("predicted_churn_probability", "mean"),
            )
            .reset_index()
            .rename(columns={column: "segment_value"})
        )
        grouped.insert(0, "segment_dimension", column)
        summaries.append(grouped)
    return pd.concat(summaries, ignore_index=True)
