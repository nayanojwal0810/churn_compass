"""Leakage-aware model comparison, model selection, and holdout evaluation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 20261014


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return a transparent baseline and three candidate models.

    Tree counts and thread counts are deliberately bounded so the complete run is
    practical on a typical laptop while remaining large enough for this dataset.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X, scale_numeric=True)),
                ("model", LogisticRegression(C=1.0, max_iter=2500, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=180,
                        learning_rate=0.04,
                        num_leaves=15,
                        min_child_samples=30,
                        reg_lambda=2.0,
                        colsample_bytree=0.9,
                        subsample=0.85,
                        subsample_freq=1,
                        n_jobs=1,
                        verbosity=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "prior_baseline": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X, scale_numeric=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
    }


def compare_models(X_development: pd.DataFrame, y_development: pd.Series) -> pd.DataFrame:
    """Compare baseline pipelines with stratified 5-fold CV on development data."""
    candidates = build_model_pipelines(X_development)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "brier_score": "neg_brier_score",
        "log_loss": "neg_log_loss",
    }
    rows: list[dict[str, Any]] = []
    for name, pipeline in candidates.items():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            results = cross_validate(
                pipeline,
                X_development,
                y_development,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
        rows.append(
            {
                "model": name,
                "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
                "roc_auc_sd": float(np.std(results["test_roc_auc"], ddof=1)),
                "average_precision_mean": float(np.mean(results["test_average_precision"])),
                "average_precision_sd": float(np.std(results["test_average_precision"], ddof=1)),
                "brier_score_mean": float(-np.mean(results["test_brier_score"])),
                "brier_score_sd": float(np.std(-results["test_brier_score"], ddof=1)),
                "log_loss_mean": float(-np.mean(results["test_log_loss"])),
                "log_loss_sd": float(np.std(-results["test_log_loss"], ddof=1)),
                "fold_evaluations": int(len(results["test_roc_auc"])),
                "evaluation_scope": "development set only; repeated stratified 5-fold CV (2 repeats); not final test",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["average_precision_mean", "roc_auc_mean"], ascending=False
    ).reset_index(drop=True)


def select_primary_model(
    cv_summary: pd.DataFrame, minimum_complex_model_gain: float = 0.01
) -> tuple[str, str]:
    """Choose a model from development-only CV results using a predeclared rule.

    Logistic Regression is preferred unless the best nonlinear candidate improves
    mean average precision by at least ``minimum_complex_model_gain``. The prior
    baseline is excluded from primary-model selection.
    """
    candidates = cv_summary.loc[cv_summary["model"].ne("prior_baseline")].copy()
    if candidates.empty or "logistic_regression" not in set(candidates["model"]):
        raise ValueError("Model selection requires Logistic Regression and at least one candidate.")
    scores = candidates.set_index("model")["average_precision_mean"].to_dict()
    logistic_score = float(scores["logistic_regression"])
    best_name = max(scores, key=scores.get)
    if best_name == "logistic_regression":
        return best_name, "Highest development cross-validation average precision among predictive candidates."
    gain = float(scores[best_name] - logistic_score)
    if gain < minimum_complex_model_gain:
        return "logistic_regression", (
            f"Best nonlinear candidate ({best_name}) improved mean development CV average precision by "
            f"{gain:.3f}, below the predeclared {minimum_complex_model_gain:.3f} threshold; "
            "prefer the simpler, more interpretable Logistic Regression model."
        )
    return best_name, (
        f"{best_name} improved mean development CV average precision over Logistic Regression by "
        f"{gain:.3f}, meeting the predeclared material-gain threshold."
    )


def fit_primary_model(model_name: str, X_development: pd.DataFrame, y_development: pd.Series) -> Pipeline:
    """Fit the selected, predeclared pipeline on all development rows only."""
    candidates = build_model_pipelines(X_development)
    if model_name not in candidates or model_name == "prior_baseline":
        raise ValueError(f"Unsupported primary model: {model_name}")
    model = candidates[model_name]
    model.fit(X_development, y_development)
    return model


def _bootstrap_metric_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bootstrap: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int | str]:
    """Estimate descriptive 95% percentile-bootstrap intervals on the holdout.

    The intervals quantify test-sample uncertainty only. They are not external
    validation and are never used for model selection. Samples lacking both classes
    are skipped.
    """
    rng = np.random.default_rng(random_state)
    n_rows = len(y_true)
    auc_values: list[float] = []
    ap_values: list[float] = []
    brier_values: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_rows, size=n_rows)
        y_sample = y_true[indices]
        if np.unique(y_sample).size < 2:
            continue
        p_sample = probabilities[indices]
        auc_values.append(float(roc_auc_score(y_sample, p_sample)))
        ap_values.append(float(average_precision_score(y_sample, p_sample)))
        brier_values.append(float(brier_score_loss(y_sample, p_sample)))
    if len(auc_values) < int(n_bootstrap * 0.90):
        raise RuntimeError("Too few valid bootstrap samples to estimate metric intervals reliably.")
    auc_low, auc_high = np.quantile(auc_values, [0.025, 0.975])
    ap_low, ap_high = np.quantile(ap_values, [0.025, 0.975])
    brier_low, brier_high = np.quantile(brier_values, [0.025, 0.975])
    return {
        "bootstrap_resamples_requested": n_bootstrap,
        "bootstrap_resamples_valid": len(auc_values),
        "roc_auc_ci_95_low": float(auc_low),
        "roc_auc_ci_95_high": float(auc_high),
        "average_precision_ci_95_low": float(ap_low),
        "average_precision_ci_95_high": float(ap_high),
        "brier_score_ci_95_low": float(brier_low),
        "brier_score_ci_95_high": float(brier_high),
        "interval_method": "percentile bootstrap on reserved test rows; descriptive uncertainty only",
    }


def evaluate_locked_test(
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate the model selected on development data on the holdout set once."""
    # LightGBM may warn when a ColumnTransformer emits an array after fitting; predictions
    # remain aligned because the exact same fitted preprocessor is used for both steps.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        probabilities = model.predict_proba(X_test)[:, 1]

    y_array = y_test.to_numpy(dtype="int8")
    predictions_05 = (probabilities >= 0.5).astype("int8")
    metrics: dict[str, Any] = {
        "selected_model": model_name,
        "test_rows": int(len(y_test)),
        "test_churners": int(y_array.sum()),
        "churn_prevalence": float(y_array.mean()),
        "roc_auc": float(roc_auc_score(y_array, probabilities)),
        "average_precision": float(average_precision_score(y_array, probabilities)),
        "average_precision_baseline": float(y_array.mean()),
        "brier_score": float(brier_score_loss(y_array, probabilities)),
        "log_loss": float(log_loss(y_array, probabilities, labels=[0, 1])),
        "recall_at_0_50": float(recall_score(y_array, predictions_05, zero_division=0)),
        "precision_at_0_50": float(precision_score(y_array, predictions_05, zero_division=0)),
        "test_evaluated_once": True,
        "threshold_note": "0.50 is a reference threshold only; CRM prioritization uses ranking, not this threshold.",
    }
    metrics.update(_bootstrap_metric_intervals(y_array, probabilities))
    return metrics, probabilities


def export_feature_importance(model_name: str, model: Pipeline) -> pd.DataFrame:
    """Export native model importance as descriptive, non-causal evidence."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        signed_values = estimator.coef_.ravel().astype(float)
        values = np.abs(signed_values)
        importance_type = "absolute_logistic_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_.astype(float)
        signed_values = np.full_like(values, np.nan, dtype=float)
        importance_type = "tree_feature_importance"
    else:
        return pd.DataFrame(columns=["model", "feature", "importance", "direction_value", "importance_type", "rank"])
    output = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "importance": values,
            "direction_value": signed_values,
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)
    output["interpretation_note"] = (
        "Descriptive model association only; not a causal effect. Logistic coefficients are not directly comparable across differently scaled feature types."
        if importance_type == "absolute_logistic_coefficient"
        else "Descriptive model feature importance only; not a causal effect."
    )
    return output
