# ChurnCompass

**Customer churn classification and CRM prioritization under limited contact capacity.**

ChurnCompass evaluates whether model-based customer rankings identify more churn-labelled customers than simple CRM rules when a team can contact only a limited share of its customer list.

## Business question

When a CRM team can contact only 10%, 20%, or 30% of customers, should it prioritize by predicted churn risk, predicted risk multiplied by monthly charges, or transparent operational rules?

## Scope and limitations

The IBM Telco sample is a cross-sectional snapshot. Its `Churn` column records current observed churn status; it does not define a future observation window. This project studies retrospective churn-status classification and prioritization, not a validated next-month forecast.

Monthly charges are only an exposure proxy. They are not profit, customer lifetime value, revenue saved, or ROI. The data contains no retention-treatment outcomes, so causal retention uplift cannot be estimated.

The dataset was used in an earlier exploratory feasibility review. The current fixed holdout is kept out of model selection, but it is not independent external validation.

## Run locally

Requirements: Python 3.10–3.13 and Git. On first run, the canonical CSV is downloaded from the IBM public repository if it is not present, and its SHA-256 checksum is verified before use. Raw data is not committed.

Create and activate a virtual environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m churn_compass.run_pipeline
pytest
```

Use a local dataset or custom output path with:

```bash
python -m churn_compass.run_pipeline --data-path path/to/Telco-Customer-Churn.csv --output-dir outputs
```

## Method

1. Validate schema, identifier uniqueness, target labels, numeric ranges, missing cumulative charges, and the canonical data checksum.
2. Separate a stratified development partition and fixed holdout before preprocessing.
3. Compare Logistic Regression, Random Forest, LightGBM, and a prior-only baseline using repeated stratified 5-fold cross-validation (two repeats) on development data.
4. Prefer Logistic Regression unless the best nonlinear model improves mean development average precision by at least 0.01.
5. Evaluate the selected model on the holdout and report ROC-AUC, average precision, Brier score, log loss, and bootstrap uncertainty intervals.
6. Compare five CRM ranking policies at equal 10%, 20%, and 30% contact budgets.
7. Export reproducible results, validation gates, a run summary, and a readable report.

## Reference run

The verified run dated 14 October 2026 selected Logistic Regression. Holdout ROC-AUC was 0.854 (95% bootstrap interval 0.831–0.876), average precision was 0.693 (0.641–0.734), and average precision was 2.61 times holdout churn prevalence. At a 20% contact budget, risk ranking captured 51.1% of churn-labelled test customers versus 41.2% for the simple contract/tenure rule, a difference of 9.9 percentage points. All three predeclared performance gates passed.

Full evidence and limitations are in [`reports/validation_report.md`](reports/validation_report.md). These results apply to this sample and split; they do not guarantee future performance.

## Generated outputs

The pipeline writes these files under `outputs/`; they are reproducible and are not committed by default:

| File | Purpose |
|---|---|
| `run_report.md` | Results, limitations, and acceptance checks |
| `run_summary.json` | Dataset fingerprint, versions, split and run metadata |
| `model_evaluation.json` | Holdout metrics and 95% bootstrap intervals |
| `model_comparison.csv` | Development-only cross-validation results |
| `targeting_policy_comparison.csv` | Policy performance at each contact budget |
| `customer_priority_queue.csv` | Holdout risk scores and retrospective policy ranks |
| `segment_performance.csv` | Results by contract, service, payment, and tenure segments |
| `model_feature_importance.csv` | Descriptive feature importance, not causal effects |
| `calibration_summary.csv` | Predicted probabilities versus observed outcomes |
| `data_quality_checks.csv` | Data checks and source fingerprint |
| `validation_checks.csv` | Predeclared pass/fail gates |

`customer_priority_queue.csv` includes `observed_churn` for retrospective evaluation only. Never use it as an input in a live scoring workflow.

## Acceptance gates

The run reports `PASS` only if data checks pass, holdout ROC-AUC is at least 0.78, average precision is at least 1.5 times holdout churn prevalence, and risk ranking captures at least 5 percentage points more churn-labelled customers than the contract/tenure rule at 20% capacity. These are project-specific gates, not general industry standards. If a gate fails, do not tune against the holdout; review the method or reject the problem statement.

## Sources

- Dataset: [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d). The canonical CSV URL and checksum are recorded in `src/churn_compass/data.py`.
- An alternative public mirror describes the same 7,043-customer sample: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Power BI handoff

After verifying a local run, import the relevant CSV/JSON outputs into Power BI. Label all observed churn metrics as retrospective test results, and describe monthly-charge totals as exposure rather than saved revenue.
