# Diabetes readmission experiment — QUICK CHECK, NOT FINAL RESULTS

Generated from an actual run. Positive class: readmitted within 30 days.

## Data and protocol

- Raw encounters: 101,766; eligible encounters in this run: 5,620.
- Death/hospice exclusions in the full source: 2,423.
- Training: 4,479 encounters (3,214 patients).
- Test: 1,141 encounters (786 patients).
- Test encounter fraction: 20.30%; positive rate: 12.62%.
- Shared patients between train/test and between fit/validation in every fold: **0**.
- 3-fold stratified, patient-grouped CV on training data; fixed seed 42.
- Small grid search optimized average precision (AP); threshold maximized training out-of-fold F1.
- All preprocessing was fitted inside each training fold. The test set was used only after selection.
- Model selected by mean CV AP before test evaluation: **logistic_regression**.
- Runtime: 0.2 minutes. Package versions and file hash: `run_metadata.json`.

Training diagnostics:

- neural_network: ConvergenceWarning — Stochastic Optimizer: Maximum iterations (40) reached and the optimization hasn't converged yet.

## Training cross-validation (development estimates)

| model | cv_ap_mean | cv_ap_std | cv_roc_auc_mean | threshold |
| --- | --- | --- | --- | --- |
| majority_baseline | 0.1084 | 0.0059 | 0.5000 | 0.5000 |
| logistic_regression | 0.1714 | 0.0020 | 0.6107 | 0.1212 |
| logistic_balanced | 0.1691 | 0.0063 | 0.6077 | 0.5877 |
| decision_tree | 0.1321 | 0.0103 | 0.5682 | 0.4939 |
| random_forest | 0.1489 | 0.0145 | 0.5940 | 0.4278 |
| gradient_boosting | 0.1673 | 0.0152 | 0.5986 | 0.1014 |
| neural_network | 0.1679 | 0.0101 | 0.6057 | 0.1020 |

These CV scores describe the selected settings on the same folds used for tuning. They can
be optimistic; they are not nested-CV estimates or confidence intervals. Training out-of-fold
scores also support threshold selection and must not be presented as independent test results.

## Untouched holdout results at the validation-selected thresholds

| model | roc_auc | average_precision | accuracy | balanced_accuracy | precision | recall | f1 | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | 0.5000 | 0.1262 | 0.8738 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.1106 |
| logistic_regression | 0.6476 | 0.2565 | 0.6643 | 0.6208 | 0.2020 | 0.5625 | 0.2972 | 0.1050 |
| logistic_balanced | 0.6518 | 0.2609 | 0.7415 | 0.6025 | 0.2214 | 0.4167 | 0.2892 | 0.2357 |
| decision_tree | 0.6282 | 0.1937 | 0.3883 | 0.5697 | 0.1485 | 0.8125 | 0.2511 | 0.2412 |
| random_forest | 0.6296 | 0.1811 | 0.5434 | 0.5991 | 0.1699 | 0.6736 | 0.2713 | 0.2010 |
| gradient_boosting | 0.6037 | 0.1996 | 0.5565 | 0.5710 | 0.1598 | 0.5903 | 0.2515 | 0.1082 |
| neural_network | 0.6559 | 0.2585 | 0.5828 | 0.6127 | 0.1808 | 0.6528 | 0.2831 | 0.1049 |

The CV-selected model obtained test AUROC **0.6476**, AP **0.2565**,
recall **0.5625**, and F1 **0.2972**, using threshold **0.1212**.
The majority baseline obtained accuracy **0.8738** with recall **0.0000**.
This shows why accuracy alone is insufficient. AP summarizes the precision–recall curve using
scikit-learn's average-precision definition, not trapezoidal PR area. The no-skill AP reference
is the test positive fraction (0.1262). Lower Brier score is better.

`test_metrics.csv` also contains results at 0.5 for every model. Comparing those rows with
the chosen-threshold rows separates the effect of threshold selection from model ranking.
Compare the two logistic models to examine class weighting; both also receive their own small
regularization search. Weighted scores are not necessarily calibrated probabilities.

## Figures and interpretation

![Training exploration](eda.png)

![Comparison](model_comparison.png)

![ROC, precision–recall, calibration](roc_pr_calibration.png)

![Confusion matrices](confusion_matrices.png)

![Forest importance](feature_importance.png)

Forest importance reflects model splits, can favor high-cardinality features, and does not
establish causes of readmission. Calibration plots describe the raw scores; no calibration
model was fitted. F1 optimization is an academic choice, not a clinical operating threshold.

## Scope and limitations

This is an educational, retrospective encounter-level experiment with patient-disjoint evaluation.
Frequent patients contribute more encounters to the metrics. It is not prospective or external
hospital validation, and the random split does not estimate performance in a future year.
The source covers 1999–2008; sensitive demographic attributes are retained, and subgroup
fairness has not been established. Prediction is at discharge, when diagnoses, medication
changes, and discharge disposition are available. Readmissions outside the source network
may be missed. Performance does not establish clinical usefulness.

The implementation deliberately omits nested CV, SMOTE, deep networks, and exact paper
replications. It uses scikit-learn histogram gradient boosting instead of XGBoost. Literature
results with different cohorts and splits must be discussed as context, not direct benchmarks.
See `../docs/methodology.md` for the mapping to the proposal and supplied examples.
