# Methodology and scope

## Question and prediction time

Can records available **at discharge** predict a diabetic patient's readmission within
30 days? Each row is one encounter. The target is binary: `<30` versus the combined
`>30`/`NO` labels. Diagnosis, medication changes, and discharge disposition are used
because the intended prediction time is discharge, not admission.

The raw file has 101,766 rows, 71,518 patients, 47 predictors, two identifiers, and one
target. Only 11,357 rows are positive. An always-negative classifier already achieves
about 88.8% accuracy on the raw data, which motivates positive-class metrics and AP.
The exact post-filter and split distributions are computed in each run.

## Cleaning and feature preparation

1. Read `?` and empty cells as missing. Keep literal `None` in the two lab fields as
   **test not performed**. This is distinct from an unknown value.
2. Remove discharge codes **11, 13, 14, 19, 20, 21** (expired or hospice), according to
   the supplied UCI `IDS_mapping.csv`. This leaves **99,343 encounters** and 69,990
   patients in the full run. Other discharge codes are retained.
3. Drop `weight` because of its approximately 97% missingness, and the two documented
   constant drugs `examide` and `citoglipton`. These are prespecified dataset rules,
   not feature selection based on test scores.
4. Never use `encounter_id` or `patient_nbr` as predictors. Patient ID defines groups;
   encounter ID lets us audit the split. Neither is a measure of medical risk.
5. Map each ICD-9 field to Circulatory, Respiratory, Digestive, Diabetes, Injury,
   Musculoskeletal, Genitourinary, Neoplasms, Other, or Missing. The broad groups
   follow Strack et al.; here the same map is applied to all three diagnosis fields.
   E/V codes map to Other, and decimal codes are handled correctly (e.g., 250.13).
6. Add `total_prior_visits` = inpatient + outpatient + emergency visits in the past
   year, and `num_medication_changes` = number of drug fields marked Up or Down.
   These use only the current row. No future encounters are aggregated.
7. Median-impute and standardize the ten numerical features. Constant-impute all
   categorical missing values with `Missing`, then one-hot encode categories, ignoring
   previously unseen categories at prediction time. The three administrative IDs
   are categorical; age remains an age band. Payer and specialty are retained.

Steps 1–6 are deterministic, prespecified row transformations. Step 7 learns values
from data and therefore belongs **inside each model pipeline**. Each CV fit gets a
fresh imputer, scaler, encoder, and estimator. EDA plots use the training portion.

## Why patient grouping changes the usual 80/20 split

Rows from one patient are related. A random row split can put that person's encounters
in both training and testing, making generalization appear easier. We use the first
fold of a seeded `StratifiedGroupKFold(n_splits=5)` as a fixed held-out test set and
the other four folds as training. No fold is chosen based on scores or a preferred
test distribution.

This yields **approximately** 80/20 by encounter while keeping each patient together.
Exact row proportions and perfectly equal class ratios cannot both be guaranteed
when groups have different sizes and sometimes different encounter labels.

We then create a separate five-fold `StratifiedGroupKFold` **within the training set**.
The same fold indices are reused across models for fair comparison. An assertion
checks patient overlap for the holdout and every CV fold. An audit file records the
actual split and class fractions. The final test is not one of the training CV folds.

## Models and tuning

| Model | Reason | Small grid |
| --- | --- | --- |
| Majority baseline | Shows what accuracy means without useful predictions | None |
| Logistic regression | Simple linear, L2-regularized baseline | C = 0.1/1 |
| Balanced logistic regression | Same family with inverse-frequency class weights | C = 0.1/1 |
| Decision tree | Explainable nonlinear splits | Depth = 4/8; leaf size 40 |
| Random forest | Averaging 100 trees; controls overfitting | Depth = 10/18; leaf size 10 |
| Histogram gradient boosting | Efficient sequential tree ensemble in scikit-learn | Leaves = 15/31; 100 iterations |
| Small MLP | Neural-network comparison promised in proposal | One 32-unit hidden layer; fixed settings |

Tree and forest models use class weights; boosting and MLP use the original class
frequencies. Ordinary versus balanced logistic regression is the controlled family
comparison for weighting, although each gets its own regularization search.

The MLP uses Adam, batches of 512, L2 penalty 0.01, and at most 100 epochs. MLP stopping
uses training-loss improvement; internal random validation early stopping is disabled.
Boosting also disables internal early stopping. Any convergence warnings are saved
to `warnings.json`, not silently hidden. A small MLP is a comparison model, not a
claim that this architecture exhausts neural-network performance.

`GridSearchCV` chooses each family's settings using average precision (AP). It then
refits on all training encounters. The family with the largest mean CV AP is frozen
as the selected model **before any test predictions are made**. No broad parameter
search, stacking, expensive kernel SVM, or deep learning is needed for this project.

## Decision threshold and imbalance

For each selected family configuration, clone and refit its whole pipeline on each
training fold and predict only that fold's validation patients. Pool these out-of-fold
probabilities and choose the threshold maximizing positive-class F1. Ties use the
higher threshold. Keep the majority baseline at 0.5. Save all thresholds before test
evaluation. Report both 0.5 and the chosen threshold on test data.

This is a simple development procedure. Hyperparameters and thresholds use the same
training folds, so selected CV/OOF performance can be optimistic. We do **not** call
these unbiased nested-CV estimates. The untouched patient holdout provides the final
assessment of the prespecified selected model. Fold standard deviation is variability
across folds, not a confidence interval.

F1 is convenient for coursework but does not encode actual clinical costs. Increasing
recall generally increases false alarms; the confusion matrix and precision quantify
that tradeoff. Class weighting can distort probability calibration, so Brier scores
and calibration plots are also reported. No probability calibration model is fitted.

## Evaluation and interpretation

- **AP:** primary selection metric; scikit-learn average precision, not trapezoidal
  PR-AUC. Compare with the positive prevalence as a no-skill reference.
- **AUROC:** probability-ranking discrimination across thresholds.
- **Precision, recall, F1:** always for the `<30` class, not weighted/macro averages.
- **Balanced accuracy:** equal importance to sensitivity and specificity.
- **Accuracy:** included for comparison with the course examples, with the baseline.
- **Brier score:** mean squared probability error; lower is better.
- **Confusion matrices:** actual negative/positive labels on rows; predicted labels
  on columns. Counts reveal the burden of false positives and missed readmissions.

Forest impurity importance is a simple descriptive visualization. It can favor variables
with many possible splits and is not evidence that an input causes readmission. The
plot shows encoded features rather than claiming a clinical ranking of original fields.

## Relationship to your documents

| Supplied material | Applied here |
| --- | --- |
| Intro and Fundamentals | Supervised binary classification, generalization, bias/variance, data quality |
| End to End Project | Fixed test set, training-only EDA, pipelines, cross-validation, small grid search |
| Classification | Majority baseline, confusion matrices, precision/recall, F1, ROC and threshold tradeoff |
| Training Models and updated slide deck | Logistic regularization, scaling, iterative training, overfitting control |
| SVM | Informs scaling and regularization; a kernel SVM is not required by this proposal |
| Sample project 1: diabetes indicators | Model-family comparison, class-imbalance handling, small MLP, feature importance |
| Sample project 2: heart disease | 80/20 holdout, GridSearchCV, confusion matrices and ROC comparison |
| Sample project 3: intrusion detection | 80/20 holdout plus five-fold training CV and comparative tables |

The samples address different prediction tasks, so their accuracies are **not** targets
for this readmission study. Practices that could leak information, such as preprocessing
or resampling before the split, should not be copied from an example.

## Explicit simplifications from the proposal

The proposal says the plan may change; this implementation follows your request for
undergraduate scope. These changes should also be reflected in your submitted report:

- **Scikit-learn histogram boosting replaces XGBoost**, retaining a boosting comparison
  while avoiding an extra framework.
- **Ordinary grouped CV plus an untouched holdout replaces nested CV.** The limitations
  of the development CV estimates are stated explicitly.
- **Class weights replace SMOTE.** Ordinary SMOTE on one-hot variables can create
  fractional category combinations. A careful mixed-data SMOTENC experiment would
  add complexity; it can be future work, not a prerequisite for this implementation.
- **No exact reimplementation of cited papers.** Their different filtering, feature
  choices, and splits prevent a like-for-like claim. The local 80/20 and five-fold
  protocol is fully run, but it is not described as a replication of Liu or Emi-Johnson.
- **No voting ensemble or eleven-model sweep.** The seven entries already compare
  linear, individual tree, bagged tree, boosted tree, and neural approaches.
- **L2-only logistic regression replaces the proposed L1/L2 comparison.** Searching
  regularization strength is sufficient here; the weighted L1 solver was expensive
  on the full data. This runtime-based simplification was made before test evaluation.

## Limits of the study

The source is historical (1999–2008) and reflects a restricted hospital network. We
retain age, race, and gender but do not establish subgroup fairness. We group by
patient, not hospital or year, and do not claim prospective/external validation. Each
encounter receives equal metric weight, so patients with more stays contribute more.
No clinical deployment or benefit is inferred from these retrospective scores.

## Sources

- [UCI dataset and license](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008),
  DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J).
- [Strack et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/), especially
  Table 2 for broad diagnosis grouping. We retain repeated encounters using groups;
  the original analysis selected one encounter per patient.
- [scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- Your proposal, seven course slide PDFs, and three sample projects, read locally.
  The uploaded documents supply project context, not instructions to perform unrelated actions.
