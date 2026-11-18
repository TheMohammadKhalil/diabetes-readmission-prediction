# Predicting 30-day readmission of diabetic patients

A bachelor's machine-learning course project using Python, pandas, and scikit-learn.
The target is **readmission within 30 days** (`<30` = 1; `>30` and `NO` = 0).
The supplied proposal and examples guide the project; their reported results are not
treated as results of this implementation.

Start with **[the walkthrough notebook](notebooks/readmission_project.ipynb)**,
**[the experiment report](results/report.md)**, and
**[the methodology](docs/methodology.md)**.
An **[HTML copy of the executed notebook](notebooks/readmission_project.html)**
can be opened directly in a browser without starting Jupyter.

The included full run used 79,625 training and 19,718 testing encounters, with no
shared patients. Gradient boosting was selected using training CV. Its held-out
AUROC was **0.6798**, average precision **0.2386**, recall **0.4593**, and F1 **0.2893**
at the validation-selected threshold. The run took about **11 minutes** on this
machine and recorded no model-training warnings. All 17 protocol tests passed;
saved predictions, metric tables, and the exported model were checked for agreement.

## Run

Python 3.11 or 3.12 is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py
```

The public UCI data download is automatic if `data/raw/diabetic_data.csv` is absent.
If downloading manually, extract `diabetic_data.csv` and `IDS_mapping.csv` from the
[UCI dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)
into `data/raw/`. You can also use `--data /path/to/diabetic_data.csv`.

The full run uses all **101,766 source encounters**, of which **99,343** meet the
fixed eligibility rule. Training time depends on the computer. The script prints
progress for each model and uses at most two numerical-library threads by default.
Full results live in `results/`; existing runs are protected from accidental overwrite:

```bash
python run_experiment.py --output results_repeat
```

For a fast installation/pipeline check:

```bash
python run_experiment.py --quick
```

This samples 4,000 whole patients, reduces model sizes, and uses 3-fold CV. It saves
separately to `results_quick/`; **do not use quick-run scores as final project results**.
Use `requirements-lock.txt` to recreate the exact environment used for the included run.

To open the notebook:

```bash
jupyter lab notebooks/readmission_project.ipynb
```

Choose the project's `.venv` Python interpreter/kernel. The notebook explains and
inspects the saved experiment; it can also launch training if results are absent.
It does not silently retrain models whenever you open it.

## What is implemented

- An approximately **80% training / 20% testing** split with no shared patients.
- **5-fold stratified group cross-validation** on the training portion only.
- Fold-fitted imputation, numerical scaling, and one-hot encoding in pipelines.
- Missing-data handling, broad diagnosis categories, and two simple derived features.
- Seven comparisons: majority baseline, ordinary logistic regression, balanced logistic
  regression, decision tree, random forest, histogram gradient boosting, and a small MLP.
- Small `GridSearchCV` searches, including L2 logistic regularization strength.
- Class weighting and training out-of-fold F1 threshold selection.
- Accuracy, AUROC, average precision (AP), positive-class precision/recall/F1,
  balanced accuracy, Brier score, and confusion-matrix counts.
- Training-data exploration, ROC/PR curves, calibration plots, and forest importance.
- A saved selected model, reproducibility metadata, split audit, and actual-result report.

The winner is selected by **training CV average precision**, not by test performance.
The ordinary and balanced logistic regressions provide a direct class-weighting comparison.
The two operating points in `test_metrics.csv` show how threshold selection affects results.
The majority baseline always predicts the negative class at threshold 0.5, but returns
the training positive fraction as its probability score.

## Files

| Path | Purpose |
| --- | --- |
| `run_experiment.py` | Complete training, selection, evaluation, and report workflow |
| `readmission/data.py` | Source download, cohort rules, feature preparation, patient splits |
| `readmission/models.py` | Preprocessing pipelines and small model grids |
| `readmission/evaluation.py` | Metrics, threshold selection, and figures |
| `notebooks/readmission_project.ipynb` | Course presentation and walkthrough |
| `docs/methodology.md` | Explanation of choices and proposal simplifications |
| `docs/presentation_guide.md` | How to explain the project and write the final paper |
| `results/report.md` | Generated report from the full experiment |
| `results/cv_summary.csv` | Training CV means, standard deviations, selected parameters |
| `results/test_metrics.csv` | Holdout results at 0.5 and validation-selected thresholds |
| `results/split_audit.json` | Exact sizes, class fractions, and patient-overlap checks |
| `results/split_membership.csv` | Encounter IDs, patient IDs, split, and validation fold |
| `results/search_*.csv` | All candidate settings and their fold scores |
| `results/warnings.json` | Any warnings from model fitting, including nonconvergence |
| `results/models/selected_model.joblib` | Fitted preprocessing/model and chosen threshold |
| `tests/test_protocol.py` | Checks for leakage, labels, missing values, and metrics |

Run the checks with `python -m pytest -q`. Source data, the virtual environment, and
large model/prediction files and patient-level split membership are ignored by Git;
retain them locally for reproducibility. The experiment regenerates the split audit
files when run; published results contain aggregate metrics and figures.

## Dataset credit

Clore, Cios, DeShazo, and Strack. *Diabetes 130-US Hospitals for Years 1999–2008*.
UCI Machine Learning Repository, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J).
Dataset license: **CC BY 4.0**. Original files remain unchanged; filtering and feature
transformations occur in memory and are documented in `docs/methodology.md`.

This is an academic experiment on historical records, not a validated clinical tool.
