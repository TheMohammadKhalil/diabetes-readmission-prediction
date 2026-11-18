"""Run the entire course experiment: python run_experiment.py."""

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from readmission.data import clean_cohort, download_data, load_data, make_features, make_splits
from readmission.evaluation import (
    choose_threshold, compute_metrics, save_eda, save_evaluation_plots, save_importance,
)
from readmission.models import model_specs


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def markdown_table(df):
    """Avoid another dependency just to format a small report table."""
    rows = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(out, audit, cv_results, test_results, selected, duration, quick):
    tuned = test_results[test_results.operating_point == "validation_threshold"]
    chosen = tuned.set_index("model").loc[selected]
    baseline = tuned.set_index("model").loc["majority_baseline"]
    warnings_found = json.loads((out / "warnings.json").read_text())
    warning_text = "No training warnings were recorded." if not warnings_found else "\n".join(
        f"- {item['model']}: {item['category']} — {item['message']}" for item in warnings_found
    )
    text = f"""# Diabetes readmission experiment — {'QUICK CHECK, NOT FINAL RESULTS' if quick else 'full dataset'}

Generated from an actual run. Positive class: readmitted within 30 days.

## Data and protocol

- Raw encounters: {audit['raw_rows']:,}; eligible encounters in this run: {audit['eligible_rows']:,}.
- Death/hospice exclusions in the full source: {audit['excluded_rows']:,}.
- Training: {audit['train_rows']:,} encounters ({audit['train_patients']:,} patients).
- Test: {audit['test_rows']:,} encounters ({audit['test_patients']:,} patients).
- Test encounter fraction: {audit['test_fraction']:.2%}; positive rate: {audit['test_positive_rate']:.2%}.
- Shared patients between train/test and between fit/validation in every fold: **0**.
- {audit['folds']}-fold stratified, patient-grouped CV on training data; fixed seed {audit['seed']}.
- Small grid search optimized average precision (AP); threshold maximized training out-of-fold F1.
- All preprocessing was fitted inside each training fold. The test set was used only after selection.
- Model selected by mean CV AP before test evaluation: **{selected}**.
- Runtime: {duration / 60:.1f} minutes. Package versions and file hash: `run_metadata.json`.

Training diagnostics:

{warning_text}

## Training cross-validation (development estimates)

{markdown_table(cv_results[['model', 'cv_ap_mean', 'cv_ap_std', 'cv_roc_auc_mean', 'threshold']])}

These CV scores describe the selected settings on the same folds used for tuning. They can
be optimistic; they are not nested-CV estimates or confidence intervals. Training out-of-fold
scores also support threshold selection and must not be presented as independent test results.

## Untouched holdout results at the validation-selected thresholds

{markdown_table(tuned[['model', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'brier_score']])}

The CV-selected model obtained test AUROC **{chosen.roc_auc:.4f}**, AP **{chosen.average_precision:.4f}**,
recall **{chosen.recall:.4f}**, and F1 **{chosen.f1:.4f}**, using threshold **{chosen.threshold:.4f}**.
The majority baseline obtained accuracy **{baseline.accuracy:.4f}** with recall **{baseline.recall:.4f}**.
This shows why accuracy alone is insufficient. AP summarizes the precision–recall curve using
scikit-learn's average-precision definition, not trapezoidal PR area. The no-skill AP reference
is the test positive fraction ({audit['test_positive_rate']:.4f}). Lower Brier score is better.

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
"""
    (out / "report.md").write_text(text)


def run(data_path=None, output=None, quick=False, seed=42, threads=2):
    started = time.perf_counter()
    out = Path(output or ("results_quick" if quick else "results"))
    if (out / "run_metadata.json").exists():
        raise FileExistsError(f"{out} already contains a run. Choose a new --output directory.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    path = Path(data_path) if data_path else download_data()
    raw = load_data(path)
    cohort = clean_cohort(raw)
    excluded_rows = len(raw) - len(cohort)
    if quick:
        # Sample whole patients, never individual rows from a patient's history.
        patients = cohort.patient_nbr.drop_duplicates().to_numpy()
        sampled = np.random.default_rng(seed).choice(patients, min(4000, len(patients)), replace=False)
        cohort = cohort[cohort.patient_nbr.isin(sampled)].reset_index(drop=True)
    X = make_features(cohort)
    y = cohort.readmitted.eq("<30").astype(int)
    groups = cohort.patient_nbr
    folds = 3 if quick else 5
    train_idx, test_idx, cv = make_splits(X, y, groups, seed, folds)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    audit = {
        "raw_rows": len(raw), "excluded_rows": excluded_rows, "eligible_rows": len(cohort),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_patients": int(groups.iloc[train_idx].nunique()),
        "test_patients": int(groups.iloc[test_idx].nunique()),
        "patient_overlap": 0, "test_fraction": len(test_idx) / len(cohort),
        "train_positive_rate": float(y_train.mean()), "test_positive_rate": float(y_test.mean()),
        "folds": folds, "seed": seed,
    }
    membership = cohort[["encounter_id", "patient_nbr"]].copy()
    membership["split"] = "train"
    membership.loc[test_idx, "split"] = "test"
    membership["cv_fold"] = -1
    fold_audit = []
    for fold, (fit_idx, val_idx) in enumerate(cv, start=1):
        membership.loc[train_idx[val_idx], "cv_fold"] = fold
        fold_audit.append({
            "fold": fold, "fit_rows": len(fit_idx), "validation_rows": len(val_idx),
            "patient_overlap": 0, "validation_positive_rate": float(y_train.iloc[val_idx].mean()),
        })
    membership.to_csv(out / "split_membership.csv", index=False)
    write_json(out / "split_audit.json", {**audit, "cv": fold_audit})
    write_json(out / "run_metadata.json", {
        "status": "running", "quick": quick, "seed": seed, "folds": folds,
        "source_csv": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    })
    print(f"Eligible rows: {len(cohort):,}; train: {len(train_idx):,}; test: {len(test_idx):,}.", flush=True)
    save_eda(cohort.iloc[train_idx], X_train, y_train, out)
    summaries, fold_scores, thresholds, fitted = [], [], {}, {}
    oof_output = cohort.iloc[train_idx][["encounter_id", "patient_nbr"]].copy()
    oof_output["target"] = y_train
    warning_log = []
    with threadpool_limits(limits=threads):
        for name, (pipeline, grid) in model_specs(seed, quick).items():
            model_started = time.perf_counter()
            print(f"Training {name}: {folds}-fold grid search and OOF predictions...", flush=True)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                search = GridSearchCV(
                    pipeline, grid, cv=cv, scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
                    refit="ap", n_jobs=1, error_score="raise", return_train_score=False,
                )
                search.fit(X_train, y_train)
                pd.DataFrame(search.cv_results_).to_csv(out / f"search_{name}.csv", index=False)
                oof = np.full(len(y_train), np.nan)
                for fold, (fit_idx, val_idx) in enumerate(cv, start=1):
                    estimator = clone(search.best_estimator_)
                    estimator.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                    probabilities = estimator.predict_proba(X_train.iloc[val_idx])[:, 1]
                    oof[val_idx] = probabilities
                    fold_scores.append({"model": name, "fold": fold,
                                        **compute_metrics(y_train.iloc[val_idx], probabilities)})
                if np.isnan(oof).any():
                    raise AssertionError("Some training rows did not receive an OOF prediction")
                thresholds[name] = 0.5 if name == "majority_baseline" else choose_threshold(y_train, oof)
                fitted[name] = search.best_estimator_
                oof_output[name] = oof
            for message in recorded:
                entry = {"model": name, "category": message.category.__name__, "message": str(message.message)}
                if entry not in warning_log:
                    warning_log.append(entry)
            selected_index = search.best_index_
            summaries.append({
                "model": name, "cv_ap_mean": search.best_score_,
                "cv_ap_std": search.cv_results_["std_test_ap"][selected_index],
                "cv_roc_auc_mean": search.cv_results_["mean_test_roc_auc"][selected_index],
                "threshold": thresholds[name], "best_params": json.dumps(search.best_params_),
                "seconds": time.perf_counter() - model_started,
            })
            pd.DataFrame(summaries).to_csv(out / "cv_summary.csv", index=False)
            write_json(out / "warnings.json", warning_log)
            print(f"  CV AP={search.best_score_:.4f}; threshold={thresholds[name]:.4f}; "
                  f"{time.perf_counter() - model_started:.1f}s", flush=True)

        # Freeze the winner and all thresholds BEFORE asking any model to predict test data.
        cv_summary = pd.DataFrame(summaries)
        selected = cv_summary.sort_values("cv_ap_mean", ascending=False).iloc[0]["model"]
        write_json(out / "selection.json", {"model": selected, "criterion": "training CV average precision",
                                            "thresholds": thresholds})
        joblib.dump({"pipeline": fitted[selected], "threshold": thresholds[selected],
                     "model_name": selected, "feature_columns": list(X.columns)},
                    out / "models" / "selected_model.joblib", compress=3)
        predictions, test_rows = {}, []
        test_output = cohort.iloc[test_idx][["encounter_id", "patient_nbr"]].copy()
        test_output["target"] = y_test
        for name, pipeline in fitted.items():
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions[name] = probabilities
            test_output[name] = probabilities
            for label, threshold in [("default_0.5", 0.5), ("validation_threshold", thresholds[name])]:
                test_rows.append({"model": name, "operating_point": label,
                                  **compute_metrics(y_test, probabilities, threshold)})
    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(out / "cv_fold_metrics.csv", index=False)
    test_output.to_csv(out / "test_predictions.csv", index=False)
    oof_output.to_csv(out / "train_oof_predictions.csv", index=False)
    save_evaluation_plots(y_test, predictions, thresholds, test_results, out)
    save_importance(fitted["random_forest"], out)
    duration = time.perf_counter() - started
    write_report(out, audit, cv_summary, test_results, selected, duration, quick)
    metadata = json.loads((out / "run_metadata.json").read_text())
    metadata.update(status="complete", elapsed_seconds=duration, selected_model=selected,
                    completed_utc=datetime.now(timezone.utc).isoformat(), warning_count=len(warning_log))
    write_json(out / "run_metadata.json", metadata)
    print(f"Finished. CV-selected model: {selected}. Report: {out / 'report.md'}", flush=True)
    if warning_log:
        print(f"Recorded {len(warning_log)} distinct training warnings; inspect {out / 'warnings.json'}.", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Existing UCI diabetic_data.csv; downloads if omitted")
    parser.add_argument("--output", help="New output directory (default: results or results_quick)")
    parser.add_argument("--quick", action="store_true", help="4,000 patients, 3 CV folds; not final results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args.data, args.output, args.quick, args.seed, args.threads)
