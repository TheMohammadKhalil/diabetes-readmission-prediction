import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from readmission.data import EXCLUDED_DISCHARGES, RAW_FEATURES, load_data, make_features
from readmission.evaluation import compute_metrics


def load_model(results):
    path = Path(results) / "models" / "selected_model.joblib"
    if not path.exists():
        raise ValueError(
            f"No saved model at {path}. Run: python run_experiment.py --output my_run"
        )
    return joblib.load(path)


def evaluate_holdout(results, data):
    results = Path(results)
    metadata = json.loads((results / "run_metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("This experiment has not finished.")
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Use the same dataset file that was used for this experiment.")
    model = load_model(results)
    membership = pd.read_csv(results / "split_membership.csv")
    if membership.groupby("patient_nbr")["split"].nunique().max() != 1:
        raise ValueError("The saved split contains overlapping patients.")
    test_ids = membership.loc[membership["split"] == "test", "encounter_id"]
    records = load_data(data).set_index("encounter_id").loc[test_ids]
    target = records["readmitted"].eq("<30").astype(int)
    with threadpool_limits(limits=2):
        scores = model["pipeline"].predict_proba(make_features(records))[:, 1]
    metrics = compute_metrics(target, scores, model["threshold"])
    saved = pd.read_csv(results / "test_metrics.csv")
    saved = saved[(saved["model"] == model["model_name"]) &
                  (saved["operating_point"] == "validation_threshold")].iloc[0]
    if not all(np.isclose(value, saved[name], rtol=1e-8, atol=1e-10)
               for name, value in metrics.items()):
        raise ValueError("The saved model no longer matches the recorded test results.")
    return model["model_name"], len(records), metrics, metadata["quick"]


def predict_csv(results, input_path, output_path):
    model = load_model(results)
    records = pd.read_csv(input_path, keep_default_na=False, na_values=["?", ""], low_memory=False)
    missing = set(RAW_FEATURES) - set(records.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    if records.empty:
        raise ValueError("The input CSV is empty.")
    discharge = pd.to_numeric(records["discharge_disposition_id"], errors="raise")
    if discharge.isin(EXCLUDED_DISCHARGES).any():
        raise ValueError("Death/hospice encounters are outside this project's study cohort.")
    with threadpool_limits(limits=2):
        scores = model["pipeline"].predict_proba(make_features(records))[:, 1]
    predictions = pd.DataFrame({
        "row": np.arange(1, len(records) + 1),
        "model_score": scores,
        "predicted_readmission": (scores >= model["threshold"]).astype(int),
    })
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError(f"{output_path} already exists. Choose a new --output file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions, model["threshold"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the saved model or predict a new CSV.")
    parser.add_argument("--results", default="results", help="Folder containing the trained model")
    parser.add_argument("--data", default="data/raw/diabetic_data.csv", help="Original dataset for the holdout check")
    parser.add_argument("--input", help="New records to predict instead of checking the holdout")
    parser.add_argument("--output", default="predictions/predictions.csv", help="Prediction CSV destination")
    args = parser.parse_args()
    try:
        if args.input:
            predictions, threshold = predict_csv(args.results, args.input, args.output)
            print(predictions.head(10).to_string(index=False))
            print(f"Threshold: {threshold:.4f}. Saved {len(predictions)} predictions to {args.output}.")
        else:
            name, rows, metrics, quick = evaluate_holdout(args.results, args.data)
            print(f"Model: {name}\nHeld-out encounters: {rows:,}")
            if quick:
                print("Quick-run model: these are not the full project results.")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
            print("The saved model matches the recorded test results. No retraining was performed.")
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
