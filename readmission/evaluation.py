"""Metrics, validation-only threshold selection, and report figures."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def choose_threshold(y, probabilities):
    """Maximize positive-class F1 on training out-of-fold predictions only."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    # Resolve ties in favour of a higher (more conservative) threshold.
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def compute_metrics(y, probabilities, threshold=0.5):
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "brier_score": brier_score_loss(y, probabilities),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_eda(raw_train, X_train, y_train, directory):
    """Explore training records only, after setting aside the test patients."""
    missing = raw_train.isna().mean().sort_values(ascending=False)
    missing.rename("missing_fraction").to_csv(directory / "training_missingness.csv")
    X_train.describe(include="all").to_csv(directory / "training_feature_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    y_train.value_counts().sort_index().plot.bar(ax=axes[0], color=["#4c78a8", "#e45756"])
    axes[0].set(title="Training class counts", xlabel="Readmitted within 30 days", ylabel="Encounters")
    axes[0].tick_params(axis="x", rotation=0)
    missing.head(8).sort_values().plot.barh(ax=axes[1], color="#4c78a8")
    axes[1].set(title="Training missingness", xlabel="Fraction missing")
    pd.DataFrame({"prior_inpatient": X_train.number_inpatient.clip(upper=5), "target": y_train}).groupby(
        "prior_inpatient"
    ).target.mean().plot.bar(ax=axes[2], color="#72b7b2")
    axes[2].set(title="Prior inpatient visits (5 = 5+)", xlabel="Visits", ylabel="Readmission fraction")
    fig.tight_layout()
    fig.savefig(directory / "eda.png", dpi=160)
    plt.close(fig)


def save_evaluation_plots(y, predictions, thresholds, test_results, directory):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, probabilities in predictions.items():
        fpr, tpr, _ = roc_curve(y, probabilities)
        precision, recall, _ = precision_recall_curve(y, probabilities)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, probabilities):.3f})")
        axes[1].step(recall, precision, where="post",
                     label=f"{name} ({average_precision_score(y, probabilities):.3f})")
        observed, predicted = calibration_curve(y, probabilities, n_bins=8, strategy="quantile")
        axes[2].plot(predicted, observed, marker="o", markersize=3, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axhline(np.mean(y), color="black", linestyle="--", alpha=0.4)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="Holdout ROC (legend: AUROC)", xlabel="False positive rate", ylabel="Recall")
    axes[1].set(title="Holdout precision–recall (legend: AP)", xlabel="Recall", ylabel="Precision")
    axes[2].set(title="Holdout calibration", xlabel="Mean predicted probability", ylabel="Observed fraction")
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(directory / "roc_pr_calibration.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, probabilities) in zip(axes.flat, predictions.items()):
        labels = (probabilities >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(y, labels, labels=[0, 1],
                                               ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nt = {thresholds[name]:.3f}", fontsize=10)
    for ax in list(axes.flat)[len(predictions):]:
        ax.set_visible(False)
    fig.suptitle("Holdout confusion matrices; 1 = readmitted within 30 days")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    tuned = test_results[test_results.operating_point == "validation_threshold"]
    ax = tuned.set_index("model")[["roc_auc", "average_precision", "recall", "f1"]].plot.bar(
        figsize=(12, 5), rot=25
    )
    ax.set(title="Holdout results — model selection used training CV only", ylabel="Score", ylim=(0, 1))
    ax.figure.tight_layout()
    ax.figure.savefig(directory / "model_comparison.png", dpi=160)
    plt.close(ax.figure)


def save_importance(pipeline, directory):
    """Forest impurity importance is descriptive, not causal or unbiased."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    table = pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    table.to_csv(directory / "forest_feature_importance.csv", index=False)
    ax = table.head(20).iloc[::-1].plot.barh(x="feature", y="importance", legend=False, figsize=(10, 7))
    ax.set(title="Random forest: top 20 encoded features (impurity importance)", ylabel="")
    ax.figure.tight_layout()
    ax.figure.savefig(directory / "feature_importance.png", dpi=160)
    plt.close(ax.figure)
