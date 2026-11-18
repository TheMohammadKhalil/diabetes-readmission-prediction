import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
RESULTS = ROOT / "results"
NAMES = {
    "majority_baseline": "Baseline",
    "logistic_regression": "Logistic regression",
    "logistic_balanced": "Balanced logistic",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "neural_network": "Neural network",
}
BLUE, ORANGE, GREEN = "#356b9b", "#c66a22", "#287b63"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "legend.fontsize": 7, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def threshold_counts(target, scores, thresholds):
    order = np.argsort(scores)
    labels = np.asarray(target)[order]
    cumulative = np.r_[0, np.cumsum(labels)]
    index = np.searchsorted(np.asarray(scores)[order], thresholds, side="left")
    tp = cumulative[-1] - cumulative[index]
    fp = len(labels) - index - tp
    fn = cumulative[-1] - tp
    tn = len(labels) - cumulative[-1] - fp
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=tp + fp != 0)
    recall = tp / cumulative[-1]
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=2 * tp + fp + fn != 0)
    return {"threshold": thresholds.tolist(), "tp": tp.tolist(), "fp": fp.tolist(),
            "fn": fn.tolist(), "tn": tn.tolist(), "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist()}


def make_workflow(audit):
    fig, ax = plt.subplots(figsize=(7.1, 3.65))
    ax.set(xlim=(0, 10), ylim=(0, 6.5))
    ax.axis("off")

    def box(x, y, w, h, text, color="#edf3f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="#8b9aa7", linewidth=.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#4b5660", "lw": .9})

    box(3.1, 5.7, 3.8, .6, f"{audit['raw_rows']:,} original encounters")
    box(3.1, 4.6, 3.8, .6, f"{audit['eligible_rows']:,} eligible encounters")
    arrow((5, 5.62), (5, 5.28))
    ax.text(7.2, 5.4, "2,423 death/hospice\nencounters excluded", va="center", fontsize=7.5)
    box(.3, 3.4, 4.1, .65, "Training: 79,625 encounters\n55,952 patients")
    box(6.1, 3.4, 3.6, .65, "Held-out test: 19,718 encounters\n14,038 patients", "#f6f0e7")
    arrow((4.2, 4.52), (2.35, 4.13))
    arrow((5.8, 4.52), (7.9, 4.13))
    ax.text(5, 4.23, "Split by patient", ha="center", fontsize=8)
    for row in range(5):
        ax.text(.45, 2.95 - row * .31, f"Fold {row + 1}", ha="left", va="center", fontsize=7.5)
        for col in range(5):
            ax.add_patch(plt.Rectangle((1.4 + col * .52, 2.84 - row * .31), .46, .23,
                                      color=ORANGE if col == row else BLUE))
    ax.text(.45, 1.43, "Blue: fit     Orange: validate", fontsize=7.5)
    arrow((2.35, 3.32), (2.35, 3.12))
    box(.3, .15, 4.1, .85, "Choose model and threshold\nusing training CV / OOF predictions")
    arrow((2.35, 1.3), (2.35, 1.08))
    box(6.1, .15, 3.6, .85, "Evaluate frozen choices\non held-out patients", "#edf5ef")
    arrow((4.48, .58), (6.02, .58))
    arrow((7.9, 3.32), (7.9, 1.08))
    ax.text(7.9, 2.25, "No shared patients\nwith training", ha="center", va="center")
    fig.tight_layout(pad=.4)
    save(fig, "workflow")


def make_static(audit, cv, folds, test, oof, selection):
    selected = selection["model"]
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    labels = [f"Train\n(n={audit['train_rows']:,})", f"Test\n(n={audit['test_rows']:,})"]
    rates = np.array([audit["train_positive_rate"], audit["test_positive_rate"]]) * 100
    ax.barh(labels, 100 - rates, color=BLUE, label="Not within 30 days")
    ax.barh(labels, rates, left=100 - rates, color=ORANGE, label="Within 30 days")
    for i, rate in enumerate(rates):
        ax.text(40, i, f"{100-rate:.1f}%", color="white", ha="center", va="center")
        ax.text(100-rate/2, i, f"{rate:.1f}%", color="white", ha="center", va="center", fontsize=7)
    ax.set(xlim=(0, 100), xlabel="Share of encounters (%)")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.27), ncol=1, frameon=False)
    fig.tight_layout()
    save(fig, "class_balance")

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for i, row in enumerate(cv.itertuples()):
        values = folds.loc[folds.model == row.model, "average_precision"].to_numpy()
        assert np.isclose(values.mean(), row.cv_ap_mean)
        ax.scatter(values, i + np.linspace(-.12, .12, len(values)), color="#9ba6af", s=13, zorder=2)
        ax.errorbar(row.cv_ap_mean, i, xerr=row.cv_ap_std, fmt="s", markersize=4,
                    capsize=3, color=GREEN if row.model == selected else BLUE, zorder=3)
    ax.set(yticks=range(len(cv)), yticklabels=[NAMES[x] for x in cv.model],
           xlabel="Training CV average precision", xlim=(.095, .265))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    save(fig, "cv_comparison")

    tuned = test[test.operating_point == "validation_threshold"].set_index("model").loc[list(NAMES)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)
    colors = [GREEN if name == selected else BLUE for name in NAMES]
    for ax, metric, label, maximum in zip(axes, ["average_precision", "recall"],
                                         ["Average precision", "Recall at saved threshold"], [.29, .6]):
        ax.barh(list(NAMES.values()), tuned[metric], color=colors)
        for i, value in enumerate(tuned[metric]):
            ax.text(value + maximum * .015, i, f"{value:.3f}", va="center", fontsize=7.5)
        ax.set(xlabel=label, xlim=(0, maximum))
        ax.grid(axis="x", alpha=.15)
    axes[0].axvline(audit["test_positive_rate"], color="#555555", ls="--", lw=.8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "test_comparison")

    thresholds = np.unique(np.r_[np.linspace(0, 1, 201), selection["thresholds"][selected]])
    counts = threshold_counts(oof.target, oof[selected], thresholds)
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for metric, color in [("precision", BLUE), ("recall", ORANGE), ("f1", GREEN)]:
        ax.plot(thresholds, counts[metric], label=metric.capitalize(), color=color)
    chosen = selection["thresholds"][selected]
    ax.axvline(chosen, color="#555555", ls="--", lw=.9)
    ax.axvline(.5, color="#999999", ls=":", lw=.9)
    ax.text(chosen + .012, .97, f"Saved\n{chosen:.4f}", va="top", fontsize=7)
    ax.text(.49, .97, "Default\n0.5", ha="right", va="top", fontsize=7)
    ax.set(xlim=(0, .6), ylim=(0, 1.02), xlabel="Threshold", ylabel="Training OOF score")
    ax.legend(loc="center right", frameon=False)
    ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, "threshold_tradeoff")

    rows = test[test.model == selected].set_index("operating_point")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for ax, key, title in zip(axes, ["default_0.5", "validation_threshold"], ["Default cutoff", "Saved cutoff"]):
        row = rows.loc[key]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        percent = matrix / matrix.sum(axis=1, keepdims=True) * 100
        ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i,j]:,}\n{percent[i,j]:.1f}% of row", ha="center", va="center",
                        color="white" if percent[i,j] > 55 else "#222222", fontsize=8)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not <30", "<30"], yticklabels=["Not <30", "<30"],
               xlabel=f"Predicted class\nPrecision {row.precision:.3f}  |  Recall {row.recall:.3f}",
               ylabel="Actual class", title=f"{title}: {row.threshold:.4f}")
    fig.tight_layout()
    save(fig, "confusion_comparison")


def interactive_data(cv, folds, test, oof, selection, audit):
    curves = {}
    for name in NAMES:
        grid = np.unique(np.r_[np.linspace(0, 1, 101), selection["thresholds"][name]])
        curves[name] = threshold_counts(oof.target, oof[name], grid)
    return {
        "names": NAMES, "selected": selection["model"], "saved": selection["thresholds"],
        "curves": curves, "cv": cv[["model", "cv_ap_mean", "cv_ap_std"]].to_dict("records"),
        "folds": folds[["model", "fold", "average_precision"]].to_dict("records"),
        "test": test.to_dict("records"), "audit": audit,
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    audit = json.loads((RESULTS / "split_audit.json").read_text())
    selection = json.loads((RESULTS / "selection.json").read_text())
    cv = pd.read_csv(RESULTS / "cv_summary.csv")
    folds = pd.read_csv(RESULTS / "cv_fold_metrics.csv")
    test = pd.read_csv(RESULTS / "test_metrics.csv")
    oof = pd.read_csv(RESULTS / "train_oof_predictions.csv", float_precision="round_trip")
    make_workflow(audit)
    make_static(audit, cv, folds, test, oof, selection)
    data = interactive_data(cv, folds, test, oof, selection, audit)
    template = (ROOT / "docs" / "interactive_template.html").read_text()
    html = template.replace("__PLOTLY_LIBRARY__", get_plotlyjs()).replace("__RESULT_DATA__", json.dumps(data))
    (ROOT / "docs" / "interactive_results.html").write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n")
    (ROOT / "docs" / "figures" / "interactive_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Created six PDF figures and the offline interactive results document.")
