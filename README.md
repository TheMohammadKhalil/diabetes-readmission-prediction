# Diabetes Readmission Prediction

Can we use a patient's hospital record to predict whether they will return within 30 days?

This is our machine learning course project. We use Python and scikit-learn to compare
logistic regression, a decision tree, random forest, gradient boosting, and a small
neural network on the UCI diabetes dataset.

We use roughly **80% of the data for training and 20% for testing**, with **5-fold
cross-validation** during training. A patient's records always stay in the same group.

## Get started

Use Python 3.11 or 3.12. On Linux or macOS:

```bash
git clone https://github.com/TheMohammadKhalil/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the models, then check the winner on the test set:

```bash
python run_experiment.py --output my_run
python test_model.py --results my_run
```

The dataset downloads automatically. The full experiment took about 11 minutes on
our machine. Use a new output folder each time you train.

To try one example record:

```bash
python test_model.py --results my_run --input examples/patient.csv
```

The example is made up for demonstration. Predictions are saved in
`predictions/predictions.csv`: **1** means predicted readmission within 30 days;
**0** means no predicted readmission within that window.

## Our results

After cleaning, we used **99,343 hospital encounters**. Gradient boosting was selected
using cross-validation and achieved these results on the held-out test set:

| Measure | Result |
| --- | --- |
| ROC AUC | 0.680 |
| Average precision | 0.239 |
| Recall | 45.9% |
| F1 score | 0.289 |

Accuracy alone is misleading here: predicting “no readmission” for everyone gives
88.9% accuracy but misses every positive case. The report explains the tradeoffs.

## Explore the project

- [Project paper — IEEE-format PDF](docs/project_paper.pdf)
- [Interactive graphs](docs/interactive_results.html) — download the file and open it in your browser
- [How to test the model](docs/testing.md)
- [Notebook walkthrough](notebooks/readmission_project.ipynb)
- [Full results and plots](results/report.md)
- [How the project works](docs/methodology.md)
- [Edit or rebuild the paper](docs/paper.md)

The saved model and raw dataset are kept locally, so train once after cloning.
The `results/` folder contains the results from our original full run.

## Quick checks

```bash
python -m pytest -q
python run_experiment.py --quick --output quick_check
```

The first command checks the code. The second runs a smaller experiment to check
that everything works; its scores are not the final project results.

## Dataset

[Diabetes 130-US Hospitals for Years 1999–2008](https://doi.org/10.24432/C5230J),
by Clore, Cios, DeShazo, and Strack, UCI Machine Learning Repository. License: CC BY 4.0.

This is an academic project, not a clinical decision tool.
