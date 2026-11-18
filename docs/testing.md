# How to test the model

There are three different things you can do: check the trained model's performance,
try a new record, or check that the code works.

## 1. Check the trained model

On the original project folder, the trained model already exists. Activate the
Python environment and run:

```bash
source .venv/bin/activate
python test_model.py
```

This loads `results/models/selected_model.joblib` and evaluates it only on the
19,718 encounters reserved for testing. It does not train a new model or choose a
new threshold. It checks that the dataset matches the original file and that the
scores match the saved results.

Expected results for the included full experiment:

| Measure | Expected value |
| --- | --- |
| Model | gradient_boosting |
| Test encounters | 19,718 |
| Threshold | 0.1383 |
| ROC AUC | 0.6798 |
| Average precision | 0.2386 |
| Accuracy | 0.7495 |
| Precision | 0.2111 |
| Recall | 0.4593 |
| F1 | 0.2893 |
| True negatives / false positives | 13,774 / 3,756 |
| False negatives / true positives | 1,183 / 1,005 |

The final line should say that the saved model matches the recorded test results.
This reproduces the original evaluation; it is not a new independent test dataset.
Avoid changing model settings based on this test set.

### If you just cloned the repository

Raw data, model files, and patient-level split files are not uploaded to GitHub.
Create them by running the full experiment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --output my_run
python test_model.py --results my_run
```

On Windows, create the environment with `py -m venv .venv` and activate it with
`.venv\Scripts\activate` in Command Prompt. The remaining Python commands are the same.

The script downloads the public dataset if needed. To use a file you downloaded
manually, pass `--data /path/to/diabetic_data.csv` to both training and testing.
A complete run took about 11 minutes on the original machine. For the exact original
package versions, install `requirements-lock.txt` instead of `requirements.txt`.

Choose a new folder name for a later run, such as `my_second_run`; existing runs are
protected from being overwritten. If you choose another name, use that same name
in `--results` when testing.

## 2. Try your own input

Start with the supplied **synthetic** record in `examples/patient.csv`:

```bash
python test_model.py --input examples/patient.csv
```

For a model trained into `my_run`, use:

```bash
python test_model.py --results my_run --input examples/patient.csv
```

Open `predictions/predictions.csv` to see the output:

| Column | Meaning |
| --- | --- |
| row | Input row number, starting at 1 |
| model_score | The classifier's estimated probability score |
| predicted_readmission | 1 if the score reaches the saved threshold; otherwise 0 |

A score of 0.20 with a threshold of 0.1383 produces class 1. The score is an estimate,
not a certainty; this project has not established clinical reliability.

Copy the example CSV and edit it, keeping all 47 column names. Add more rows to test
several records. Identifiers and the `readmitted` label are not needed. Use `?` for
unknown values. Keep the exact lab value `None` when a test was not performed. Use
the original UCI categories and units: age bands such as `[50-60)`, medication values
`No`, `Steady`, `Up`, or `Down`, and the administrative codes in `IDS_mapping.csv`.
The example's values are invented; they are not a real patient or a labelled test case.

```bash
python test_model.py --input my_patients.csv --output predictions/my_predictions.csv
```

Use a new output filename for another prediction run. Death/hospice encounters are
rejected because they are outside the study cohort. If an input file contains a
`readmitted` column, prediction ignores it: this mode does not calculate accuracy.
Predictions without known outcomes demonstrate that the program works, but do not
measure predictive performance. For a new performance estimate, collect a separate
labelled dataset with patients absent from training and keep the model and threshold fixed.

## 3. Check the code and installation

```bash
python -m pytest -q
python run_experiment.py --quick --output quick_check
python test_model.py --results quick_check
```

The tests check label handling, missing values, diagnosis grouping, patient separation,
and evaluation. The quick experiment uses 4,000 patients and three CV folds, so its
results differ from the full experiment. A quick MLP run can hit its shorter iteration
limit; warnings are recorded in its output folder.

## Common problems

| Message | What to do |
| --- | --- |
| No saved model | Train first and point `--results` to that run's folder. |
| Dataset differs | Use the unchanged source CSV used for that run. |
| Missing input columns | Start from `examples/patient.csv` and keep its headers. |
| Output already exists | Choose another output folder or prediction filename. |
| Model results do not match | Check that the model, metadata, and CSVs are from the same run; use the locked environment. |

The notebook also explains the workflow. Its default results folder is `results/`;
change the `RESULTS` variable to `ROOT / "my_run"` when viewing a new run. Keep
`RUN_FULL_EXPERIMENT = False` if you only want to display existing results.
