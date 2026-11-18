# Project paper

[Read the IEEE-format PDF](project_paper.pdf).

The paper uses the `IEEEtran` conference class: two columns, numbered sections,
numbered references, figures, and tables. Its organization follows the supplied
student examples: abstract, introduction, related work, dataset, methodology,
results, discussion, and conclusion. It also includes reproducibility instructions.
It is a course project paper, not an IEEE-published or peer-reviewed article.

The results come from the completed full experiment in `results/`. Table values
were copied directly from the result CSVs. The literature section uses the source
papers and clearly separates their protocols from ours. No new model scores or
paper-replication experiments have been invented.

## Edit the paper

The editable source is `project_paper.tex`. The nine figures are in `figures/`.
Upload the `.tex` file and that folder to an Overleaf project, then set
`project_paper.tex` as the main document. The IEEEtran class is available in standard
TeX distributions. The author names are taken from the proposal; add your department,
university affiliation, or student emails if your instructor requires them.

To compile locally with a TeX distribution that includes IEEEtran:

```bash
cd docs
pdflatex project_paper.tex
pdflatex project_paper.tex
```

Alternatively, use Tectonic:

```bash
cd docs
tectonic project_paper.tex
```

The supplied PDF was built with Tectonic 0.17.0. LaTeX is needed only to rebuild the
paper, not to train or test the Python models. The figures are vector PDFs made from
the saved split summaries, cross-validation results, predictions, and forest importances.

## Explore the graphs

The paper now illustrates the patient split and five-fold workflow, class imbalance,
fold-by-fold model comparison, held-out model scores, threshold tradeoff, confusion
matrices, ROC and precision–recall curves, calibration, and feature importance.
Captions explain what each figure shows and whether it uses training or test data.

[Download the interactive companion](interactive_results.html) using GitHub's
**Download raw file** button, then open the downloaded HTML in your browser. After
cloning the repository, you can simply open `docs/interactive_results.html`.
GitHub's file preview does not run the controls. No server, Python installation,
or internet connection is needed to explore the downloaded file.

- Hover over validation scores to inspect individual folds.
- Choose a metric to compare the frozen models on the held-out set.
- Choose a model and move the cutoff slider to see precision, recall, F1, and
  confusion counts change on **training out-of-fold predictions**.
- Use **Saved cutoff** or **Use 0.5** to compare the two choices.

The slider deliberately uses training predictions. It does not tune a threshold
on test data or change the paper's reported results. Only aggregate results are
included; no individual patient records are embedded.

To rebuild the six new figures and the companion from the original local `results/`
folder, activate the project environment and run from the repository root:

```bash
pip install -r docs/requirements.txt
python docs/make_figures.py
```

The generator needs `results/train_oof_predictions.csv`, which stays local and is
created by a full experiment. Cloning alone is enough to read the supplied figures
and HTML, but not to regenerate them. `interactive_template.html` is the editable
companion source; `figures/interactive_data.json` contains its aggregate chart data.
The three original figures remain the exports from the completed experiment.

## Read alongside the paper

- [Testing instructions](testing.md): exact commands and expected results.
- [Methodology](methodology.md): preparation rules, models, metrics, and scope changes.
- [Results report](../results/report.md): original tables and plots.
- [Presentation guide](presentation_guide.md): questions to prepare for your discussion.

The paper reports a fixed completed experiment. If you later change the methods or
run a new experiment, update its tables, figures, and discussion together.
