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

The editable source is `project_paper.tex`. The three figures are in `figures/`.
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
the saved test predictions and forest importances.

## Read alongside the paper

- [Testing instructions](testing.md): exact commands and expected results.
- [Methodology](methodology.md): preparation rules, models, metrics, and scope changes.
- [Results report](../results/report.md): original tables and plots.
- [Presentation guide](presentation_guide.md): questions to prepare for your discussion.

The paper reports a fixed completed experiment. If you later change the methods or
run a new experiment, update its tables, figures, and discussion together.
