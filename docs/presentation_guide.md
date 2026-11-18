# Explaining and presenting the project

Start with [the project paper](project_paper.pdf), `results/report.md`, and the
notebook. Read the discussion and make sure your group can explain the measured results.

## Suggested paper structure

1. **Abstract:** state the task, cohort size, patient-separated 80/20 split, five-fold
   training CV, model families, and the CV-selected model's held-out AUROC/AP/recall/F1.
2. **Introduction:** motivate readmission prediction at discharge and explain why
   89% accuracy can still mean missing every early readmission.
3. **Related work:** discuss the studies in the proposal after reading their methods.
   Put results from other cohorts/splits in a separate literature table. Do not imply
   that the sample students' diabetes classification task is the same as readmission.
4. **Dataset and preparation:** source, target mapping, exclusions, missingness, mixed
   features, grouped diagnoses, and the two derived features. Include `eda.png`.
5. **Methods:** show the split/fold workflow, model table, small grids, class weighting,
   selection by AP, and threshold choice using training OOF predictions.
6. **Results:** CV mean/std table and a separate holdout table. Include PR/ROC curves,
   confusion matrices, and a brief calibration/feature-importance discussion.
7. **Discussion and conclusion:** interpret false positives/negatives, compare ordinary
   and balanced logistic regression, acknowledge modest predictive signal and scope
   changes, and explain what future external validation would require.

## Questions you should be able to answer

- **Why not split rows randomly?** A patient may have multiple rows. Sharing that
  patient's history across train and test weakens the independence of the evaluation.
- **Where does five-fold CV happen?** Only inside the training 80%; the held-out 20%
  is untouched until the models, settings, and thresholds are fixed.
- **Why is the split not exactly 80.000%?** Patients are indivisible groups of unequal
  size. The exact proportions appear in `split_audit.json`.
- **Why use AP rather than accuracy to choose a winner?** The positive class is rare;
  AP focuses on the precision/recall ranking of that class.
- **What does class weighting do?** It makes mistakes on the minority class more
  costly during fitting. It does not change test prevalence or guarantee calibration.
- **Why lower the threshold?** A 0.5 cutoff may miss nearly all positive encounters.
  A threshold selected on training OOF predictions trades precision for recall.
- **Are the CV scores unbiased?** They are development estimates used for parameter
  selection. The untouched holdout assesses the selected model independently.
- **Which result is the headline?** The holdout results of the model selected using
  training CV, even if another model happens to score better on this particular test.
- **Is feature importance medical evidence?** No; it describes this model's use of
  inputs and can reflect correlations, coding practices, and bias.
- **Did we reproduce all published papers?** No. The implementation runs its own
  documented protocol and explicitly narrows the original proposal's ambitious scope.

## Before submission

Confirm that you are using `results/`, not the quick-check folder. Read any model
warnings, use measured values rather than proposal expectations, cite the dataset
and source paper, and make sure every group member can explain the split, pipeline,
metrics, and threshold choice. Do not tune again based on which test score looks best.
