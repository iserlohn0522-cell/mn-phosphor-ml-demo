# Modeling Notes

This repository demonstrates a baseline engineering workflow for materials informatics. It is not presented as evidence of real phosphor discovery or predictive performance.

## Baseline Models

The demo includes two modeling paths:

- `scripts/train_baseline.py` trains a composition-only ridge regression model from element-fraction features.
- `scripts/train_model_suite.py` compares simple regression families, including mean baseline, ridge, elastic net, k-nearest neighbors, tree ensembles, support vector regression, and optional XGBoost when installed.

The default CI and install path avoid optional heavy dependencies. Matminer, Pymatgen, and XGBoost are optional extensions rather than required demo dependencies.

## Composition Features

Level 1 features are element fractions parsed from formula strings. Level 2 features add simple formula-derived descriptors such as element count, total atom count, Mn/O/F fractions, and weighted atomic-property summaries.

The descriptor code is intentionally conservative. For formulas such as `K2SiF6:Mn4+`, where the dopant amount is not specified, the workflow records the Mn4+ annotation but does not invent a numeric Mn fraction.

## Grouped Evaluation

Scientific datasets often contain related rows from the same paper, composition family, or source table. Random row-level splits can overstate performance when similar records appear in both train and validation folds.

The model suite supports grouped cross-validation through a group column such as `doi`. When enough groups are available, rows from the same group are kept out of the validation fold together. If grouping is not usable, the workflow falls back to ordinary K-fold cross-validation.

## Why Simple Models

Small scientific datasets can be noisy, sparse, and biased by reporting practices. Simple baselines are useful because they:

- establish whether the pipeline can run end to end;
- make feature leakage easier to notice;
- provide a reference point before using heavier models;
- are easier to audit when labels or sources change.

The synthetic demo labels are placeholders. They should not be cited as measurements or used to claim real predictive performance.
