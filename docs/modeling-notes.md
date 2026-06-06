# Modeling Notes

This repo shows the modeling scaffold I would use before bringing in a licensed or curated phosphor dataset. The synthetic-label results check code paths. They do not measure discovery or predictive performance.

## Baseline Models

The demo has two modeling paths:

- `scripts/train_baseline.py` trains a composition-only ridge regression model from element-fraction features.
- `scripts/train_model_suite.py` compares simple regression families, including mean baseline, ridge, elastic net, k-nearest neighbors, tree ensembles, support vector regression, and optional XGBoost.

CI installs the default package plus test dependencies. Matminer, Pymatgen, and XGBoost stay optional.

## Composition Features

Level 1 features use element fractions parsed from formula strings. Level 2 adds formula-derived descriptors such as element count, total atom count, Mn/O/F fractions, and weighted atomic-property summaries.

For formulas such as `K2SiF6:Mn4+`, the dopant amount does not appear in the formula. The descriptor code records the Mn4+ annotation and leaves the Mn fraction at zero rather than inventing a value.

## Grouped Evaluation

Scientific datasets often contain related rows from the same paper, composition family, or source table. Random row-level splits can make a model look better when similar rows land in both train and validation folds.

The model suite can use a group column such as `doi`. With enough groups, it keeps rows from the same group in the same fold. If the CSV lacks usable groups, it uses K-fold cross-validation.

## Why Simple Models

Small scientific datasets can be noisy, sparse, and shaped by reporting bias. Simple baselines help because they:

- check that the pipeline runs end to end;
- make feature leakage easier to spot;
- give heavier models a reference point;
- keep label and source changes easy to audit.

The synthetic labels are placeholders. Do not cite them as measurements or use them to claim predictive performance.
