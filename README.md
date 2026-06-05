# Mn Phosphor ML Demo

Public portfolio demo for a Python workflow that collects literature metadata, normalizes material records, generates composition features, and runs small regression baselines for Mn-based phosphor research.

This repository is intentionally scoped as a safe engineering showcase. It does not include non-public datasets, unpublished reports, lab notebooks, cluster submission files, generated model artifacts, personal contact information, or draft research results.

## What This Demo Shows

- Typed Python package layout under `src/phosphor_ml`.
- Literature search clients for OpenAlex, Semantic Scholar, and Crossref.
- Dry-run literature collection using local fixtures, so the demo works without API keys.
- Canonical paper and material schemas.
- Rule-based candidate paper ranking for manual review triage.
- Composition parsing and feature generation from chemical formulas.
- Small supervised regression pipelines with cross-validation and grouped split support.
- Unit tests for schema validation, formula parsing, feature generation, literature normalization, and model evaluation.

## Repository Layout

```text
mn-phosphor-ml-demo/
  data/
    examples/synthetic_materials.csv
    raw/literature/.gitkeep
    interim/.gitkeep
    processed/.gitkeep
  scripts/
    collect_literature.py
    rank_candidate_papers.py
    validate_dataset.py
    generate_composition_descriptors.py
    train_baseline.py
    train_model_suite.py
  src/phosphor_ml/
  tests/
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run The Demo

Dry-run literature collection:

```bash
python scripts/collect_literature.py --dry-run --max-results 2
python scripts/rank_candidate_papers.py
```

Validate the included synthetic material table:

```bash
python scripts/validate_dataset.py --materials data/examples/synthetic_materials.csv --papers data/interim/candidate_papers.csv
```

Generate composition descriptors:

```bash
python scripts/generate_composition_descriptors.py --descriptor-backend basic
```

Train the small demo baseline:

```bash
python scripts/train_baseline.py
python scripts/train_model_suite.py --models mean_baseline ridge elastic_net knn
```

Run tests:

```bash
pytest
```

## Public Scope

The included CSV is synthetic and exists only to exercise the pipeline. Numeric labels should not be interpreted as literature measurements or research conclusions.

Real API keys or contact emails should be placed in a local `.env` file only. The `.env` file is ignored by git.
