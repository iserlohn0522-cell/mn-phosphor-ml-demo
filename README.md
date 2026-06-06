# Mn Phosphor ML Demo

[![CI](https://github.com/iserlohn0522-cell/mn-phosphor-ml-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/iserlohn0522-cell/mn-phosphor-ml-demo/actions/workflows/ci.yml)

Public portfolio demo for a Python workflow that collects literature metadata, normalizes material records, generates composition features, and runs small regression baselines for Mn-based phosphor research.

This repository is intentionally scoped as a safe engineering showcase. It does not include non-public datasets, unpublished reports, lab notebooks, cluster submission files, generated model artifacts, personal contact information, or draft research results.

## Portfolio Snapshot

| Area | Summary |
| --- | --- |
| Domain | Materials informatics / Mn phosphor research |
| Public scope | Synthetic data and dry-run literature metadata fixtures |
| Demonstrated skills | API client structure, schema design, material data normalization, composition features, model baselines, grouped evaluation, tests |
| Outputs | Candidate paper ranking, validated material table, descriptor table, baseline metrics |
| Private research boundary | No unpublished data, no API keys, no paper PDFs, no draft research conclusions |

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

## Documentation

- [Data schema](docs/data-schema.md)
- [Modeling notes](docs/modeling-notes.md)
- [Public demo scope](docs/public_scope.md)
- [Publication roadmap](docs/publication-roadmap.md)

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

## Example Workflow

Step 1: collect dry-run literature metadata.

```bash
python scripts/collect_literature.py --dry-run --max-results 2
```

Step 2: rank candidate papers for manual triage.

```bash
python scripts/rank_candidate_papers.py
```

Step 3: validate the synthetic material table and dry-run paper table.

```bash
python scripts/validate_dataset.py --materials data/examples/synthetic_materials.csv --papers data/interim/candidate_papers.csv
```

Step 4: generate formula-derived composition descriptors.

```bash
python scripts/generate_composition_descriptors.py --descriptor-backend basic
```

Step 5: train a small baseline regression model.

```bash
python scripts/train_baseline.py
```

Step 6: compare a lightweight model suite.

```bash
python scripts/train_model_suite.py --models mean_baseline ridge elastic_net knn
```

Run tests:

```bash
pytest
```

## Example Outputs

These files are generated locally and ignored by git. They are useful for checking that the workflow runs end to end.

| Output | Created by | Meaning |
| --- | --- | --- |
| `data/interim/candidate_papers.csv` | `collect_literature.py --dry-run` | Normalized paper metadata from dry-run fixtures |
| `data/interim/candidate_papers_review.csv` | `rank_candidate_papers.py` | Candidate paper ranking fields for manual triage |
| `data/processed/composition_descriptors.csv` | `generate_composition_descriptors.py` | Formula-derived descriptor table for the synthetic material records |
| `models/demo_baseline_emission_peak_nm/metrics.json` | `train_baseline.py` | Baseline cross-validation metrics on synthetic labels |
| `models/demo_baseline_emission_peak_nm/training_report.md` | `train_baseline.py` | Human-readable baseline training summary |

The synthetic labels are not literature measurements. Baseline metrics from this demo show that the engineering workflow runs; they should not be interpreted as scientific model performance.

## Public Scope

The included CSV is synthetic and exists only to exercise the pipeline. Numeric labels should not be interpreted as literature measurements or research conclusions.

Real API keys or contact emails should be placed in a local `.env` file only. The `.env` file is ignored by git. The demo and CI do not require real API keys.
