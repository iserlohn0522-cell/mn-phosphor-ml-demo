# Mn Phosphor ML Demo

[![CI](https://github.com/iserlohn0522-cell/mn-phosphor-ml-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/iserlohn0522-cell/mn-phosphor-ml-demo/actions/workflows/ci.yml)

A compact Python project for literature triage and baseline modeling in Mn phosphor materials informatics.

The repo shows how I structure a research codebase before private data enters the loop: schema-first tables, fixture-backed API clients, formula features, grouped evaluation, and tests. It uses synthetic material rows and dry-run literature fixtures. Lab notes, non-public data, raw papers, PDFs, keys, generated model artifacts, and draft conclusions stay out of this public repo.

## Portfolio Snapshot

| Area | Summary |
| --- | --- |
| Domain | Materials informatics for Mn phosphor research |
| Data in repo | Synthetic material table plus dry-run literature metadata fixtures |
| Skills shown | API clients, schema design, material data normalization, composition features, baseline models, grouped evaluation, tests |
| Generated outputs | Candidate ranking, schema check, descriptor table, baseline metrics |
| Boundary | Fixture data in git; lab data, keys, paper PDFs, raw downloads, and draft conclusions stay local |

## What Reviewers Can Inspect

- Typed Python package layout under `src/phosphor_ml`.
- Literature search clients for OpenAlex, Semantic Scholar, and Crossref.
- Dry-run literature collection with local fixtures, so the demo works without API keys.
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
- [Repository boundary](docs/public_scope.md)
- [Release plan](docs/publication-roadmap.md)

## Install

From the repository root:

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

Run the commands below from the repository root. They use fixture data and the synthetic material table.

Step 1: collect literature metadata from dry-run fixtures.

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

The demo writes these files during a local run. Git ignores them, so cleanup returns the repository to source files and fixture data.

| Output | Created by | Meaning |
| --- | --- | --- |
| `data/interim/candidate_papers.csv` | `collect_literature.py --dry-run` | Paper metadata from dry-run fixtures |
| `data/interim/candidate_papers_review.csv` | `rank_candidate_papers.py` | Candidate paper ranking fields for manual triage |
| `data/processed/composition_descriptors.csv` | `generate_composition_descriptors.py` | Formula-derived descriptors for synthetic material rows |
| `models/demo_baseline_emission_peak_nm/metrics.json` | `train_baseline.py` | Baseline cross-validation metrics on synthetic labels |
| `models/demo_baseline_emission_peak_nm/training_report.md` | `train_baseline.py` | Baseline training summary |

The synthetic labels do not come from literature. Use the metrics to check the engineering workflow. They say nothing about phosphor performance.

## Repository Boundary

This repo keeps public demo code separate from research material that needs permission, review, or publication clearance. The committed CSV gives the scripts enough rows to run as fixture data. Released scientific datasets belong in a separate, licensed release.

Put real API keys or contact emails in a local `.env` file. Git ignores `.env`, and the demo and CI run without it.

## Acknowledgments

This demo is inspired by a Mn phosphor research direction suggested by Prof. Peifen Zhu.
