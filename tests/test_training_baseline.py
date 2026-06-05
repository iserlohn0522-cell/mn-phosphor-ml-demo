from __future__ import annotations

import csv

import numpy as np

from phosphor_ml.training.baseline import (
    CrossValidationResult,
    build_element_vocabulary,
    cross_validate_ridge,
    describe_training_rows,
    featurize_formulas,
    load_labeled_rows,
    parse_formula_counts,
)


def test_parse_formula_counts_handles_decimals_and_dopant_notation():
    assert parse_formula_counts("K2SiF6:Mn4+") == {"K": 2.0, "Si": 1.0, "F": 6.0, "Mn": 1.0}
    assert parse_formula_counts("Ba2TiGe1.997O8Mn0.003") == {
        "Ba": 2.0,
        "Ti": 1.0,
        "Ge": 1.997,
        "O": 8.0,
        "Mn": 0.003,
    }


def test_featurize_formulas_uses_element_fraction_features():
    formulas = ["K2SiF6:Mn4+", "CaAlSiN3:Mn"]
    vocabulary = build_element_vocabulary(formulas)
    features = featurize_formulas(formulas, vocabulary)

    assert vocabulary == ["Al", "Ca", "F", "K", "Mn", "N", "Si"]
    assert features.shape == (2, 7)
    assert np.isclose(features[0, vocabulary.index("F")], 6 / 10)
    assert np.isclose(features[0].sum(), 1.0)


def test_load_labeled_rows_filters_missing_or_non_numeric_targets(tmp_path):
    path = tmp_path / "materials.csv"
    rows = [
        {"material_id": "m1", "composition_normalized": "K2SiF6:Mn4+", "emission_peak_nm": "631"},
        {"material_id": "m2", "composition_normalized": "CaAlSiN3:Mn", "emission_peak_nm": ""},
        {"material_id": "m3", "composition_normalized": "BaTiF6:Mn4+", "emission_peak_nm": "not numeric"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_labeled_rows(path, target_column="emission_peak_nm")

    assert loaded == [{"material_id": "m1", "source_dataset": "", "formula": "K2SiF6:Mn4+", "target": 631.0}]


def test_cross_validate_ridge_returns_fold_metrics_and_predictions():
    formulas = [
        "K2SiF6:Mn4+",
        "BaTiF6:Mn4+",
        "CaAlSiN3:Mn",
        "Zn2GeO4:Mn",
        "Mg4Ta2O9:Mn4+",
        "Ca3Al2O6:Mn",
    ]
    targets = np.array([631.0, 632.0, 650.0, 525.0, 660.0, 610.0])
    vocabulary = build_element_vocabulary(formulas)
    features = featurize_formulas(formulas, vocabulary)

    result = cross_validate_ridge(features, targets, k_folds=3, alpha=1.0, random_seed=7)

    assert isinstance(result, CrossValidationResult)
    assert result.predictions.shape == targets.shape
    assert len(result.fold_metrics) == 3
    assert result.overall_metrics["n_samples"] == 6
    assert result.overall_metrics["rmse"] >= 0
    assert result.overall_metrics["mae"] >= 0


def test_describe_training_rows_counts_sources_and_target_range():
    rows = [
        {"material_id": "a", "source_dataset": "synthetic_demo", "formula": "K2SiF6:Mn4+", "target": 631.0},
        {"material_id": "b", "source_dataset": "synthetic_demo", "formula": "BaTiF6:Mn4+", "target": 632.0},
        {"material_id": "c", "source_dataset": "demo_review", "formula": "CaAlSiN3:Mn", "target": 650.0},
    ]

    summary = describe_training_rows(rows, target_column="emission_peak_nm")

    assert summary["target_column"] == "emission_peak_nm"
    assert summary["n_rows"] == 3
    assert summary["target_min"] == 631.0
    assert summary["target_max"] == 650.0
    assert summary["source_counts"] == {"demo_review": 1, "synthetic_demo": 2}
