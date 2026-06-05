from __future__ import annotations

import csv

from phosphor_ml.training.model_suite import (
    descriptor_feature_columns,
    format_model_suite_report,
    load_descriptor_feature_rows,
    load_supervised_rows,
    run_model_suite,
    select_best_model,
)


def test_load_supervised_rows_filters_to_numeric_target(tmp_path):
    path = tmp_path / "materials.csv"
    rows = [
        {
            "material_id": "m1",
            "source_dataset": "synthetic_demo",
            "composition_normalized": "K2SiF6:Mn4+",
            "emission_peak_nm": "631",
        },
        {
            "material_id": "m2",
            "source_dataset": "synthetic_demo",
            "composition_normalized": "BaTiF6:Mn4+",
            "emission_peak_nm": "",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_supervised_rows(path, target_column="emission_peak_nm")

    assert loaded == [
        {
            "material_id": "m1",
            "source_dataset": "synthetic_demo",
            "formula": "K2SiF6:Mn4+",
            "target": 631.0,
        }
    ]


def test_select_best_model_prefers_lowest_mae_then_rmse():
    results = [
        {"model": "ridge", "mae": 10.0, "rmse": 15.0, "r2": 0.2},
        {"model": "extra_trees", "mae": 8.0, "rmse": 14.0, "r2": 0.4},
        {"model": "knn", "mae": 8.0, "rmse": 12.0, "r2": 0.3},
    ]

    assert select_best_model(results)["model"] == "knn"


def test_descriptor_feature_columns_exclude_labels_and_metadata():
    columns = [
        "composition_normalized",
        "doi",
        "emission_peak_nm",
        "descriptor_status",
        "descriptor_n_elements",
        "matminer_MagpieData_mean_Number",
        "element_fraction_F",
        "frac_Mn",
        "mean_atomic_number",
    ]

    assert descriptor_feature_columns(columns) == [
        "descriptor_n_elements",
        "matminer_MagpieData_mean_Number",
        "element_fraction_F",
        "frac_Mn",
        "mean_atomic_number",
    ]


def test_load_descriptor_feature_rows_treats_blank_descriptor_values_as_zero(tmp_path):
    path = tmp_path / "descriptors.csv"
    rows = [
        {
            "material_id": "m1",
            "source_dataset": "synthetic_demo",
            "composition_normalized": "K2SiF6:Mn4+",
            "doi": "doi-a",
            "emission_peak_nm": "631",
            "descriptor_n_elements": "3",
            "frac_Mn": "",
        },
        {
            "material_id": "m2",
            "source_dataset": "synthetic_demo",
            "composition_normalized": "BaTiF6Mn0.02",
            "doi": "doi-b",
            "emission_peak_nm": "632",
            "descriptor_n_elements": "4",
            "frac_Mn": "0.002",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    loaded_rows, feature_names, features = load_descriptor_feature_rows(
        path,
        target_column="emission_peak_nm",
        group_column="doi",
    )

    assert len(loaded_rows) == 2
    assert feature_names == ["descriptor_n_elements", "frac_Mn"]
    assert features[0, feature_names.index("frac_Mn")] == 0.0


def test_run_model_suite_uses_group_kfold_when_groups_are_available():
    import numpy as np

    features = np.array([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]])
    targets = np.array([600.0, 601.0, 620.0, 621.0, 640.0, 641.0])
    groups = ["doi-a", "doi-a", "doi-b", "doi-b", "doi-c", "doi-c"]

    payload = run_model_suite(
        features,
        targets,
        groups=groups,
        model_names=["mean_baseline", "ridge", "elastic_net"],
        k_folds=3,
        random_seed=42,
    )

    model_names = {row["model"] for row in payload["models"]}
    assert {"mean_baseline", "ridge", "elastic_net"} == model_names
    for row in payload["models"]:
        assert row["model_name"] == row["model"]
        assert row["split_policy"] == "group_kfold"
    preprocess = {row["model"]: row["preprocess"] for row in payload["models"]}
    assert preprocess["mean_baseline"] == "none"
    assert preprocess["ridge"] == "standard_scaler"
    assert preprocess["elastic_net"] == "standard_scaler"


def test_format_model_suite_report_includes_level_and_best_model():
    payload = {
        "target_column": "emission_peak_nm",
        "dataset_branch": "synthetic_demo",
        "feature_level": "level1",
        "descriptor_backend": "element_fraction",
        "group_column": "doi",
        "n_rows": 12,
        "n_features": 6,
        "k_folds": 3,
        "source_counts": {"synthetic_demo": 12},
        "models": [{"model": "ridge", "mae": 10.0, "rmse": 15.0, "r2": 0.2}],
        "best_by_mae": {"model": "ridge", "mae": 10.0, "rmse": 15.0, "r2": 0.2},
        "unavailable_models": ["xgboost"],
    }

    report = format_model_suite_report(payload)

    assert "feature level: level1" in report
    assert "dataset branch: synthetic_demo" in report
    assert "group column: doi" in report
    assert "best model by MAE: ridge" in report
    assert "xgboost" in report
