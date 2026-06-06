from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from phosphor_ml.features.composition_descriptors import build_level1_feature_table, build_level2_feature_table
from phosphor_ml.training.baseline import regression_metrics
from phosphor_ml.training.splits import make_cv_splits, split_policy


ModelFactory = Callable[[int], Any]

DEFAULT_MODEL_ORDER = [
    "mean_baseline",
    "ridge",
    "elastic_net",
    "knn",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "svr",
    "xgboost",
]


def load_supervised_rows(
    csv_path: Path,
    *,
    target_column: str,
    group_column: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            formula = source_row.get("composition_normalized") or source_row.get("composition_raw") or ""
            target = _parse_float(source_row.get(target_column, ""))
            if not formula or target is None:
                continue
            row = {
                "material_id": source_row.get("material_id", ""),
                "source_dataset": source_row.get("source_dataset", source_row.get("data_source", "")),
                "formula": formula,
                "target": target,
            }
            if group_column:
                row["group"] = source_row.get(group_column, "")
            rows.append(row)
    return rows


def run_model_suite_from_csv(
    input_csv: Path,
    *,
    target_column: str,
    feature_level: str,
    feature_set: str | None = None,
    descriptor_backend: str = "basic",
    allow_basic_fallback: bool = False,
    group_column: str | None = None,
    model_names: Iterable[str] | None = None,
    k_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    if feature_set == "descriptor_columns":
        rows, feature_names, features = load_descriptor_feature_rows(
            input_csv,
            target_column=target_column,
            group_column=group_column,
        )
        feature_level = "level2"
        descriptor_backend = "precomputed_descriptor_columns"
    else:
        rows = load_supervised_rows(input_csv, target_column=target_column, group_column=group_column)
        formulas = [str(row["formula"]) for row in rows]
        feature_names, features = build_feature_table(
            formulas,
            feature_level=feature_level,
            descriptor_backend=descriptor_backend,
            allow_basic_fallback=allow_basic_fallback,
        )

    if len(rows) < 5:
        raise ValueError(f"Need at least 5 labeled rows for model-suite training; found {len(rows)}.")
    targets = np.array([float(row["target"]) for row in rows], dtype=float)
    comparison = run_model_suite(
        features,
        targets,
        groups=[str(row.get("group", "")) for row in rows] if group_column else None,
        model_names=model_names,
        k_folds=k_folds,
        random_seed=random_seed,
    )
    source_counts = Counter(str(row.get("source_dataset", "")) for row in rows)
    comparison.update(
        {
            "input_csv": str(input_csv),
            "dataset_branch": dataset_branch_for_input(input_csv),
            "target_column": target_column,
            "feature_level": feature_level,
            "descriptor_backend": "element_fraction" if feature_level == "level1" else descriptor_backend,
            "group_column": group_column or "",
            "n_rows": len(rows),
            "n_features": int(features.shape[1]),
            "feature_names": feature_names,
            "source_counts": dict(sorted(source_counts.items())),
        }
    )
    return comparison


def load_descriptor_feature_rows(
    csv_path: Path,
    *,
    target_column: str,
    group_column: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    source_rows: list[dict[str, str]]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    feature_names = descriptor_feature_columns(source_rows[0].keys() if source_rows else [])
    rows: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    for source_row in source_rows:
        target = _parse_float(source_row.get(target_column, ""))
        formula = source_row.get("composition_normalized") or source_row.get("composition_raw") or source_row.get("formula") or ""
        if target is None or not formula:
            continue
        feature_values = [_parse_float(source_row.get(feature_name, "")) for feature_name in feature_names]
        row = {
            "material_id": source_row.get("material_id", ""),
            "source_dataset": source_row.get("source_dataset", source_row.get("data_source", "")),
            "formula": formula,
            "target": target,
        }
        if group_column:
            row["group"] = source_row.get(group_column, "")
        rows.append(row)
        feature_rows.append([float(value) if value is not None else 0.0 for value in feature_values])

    if not feature_names:
        raise ValueError("No descriptor feature columns were found.")
    return rows, feature_names, np.array(feature_rows, dtype=float)


def descriptor_feature_columns(columns: Iterable[str]) -> list[str]:
    exact_names = {"num_elements", "total_atoms", "mean_atomic_number", "atomic_number_range"}
    prefixes = ("element_fraction_", "descriptor_", "matminer_", "frac_")
    excluded = {"descriptor_status", "descriptor_provenance"}
    selected = []
    for column in columns:
        if column in excluded:
            continue
        if column in exact_names or any(column.startswith(prefix) for prefix in prefixes):
            selected.append(column)
    return selected


def build_feature_table(
    formulas: list[str],
    *,
    feature_level: str,
    descriptor_backend: str = "basic",
    allow_basic_fallback: bool = False,
) -> tuple[list[str], np.ndarray]:
    if feature_level == "level1":
        return build_level1_feature_table(formulas)
    if feature_level == "level2":
        return build_level2_feature_table(
            formulas,
            descriptor_backend=descriptor_backend,
            allow_basic_fallback=allow_basic_fallback,
        )
    raise ValueError(f"Unknown feature level: {feature_level}")


def dataset_branch_for_input(input_csv: Path) -> str:
    lowered = str(input_csv).lower()
    if "synthetic" in lowered or "demo" in lowered:
        return "synthetic_demo"
    return "custom_input"


def run_model_suite(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    groups: list[str] | None = None,
    model_names: Iterable[str] | None = None,
    k_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must contain the same number of rows")
    if targets.shape[0] < 5:
        raise ValueError("at least five rows are required for model comparison")

    factories, unavailable_models = make_model_factories(random_seed=random_seed)
    requested = list(model_names) if model_names else list(DEFAULT_MODEL_ORDER)
    results: list[dict[str, Any]] = []
    predictions_by_model: dict[str, list[float]] = {}
    unavailable = list(unavailable_models)

    for model_name in requested:
        factory = factories.get(model_name)
        if factory is None:
            if model_name not in unavailable:
                unavailable.append(model_name)
            continue
        predictions, fold_metrics = cross_validated_predictions(
            features,
            targets,
            groups=groups,
            model_factory=factory,
            k_folds=k_folds,
            random_seed=random_seed,
        )
        split_policy = fold_metrics[0]["split_policy"] if fold_metrics else "kfold"
        metrics = regression_metrics(targets, predictions)
        metrics.update(
            {
                "model": model_name,
                "model_name": model_name,
                "preprocess": model_preprocess_name(model_name),
                "split_policy": split_policy,
                "n_samples": int(targets.shape[0]),
                "k_folds": int(min(k_folds, targets.shape[0])),
                "fold_metrics": fold_metrics,
            }
        )
        results.append(metrics)
        predictions_by_model[model_name] = [float(value) for value in predictions]

    if not results:
        raise RuntimeError("No requested models were available. Install scikit-learn to run the model suite.")

    return {
        "models": results,
        "best_by_mae": select_best_model(results),
        "unavailable_models": sorted(set(unavailable)),
        "predictions_by_model": predictions_by_model,
        "k_folds": int(min(k_folds, targets.shape[0])),
    }


def make_model_factories(*, random_seed: int = 42) -> tuple[dict[str, ModelFactory], list[str]]:
    try:
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
        raise RuntimeError("scikit-learn is required to run the model suite.") from exc

    factories: dict[str, ModelFactory] = {
        "mean_baseline": lambda _n_train: DummyRegressor(strategy="mean"),
        "ridge": lambda _n_train: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "elastic_net": lambda _n_train: make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.25, random_state=random_seed, max_iter=10000),
        ),
        "knn": lambda n_train: make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=max(1, min(5, n_train))),
        ),
        "decision_tree": lambda _n_train: DecisionTreeRegressor(random_state=random_seed, min_samples_leaf=2),
        "random_forest": lambda _n_train: RandomForestRegressor(
            n_estimators=200,
            random_state=random_seed,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "extra_trees": lambda _n_train: ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_seed,
            min_samples_leaf=1,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda _n_train: GradientBoostingRegressor(random_state=random_seed),
        "svr": lambda _n_train: make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1)),
    }
    unavailable: list[str] = []

    try:
        from xgboost import XGBRegressor

        factories["xgboost"] = lambda _n_train: XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=-1,
        )
    except Exception:
        unavailable.append("xgboost")

    return factories, unavailable


def cross_validated_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    groups: list[str] | None = None,
    model_factory: ModelFactory,
    k_folds: int,
    random_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = min(k_folds, len(targets))
    policy = split_policy(n_samples=len(targets), k_folds=folds, groups=groups)
    split_iter = make_cv_splits(n_samples=len(targets), k_folds=folds, random_seed=random_seed, groups=groups)
    predictions = np.full(len(targets), np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []

    for fold_number, (train_index_list, test_index_list) in enumerate(split_iter, start=1):
        train_indices = np.array(train_index_list, dtype=int)
        test_indices = np.array(test_index_list, dtype=int)
        model = model_factory(len(train_indices))
        model.fit(features[train_indices], targets[train_indices])
        fold_predictions = np.asarray(model.predict(features[test_indices]), dtype=float)
        predictions[test_indices] = fold_predictions
        metrics = regression_metrics(targets[test_indices], fold_predictions)
        metrics["fold"] = float(fold_number)
        metrics["n_test"] = float(len(test_indices))
        metrics["split_policy"] = policy
        fold_metrics.append(metrics)

    return predictions, fold_metrics


def select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No model results were provided")
    return min(results, key=lambda row: (float(row["mae"]), float(row["rmse"]), -float(row["r2"])))


def model_preprocess_name(model_name: str) -> str:
    if model_name in {"ridge", "elastic_net", "knn", "svr"}:
        return "standard_scaler"
    return "none"


def write_model_suite_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(_json_safe_payload(payload), indent=2), encoding="utf-8")
    _write_metrics_csv(payload, output_dir / "model_metrics.csv")
    _write_metrics_csv(payload, output_dir / "model_comparison.csv")
    report = format_model_suite_report(payload)
    (output_dir / "model_suite_report.md").write_text(report, encoding="utf-8")
    (output_dir / "training_report.md").write_text(report, encoding="utf-8")


def format_model_suite_report(payload: dict[str, Any]) -> str:
    best = payload["best_by_mae"]
    lines = [
        "# Model Suite Report",
        "",
        f"- dataset branch: {payload.get('dataset_branch', 'verified_labels')}",
        f"- target: {payload['target_column']}",
        f"- feature level: {payload['feature_level']}",
        f"- descriptor backend: {payload['descriptor_backend']}",
        f"- group column: {payload.get('group_column') or 'none'}",
        f"- labeled rows: {payload['n_rows']}",
        f"- features: {payload['n_features']}",
        f"- k-folds: {payload['k_folds']}",
        f"- best model by MAE: {best['model']} (MAE={best['mae']:.3f}, RMSE={best['rmse']:.3f}, R2={best['r2']:.3f})",
        "",
        "## Model Metrics",
        "",
        "| Model | MAE | RMSE | R2 | Split | Preprocess |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for result in sorted(payload["models"], key=lambda row: (float(row["mae"]), float(row["rmse"]))):
        lines.append(
            f"| {result['model']} | {result['mae']:.3f} | {result['rmse']:.3f} | {result['r2']:.3f} | "
            f"{result.get('split_policy', 'kfold')} | {result.get('preprocess', 'none')} |"
        )
    lines.extend(["", "## Source Counts", ""])
    for source, count in payload["source_counts"].items():
        lines.append(f"- {source or 'unknown'}: {count}")
    if payload.get("unavailable_models"):
        lines.extend(["", "## Unavailable Models", ""])
        for model_name in payload["unavailable_models"]:
            lines.append(f"- {model_name}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Demo labels are synthetic.",
            "- These metrics check the workflow. They do not support phosphor performance claims.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _write_metrics_csv(payload: dict[str, Any], output_path: Path) -> None:
    fieldnames = ["model", "model_name", "mae", "rmse", "r2", "split_policy", "preprocess", "n_samples", "k_folds"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["models"])


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    safe["models"] = [_json_safe_row(row) for row in payload["models"]]
    safe["best_by_mae"] = _json_safe_row(payload["best_by_mae"])
    return safe


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            safe[key] = value.item()
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def _parse_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
