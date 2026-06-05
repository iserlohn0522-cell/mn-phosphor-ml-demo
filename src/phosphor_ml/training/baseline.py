from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ELEMENT_SYMBOLS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
}

_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
_DOPANT_CHARGE_PATTERN = re.compile(r"([A-Z][a-z]?)([234])\+")


@dataclass(frozen=True)
class CrossValidationResult:
    predictions: np.ndarray
    fold_metrics: list[dict[str, float]]
    overall_metrics: dict[str, float]


@dataclass(frozen=True)
class TrainResult:
    target_column: str
    rows: list[dict[str, Any]]
    vocabulary: list[str]
    coefficients: np.ndarray
    intercept: float
    cv_result: CrossValidationResult
    summary: dict[str, Any]


def parse_formula_counts(formula: str) -> dict[str, float]:
    normalized = _normalize_formula(formula)
    counts: Counter[str] = Counter()
    for match in _ELEMENT_PATTERN.finditer(normalized):
        symbol = match.group(1)
        if symbol not in ELEMENT_SYMBOLS:
            continue
        amount_text = match.group(2)
        amount = float(amount_text) if amount_text else 1.0
        counts[symbol] += amount
    return dict(counts)


def build_element_vocabulary(formulas: list[str]) -> list[str]:
    elements: set[str] = set()
    for formula in formulas:
        elements.update(parse_formula_counts(formula))
    return sorted(elements)


def featurize_formulas(formulas: list[str], vocabulary: list[str]) -> np.ndarray:
    features = np.zeros((len(formulas), len(vocabulary)), dtype=float)
    index_by_element = {element: index for index, element in enumerate(vocabulary)}
    for row_index, formula in enumerate(formulas):
        counts = parse_formula_counts(formula)
        total = sum(counts.values())
        if total <= 0:
            continue
        for element, count in counts.items():
            if element in index_by_element:
                features[row_index, index_by_element[element]] = count / total
    return features


def load_labeled_rows(csv_path: Path, *, target_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            formula = source_row.get("composition_normalized") or source_row.get("composition_raw") or ""
            target = _parse_float(source_row.get(target_column, ""))
            if not formula or target is None:
                continue
            rows.append(
                {
                    "material_id": source_row.get("material_id", ""),
                    "source_dataset": source_row.get("source_dataset", source_row.get("data_source", "")),
                    "formula": formula,
                    "target": target,
                }
            )
    return rows


def describe_training_rows(rows: list[dict[str, Any]], *, target_column: str) -> dict[str, Any]:
    targets = [float(row["target"]) for row in rows]
    source_counts = Counter(str(row.get("source_dataset", "")) for row in rows)
    return {
        "target_column": target_column,
        "n_rows": len(rows),
        "target_min": min(targets) if targets else None,
        "target_max": max(targets) if targets else None,
        "target_mean": float(np.mean(targets)) if targets else None,
        "source_counts": dict(sorted(source_counts.items())),
    }


def cross_validate_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    k_folds: int = 5,
    alpha: float = 1.0,
    random_seed: int = 42,
) -> CrossValidationResult:
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if targets.shape[0] < 2:
        raise ValueError("at least two labeled rows are required")

    n_samples = targets.shape[0]
    folds = min(k_folds, n_samples)
    rng = np.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(n_samples)
    fold_indices = np.array_split(shuffled_indices, folds)
    predictions = np.full(n_samples, np.nan, dtype=float)
    fold_metrics: list[dict[str, float]] = []

    for fold_number, test_indices in enumerate(fold_indices, start=1):
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
        coefficients, intercept = fit_ridge(features[train_indices], targets[train_indices], alpha=alpha)
        fold_predictions = predict_ridge(features[test_indices], coefficients, intercept)
        predictions[test_indices] = fold_predictions
        metrics = regression_metrics(targets[test_indices], fold_predictions)
        metrics["fold"] = float(fold_number)
        metrics["n_test"] = float(len(test_indices))
        fold_metrics.append(metrics)

    overall_metrics = regression_metrics(targets, predictions)
    overall_metrics["n_samples"] = int(n_samples)
    overall_metrics["k_folds"] = int(folds)
    overall_metrics["alpha"] = float(alpha)
    return CrossValidationResult(predictions=predictions, fold_metrics=fold_metrics, overall_metrics=overall_metrics)


def fit_ridge(features: np.ndarray, targets: np.ndarray, *, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    target_mean = float(targets.mean())

    standardized_features = (features - feature_mean) / feature_scale
    centered_targets = targets - target_mean
    regularizer = alpha * np.eye(standardized_features.shape[1])
    coefficients_standardized = np.linalg.solve(
        standardized_features.T @ standardized_features + regularizer,
        standardized_features.T @ centered_targets,
    )
    coefficients = coefficients_standardized / feature_scale
    intercept = target_mean - float(feature_mean @ coefficients)
    return coefficients, intercept


def predict_ridge(features: np.ndarray, coefficients: np.ndarray, intercept: float) -> np.ndarray:
    return features @ coefficients + intercept


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residuals = predicted - actual
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(math.sqrt(np.mean(residuals**2)))
    baseline = actual - float(np.mean(actual))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum(baseline**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_baseline_from_csv(
    input_csv: Path,
    *,
    target_column: str,
    k_folds: int = 5,
    alpha: float = 1.0,
    random_seed: int = 42,
) -> TrainResult:
    rows = load_labeled_rows(input_csv, target_column=target_column)
    if len(rows) < 5:
        raise ValueError(f"Need at least 5 labeled rows for baseline training; found {len(rows)}.")

    formulas = [str(row["formula"]) for row in rows]
    targets = np.array([float(row["target"]) for row in rows], dtype=float)
    vocabulary = build_element_vocabulary(formulas)
    features = featurize_formulas(formulas, vocabulary)
    cv_result = cross_validate_ridge(features, targets, k_folds=k_folds, alpha=alpha, random_seed=random_seed)
    coefficients, intercept = fit_ridge(features, targets, alpha=alpha)
    summary = describe_training_rows(rows, target_column=target_column)
    summary.update(cv_result.overall_metrics)
    return TrainResult(
        target_column=target_column,
        rows=rows,
        vocabulary=vocabulary,
        coefficients=coefficients,
        intercept=intercept,
        cv_result=cv_result,
        summary=summary,
    )


def write_training_outputs(result: TrainResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "summary": result.summary,
        "fold_metrics": result.cv_result.fold_metrics,
        "model": {
            "model_type": "ridge_regression_element_fractions",
            "intercept": result.intercept,
            "alpha": result.cv_result.overall_metrics["alpha"],
            "vocabulary": result.vocabulary,
            "coefficients": [float(value) for value in result.coefficients],
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    predictions_path = output_dir / "cross_validated_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["material_id", "source_dataset", "formula", "target", "prediction", "residual"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction in zip(result.rows, result.cv_result.predictions):
            target = float(row["target"])
            writer.writerow(
                {
                    "material_id": row.get("material_id", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "formula": row.get("formula", ""),
                    "target": target,
                    "prediction": float(prediction),
                    "residual": float(prediction - target),
                }
            )

    report_path = output_dir / "training_report.md"
    report_path.write_text(_format_training_report(result), encoding="utf-8")


def _format_training_report(result: TrainResult) -> str:
    summary = result.summary
    lines = [
        "# Baseline Training Report",
        "",
        f"- target: {result.target_column}",
        f"- labeled rows: {summary['n_rows']}",
        f"- features: {len(result.vocabulary)} element-fraction columns",
        f"- k-folds: {summary['k_folds']}",
        f"- alpha: {summary['alpha']}",
        f"- MAE: {summary['mae']:.3f}",
        f"- RMSE: {summary['rmse']:.3f}",
        f"- R2: {summary['r2']:.3f}",
        "",
        "## Source Counts",
        "",
        *(f"- {source or 'unknown'}: {count}" for source, count in summary["source_counts"].items()),
        "",
        "## Notes",
        "",
        "- This is a composition-only ridge-regression baseline using parsed element fractions.",
        "- Use this as a pipeline smoke test and reference baseline, not as a final discovery model.",
        "- Demo data is synthetic and should not be interpreted as literature measurements.",
        "",
    ]
    return "\n".join(lines)


def _normalize_formula(formula: str) -> str:
    text = str(formula).strip()
    text = text.replace(" ", "")
    text = _DOPANT_CHARGE_PATTERN.sub(r"\1", text)
    return text


def _parse_float(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
