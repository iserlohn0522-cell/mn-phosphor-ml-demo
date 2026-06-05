from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from phosphor_ml.training.baseline import ELEMENT_SYMBOLS, parse_formula_counts


ATOMIC_NUMBERS = {
    "H": 1,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Mn": 25,
    "Co": 27,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Eu": 63,
    "Gd": 64,
    "Lu": 71,
    "Ta": 73,
    "W": 74,
    "Pb": 82,
    "Bi": 83,
}

ATOMIC_WEIGHTS = {
    "H": 1.008,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.99,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Mn": 54.938,
    "Co": 58.933,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "Cs": 132.905,
    "Ba": 137.327,
    "La": 138.905,
    "Eu": 151.964,
    "Gd": 157.25,
    "Lu": 174.967,
    "Ta": 180.948,
    "W": 183.84,
    "Pb": 207.2,
    "Bi": 208.98,
}

ELECTRONEGATIVITY = {
    "H": 2.20,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "Cl": 3.16,
    "K": 0.82,
    "Ca": 1.00,
    "Sc": 1.36,
    "Ti": 1.54,
    "V": 1.63,
    "Mn": 1.55,
    "Co": 1.88,
    "Zn": 1.65,
    "Ga": 1.81,
    "Ge": 2.01,
    "Rb": 0.82,
    "Sr": 0.95,
    "Y": 1.22,
    "Zr": 1.33,
    "Nb": 1.60,
    "Mo": 2.16,
    "In": 1.78,
    "Sn": 1.96,
    "Sb": 2.05,
    "Te": 2.10,
    "Cs": 0.79,
    "Ba": 0.89,
    "La": 1.10,
    "Eu": 1.20,
    "Gd": 1.20,
    "Lu": 1.27,
    "Ta": 1.50,
    "W": 2.36,
    "Pb": 2.33,
    "Bi": 2.02,
}

BASIC_DESCRIPTOR_NAMES = [
    "descriptor_n_elements",
    "descriptor_total_atoms",
    "descriptor_max_fraction",
    "descriptor_min_nonzero_fraction",
    "descriptor_fraction_mn",
    "descriptor_fraction_o",
    "descriptor_fraction_f",
    "descriptor_anion_fraction",
    "descriptor_mean_atomic_number",
    "descriptor_atomic_number_range",
    "descriptor_mean_atomic_weight",
    "descriptor_mean_electronegativity",
    "descriptor_electronegativity_range",
]

_MN4_ANNOTATION_PATTERN = re.compile(r"(?::\s*)?Mn\s*(?:4\+|\(IV\)|IV)", re.IGNORECASE)


def build_level1_feature_table(formulas: Sequence[str]) -> tuple[list[str], np.ndarray]:
    vocabulary = _build_feature_safe_element_vocabulary(formulas)
    features = _featurize_formulas_safely(formulas, vocabulary)
    names = [f"element_fraction_{element}" for element in vocabulary]
    return names, features


def build_level2_feature_table(
    formulas: Sequence[str],
    *,
    descriptor_backend: str = "basic",
    allow_basic_fallback: bool = False,
) -> tuple[list[str], np.ndarray]:
    level1_names, level1_features = build_level1_feature_table(formulas)
    if descriptor_backend == "basic":
        descriptor_names, descriptor_features = build_basic_descriptor_table(formulas)
    elif descriptor_backend == "matminer":
        try:
            descriptor_names, descriptor_features = build_matminer_descriptor_table(formulas)
        except RuntimeError:
            if not allow_basic_fallback:
                raise
            descriptor_names, descriptor_features = build_basic_descriptor_table(formulas)
    else:
        raise ValueError(f"Unknown descriptor backend: {descriptor_backend}")
    return level1_names + descriptor_names, np.hstack([level1_features, descriptor_features])


def build_basic_descriptor_table(formulas: Sequence[str]) -> tuple[list[str], np.ndarray]:
    rows = [_basic_descriptors(formula) for formula in formulas]
    return list(BASIC_DESCRIPTOR_NAMES), np.array(rows, dtype=float)


def generate_composition_descriptor_row(formula: str, *, descriptor_backend: str = "basic") -> dict[str, object]:
    normalized_formula, descriptor_status, has_mn4_annotation = normalize_formula_for_composition(formula)
    counts = parse_formula_counts(normalized_formula) if normalized_formula else {}
    counts = {element: amount for element, amount in counts.items() if element in ELEMENT_SYMBOLS}
    total = sum(counts.values())
    fractions = {element: amount / total for element, amount in counts.items()} if total > 0 else {}
    provenance = _descriptor_provenance(descriptor_backend)

    row: dict[str, object] = {
        "formula": formula,
        "normalized_formula_for_descriptors": normalized_formula,
        "descriptor_status": descriptor_status,
        "has_mn4_annotation": has_mn4_annotation,
        "descriptor_provenance": provenance,
        "num_elements": float(len(counts)),
        "total_atoms": float(total),
        "mean_atomic_number": _weighted_mean(fractions, ATOMIC_NUMBERS),
        "atomic_number_range": _weighted_range(fractions, ATOMIC_NUMBERS),
    }
    for element in sorted(ELEMENT_SYMBOLS):
        fraction = fractions.get(element, 0.0)
        if fraction:
            row[f"frac_{element}"] = fraction
    row.setdefault("frac_Mn", 0.0)
    row.setdefault("frac_F", 0.0)
    row.setdefault("frac_O", 0.0)
    return row


def normalize_formula_for_composition(formula: str) -> tuple[str, str, bool]:
    text = str(formula).strip()
    if not text:
        return "", "unparseable_formula", False

    has_mn4_annotation = bool(_MN4_ANNOTATION_PATTERN.search(text))
    if ":" in text:
        host_formula = text.split(":", 1)[0].strip()
        if parse_formula_counts(host_formula):
            return host_formula, "partial_missing_dopant_amount", has_mn4_annotation
        return "", "unparseable_formula", has_mn4_annotation

    if parse_formula_counts(text):
        return text, "parsed_full_formula", has_mn4_annotation
    return "", "unparseable_formula", has_mn4_annotation


def build_matminer_descriptor_table(formulas: Sequence[str]) -> tuple[list[str], np.ndarray]:
    try:
        from matminer.featurizers.composition import ElementFraction, ElementProperty, Stoichiometry, ValenceOrbital
        from pymatgen.core import Composition
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
        raise RuntimeError("Matminer/Pymatgen descriptors require the optional descriptor dependencies.") from exc

    featurizers = [ElementFraction(), Stoichiometry(), ElementProperty.from_preset("magpie"), ValenceOrbital()]
    labels: list[str] = []
    for featurizer in featurizers:
        labels.extend(f"matminer_{label}" for label in featurizer.feature_labels())

    rows: list[list[float]] = []
    for formula in formulas:
        composition = Composition(formula_to_pymatgen_string(formula))
        values: list[float] = []
        for featurizer in featurizers:
            values.extend(float(value) if _is_number(value) else 0.0 for value in featurizer.featurize(composition))
        rows.append(values)

    features = np.array(rows, dtype=float)
    features[~np.isfinite(features)] = 0.0
    return labels, features


def formula_to_pymatgen_string(formula: str) -> str:
    counts = parse_feature_formula_counts(formula)
    parts = []
    for element in sorted(counts):
        amount = counts[element]
        amount_text = "" if math.isclose(amount, 1.0) else _format_amount(amount)
        parts.append(f"{element}{amount_text}")
    return "".join(parts)


def _basic_descriptors(formula: str) -> list[float]:
    counts = parse_feature_formula_counts(formula)
    total = sum(counts.values())
    if total <= 0:
        return [0.0] * len(BASIC_DESCRIPTOR_NAMES)

    fractions = {element: count / total for element, count in counts.items()}
    nonzero_fractions = list(fractions.values())
    anions = {"O", "F", "N", "Cl", "S", "P"}
    return [
        float(len(counts)),
        float(total),
        max(nonzero_fractions),
        min(nonzero_fractions),
        fractions.get("Mn", 0.0),
        fractions.get("O", 0.0),
        fractions.get("F", 0.0),
        sum(fractions.get(element, 0.0) for element in anions),
        _weighted_mean(fractions, ATOMIC_NUMBERS),
        _weighted_range(fractions, ATOMIC_NUMBERS),
        _weighted_mean(fractions, ATOMIC_WEIGHTS),
        _weighted_mean(fractions, ELECTRONEGATIVITY),
        _weighted_range(fractions, ELECTRONEGATIVITY),
    ]


def _weighted_mean(fractions: dict[str, float], values: dict[str, float]) -> float:
    available = [(fraction, values[element]) for element, fraction in fractions.items() if element in values]
    total_fraction = sum(fraction for fraction, _value in available)
    if total_fraction <= 0:
        return 0.0
    return sum(fraction * value for fraction, value in available) / total_fraction


def _weighted_range(fractions: dict[str, float], values: dict[str, float]) -> float:
    available = [values[element] for element in fractions if element in values]
    if not available:
        return 0.0
    return max(available) - min(available)


def _format_amount(amount: float) -> str:
    text = f"{amount:.8f}".rstrip("0").rstrip(".")
    return text


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def parse_feature_formula_counts(formula: str) -> dict[str, float]:
    feature_formula, _descriptor_status, _has_mn4_annotation = normalize_formula_for_composition(formula)
    raw_counts = parse_formula_counts(feature_formula)
    return {element: amount for element, amount in raw_counts.items() if element in ELEMENT_SYMBOLS}


def _descriptor_provenance(descriptor_backend: str) -> str:
    if descriptor_backend == "matminer":
        try:
            import matminer  # noqa: F401
            import pymatgen  # noqa: F401

            return "matminer_enabled"
        except Exception:
            return "basic_only"
    if descriptor_backend == "pymatgen":
        try:
            import pymatgen  # noqa: F401

            return "pymatgen_enabled"
        except Exception:
            return "basic_only"
    return "basic_only"


def _build_feature_safe_element_vocabulary(formulas: Sequence[str]) -> list[str]:
    elements: set[str] = set()
    for formula in formulas:
        elements.update(parse_feature_formula_counts(formula))
    return sorted(elements)


def _featurize_formulas_safely(formulas: Sequence[str], vocabulary: list[str]) -> np.ndarray:
    features = np.zeros((len(formulas), len(vocabulary)), dtype=float)
    index_by_element = {element: index for index, element in enumerate(vocabulary)}
    for row_index, formula in enumerate(formulas):
        counts: Counter[str] = Counter(parse_feature_formula_counts(formula))
        total = sum(counts.values())
        if total <= 0:
            continue
        for element, count in counts.items():
            if element in index_by_element:
                features[row_index, index_by_element[element]] = count / total
    return features
