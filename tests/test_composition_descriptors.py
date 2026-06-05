from __future__ import annotations

import numpy as np

from phosphor_ml.features.composition_descriptors import (
    build_basic_descriptor_table,
    build_level1_feature_table,
    build_level2_feature_table,
    formula_to_pymatgen_string,
    generate_composition_descriptor_row,
    parse_feature_formula_counts,
)


def test_level1_feature_table_uses_element_fraction_features_only():
    feature_names, features = build_level1_feature_table(["K2SiF6:Mn4+", "BaTi0.98F6Mn0.02"])

    assert feature_names == [
        "element_fraction_Ba",
        "element_fraction_F",
        "element_fraction_K",
        "element_fraction_Mn",
        "element_fraction_Si",
        "element_fraction_Ti",
    ]
    assert features.shape == (2, 6)
    assert np.isclose(features[0, feature_names.index("element_fraction_F")], 6 / 9)
    assert features[0, feature_names.index("element_fraction_Mn")] == 0.0
    assert "emission_peak_nm" not in feature_names
    assert "excitation_peak_nm" not in feature_names


def test_basic_descriptor_table_handles_dopant_notation_and_returns_finite_values():
    feature_names, features = build_basic_descriptor_table(["K2SiF6:Mn4+", "BaTi0.98F6Mn0.02"])

    assert "descriptor_n_elements" in feature_names
    assert "descriptor_fraction_mn" in feature_names
    assert features.shape == (2, len(feature_names))
    assert np.isfinite(features).all()
    assert features[0, feature_names.index("descriptor_n_elements")] == 3
    assert features[0, feature_names.index("descriptor_fraction_mn")] == 0.0


def test_level2_feature_table_combines_level1_and_basic_descriptors_without_label_leakage():
    feature_names, features = build_level2_feature_table(
        ["K2SiF6:Mn4+", "BaTi0.98F6Mn0.02"],
        descriptor_backend="basic",
    )

    assert any(name.startswith("element_fraction_") for name in feature_names)
    assert any(name.startswith("descriptor_") for name in feature_names)
    assert "emission_peak_nm" not in feature_names
    assert "excitation_peak_nm" not in feature_names
    assert features.shape[0] == 2


def test_formula_to_pymatgen_string_removes_charge_notation():
    assert formula_to_pymatgen_string("K2SiF6:Mn4+") == "F6K2Si"


def test_parse_feature_formula_counts_does_not_invent_dopant_amount():
    assert parse_feature_formula_counts("K2SiF6:Mn4+") == {"K": 2.0, "Si": 1.0, "F": 6.0}
    assert parse_feature_formula_counts("BaTi0.98F6Mn0.02")["Mn"] == 0.02


def test_generate_composition_descriptor_row_contains_named_physical_features():
    row = generate_composition_descriptor_row("K2SiF6Mn0.06")

    assert row["formula"] == "K2SiF6Mn0.06"
    assert row["descriptor_status"] == "parsed_full_formula"
    assert "num_elements" in row
    assert "mean_atomic_number" in row
    assert "frac_F" in row
    assert "frac_Mn" in row
    assert row["frac_F"] > 0
    assert row["frac_Mn"] > 0


def test_descriptor_generation_does_not_invent_mn_fraction_from_annotation():
    row = generate_composition_descriptor_row("K2SiF6:Mn4+")

    assert row["formula"] == "K2SiF6:Mn4+"
    assert row["descriptor_status"] == "partial_missing_dopant_amount"
    assert row["has_mn4_annotation"] is True
    assert row.get("frac_Mn", 0.0) == 0.0
