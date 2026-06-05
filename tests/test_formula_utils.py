from phosphor_ml.utils.formula_utils import (
    contains_manganese,
    detect_rare_earth_elements,
    extract_element_symbols,
    is_rare_earth_free_formula,
    normalize_formula_text,
)


def test_extract_element_symbols_from_phosphor_formula_text():
    symbols = extract_element_symbols("K2SiF6:Mn4+")

    assert symbols == ["F", "K", "Mn", "Si"]


def test_detect_rare_earth_elements_includes_y_and_lanthanides():
    assert detect_rare_earth_elements("Y2O3:Eu3+, Mn4+") == ["Eu", "Y"]
    assert detect_rare_earth_elements("K2SiF6:Mn4+") == []


def test_formula_helpers_classify_manganese_and_rare_earth_free_text():
    assert contains_manganese("Na2TiF6:Mn4+")
    assert is_rare_earth_free_formula("Na2TiF6:Mn4+")
    assert not is_rare_earth_free_formula("YAG:Ce3+")
    assert normalize_formula_text("  K2 Si F6 : Mn4+  ") == "K2SiF6:Mn4+"
