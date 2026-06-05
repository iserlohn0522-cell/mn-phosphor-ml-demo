from __future__ import annotations

import re


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
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}

RARE_EARTH_ELEMENTS = {
    "Sc",
    "Y",
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
}

_ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_formula_text(formula_text: str | None) -> str:
    if not formula_text:
        return ""
    return _WHITESPACE_PATTERN.sub("", formula_text.strip())


def extract_element_symbols(formula_text: str | None) -> list[str]:
    normalized = normalize_formula_text(formula_text)
    symbols = {
        match.group(0)
        for match in _ELEMENT_PATTERN.finditer(normalized)
        if match.group(0) in ELEMENT_SYMBOLS
    }
    return sorted(symbols)


def detect_rare_earth_elements(formula_text: str | None) -> list[str]:
    return sorted(set(extract_element_symbols(formula_text)) & RARE_EARTH_ELEMENTS)


def contains_manganese(formula_text: str | None) -> bool:
    return "Mn" in extract_element_symbols(formula_text)


def is_rare_earth_free_formula(formula_text: str | None) -> bool:
    return not detect_rare_earth_elements(formula_text)
