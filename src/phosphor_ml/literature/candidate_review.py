from __future__ import annotations

import csv
import html
import re
from pathlib import Path
from typing import Any

from phosphor_ml.utils.formula_utils import extract_element_symbols


REVIEW_SCHEMA_EXTENSION = [
    "relevance_score",
    "priority",
    "review_suggestion",
    "contains_manganese_text",
    "mentions_mn4_plus",
    "phosphor_relevance",
    "display_relevance",
    "optical_property_mentions",
    "rare_earth_free_claim",
    "rare_earth_elements_detected",
    "ranking_notes",
]

RARE_EARTH_TERMS = {
    "Sc": ["scandium"],
    "Y": ["yttrium", "yag"],
    "La": ["lanthanum"],
    "Ce": ["cerium"],
    "Pr": ["praseodymium"],
    "Nd": ["neodymium"],
    "Pm": ["promethium"],
    "Sm": ["samarium"],
    "Eu": ["europium"],
    "Gd": ["gadolinium"],
    "Tb": ["terbium"],
    "Dy": ["dysprosium"],
    "Ho": ["holmium"],
    "Er": ["erbium"],
    "Tm": ["thulium"],
    "Yb": ["ytterbium"],
    "Lu": ["lutetium"],
}

OPTICAL_TERMS = [
    "emission",
    "excitation",
    "plqy",
    "quantum yield",
    "lifetime",
    "fwhm",
    "thermal quenching",
    "thermal stability",
    "color purity",
    "photoluminescence",
]


def score_candidate_paper(record: dict[str, Any]) -> dict[str, str]:
    text = _combined_text(record)
    text_lower = text.lower()

    contains_manganese = _contains_manganese(text)
    mentions_mn4 = bool(
        re.search(r"\bmn\s*(?:4\s*\+|\(iv\)|iv)(?![a-z])|manganese\s*(?:4\s*\+|\(iv\)|iv)", text_lower)
    )
    phosphor_relevance = _has_any(text_lower, ["phosphor", "luminescen", "photoluminescen", "emitter"])
    display_relevance = _has_any(text_lower, ["display", "wled", "white led", "backlight", "warm led"])
    rare_earth_free_claim = bool(re.search(r"rare[- ]earth[- ]free|rare earth free", text_lower))
    rare_earth_elements = _detect_rare_earth_mentions(text)
    optical_mentions = [term for term in OPTICAL_TERMS if term in text_lower]

    score = 0
    notes: list[str] = []
    if contains_manganese:
        score += 4
        notes.append("mentions Mn/manganese")
    else:
        score -= 5
        notes.append("no clear Mn mention")

    if mentions_mn4:
        score += 3
        notes.append("mentions Mn4+")
    if phosphor_relevance:
        score += 3
        notes.append("phosphor/luminescence relevant")
    if display_relevance:
        score += 1
        notes.append("display or LED application")
    if rare_earth_free_claim:
        score += 3
        notes.append("claims rare-earth-free")
    if optical_mentions:
        score += min(4, len(optical_mentions))
        notes.append("mentions optical properties")
    if _has_any(text_lower, ["fluoride", "fluoro", "oxide", "oxyfluoride", "nitride"]):
        score += 1
        notes.append("host-family keyword")
    if rare_earth_elements and not rare_earth_free_claim:
        score -= 4
        notes.append("rare-earth element mention")

    priority = _priority(score, contains_manganese, phosphor_relevance, rare_earth_elements, rare_earth_free_claim)
    suggestion = _review_suggestion(priority, contains_manganese, rare_earth_elements, rare_earth_free_claim)

    return {
        "relevance_score": str(score),
        "priority": priority,
        "review_suggestion": suggestion,
        "contains_manganese_text": _yes_no(contains_manganese),
        "mentions_mn4_plus": _yes_no(mentions_mn4),
        "phosphor_relevance": _yes_no(phosphor_relevance),
        "display_relevance": _yes_no(display_relevance),
        "optical_property_mentions": "; ".join(optical_mentions),
        "rare_earth_free_claim": _yes_no(rare_earth_free_claim),
        "rare_earth_elements_detected": "; ".join(rare_earth_elements),
        "ranking_notes": "; ".join(notes),
    }


def rank_candidate_papers(input_csv: str | Path, output_csv: str | Path) -> list[dict[str, str]]:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(handle_fieldnames(reader_fieldnames=rows[0].keys() if rows else []))

    ranked_rows: list[dict[str, str]] = []
    for row in rows:
        scored = score_candidate_paper(row)
        ranked_rows.append({**row, **scored})

    ranked_rows.sort(key=_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + REVIEW_SCHEMA_EXTENSION)
        writer.writeheader()
        writer.writerows(ranked_rows)
    return ranked_rows


def handle_fieldnames(reader_fieldnames: Any) -> list[str]:
    return list(reader_fieldnames or [])


def _sort_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    year_text = str(row.get("year", ""))
    year = int(year_text) if year_text.isdigit() else 0
    return (
        priority_rank.get(row.get("priority", "low"), 2),
        -int(row.get("relevance_score", "0")),
        -year,
        str(row.get("title", "")).lower(),
    )


def _combined_text(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("title", "")),
        str(record.get("abstract", "")),
        str(record.get("keywords", "")),
    ]
    text = " ".join(parts)
    text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def _contains_manganese(text: str) -> bool:
    text_lower = text.lower()
    return bool(re.search(r"\bmn\b|mn\s*\d*\s*\+|manganese", text_lower))


def _detect_rare_earth_mentions(text: str) -> list[str]:
    text_lower = text.lower()
    found: set[str] = set()
    found.update(_rare_earth_symbols_from_formula_like_text(text))
    for symbol, names in RARE_EARTH_TERMS.items():
        if any(re.search(rf"\b{re.escape(name)}\b", text_lower) for name in names):
            found.add(symbol)
            continue
        if symbol == "Y":
            if re.search(r"\bY(?:AG|2O3|[0-9:.])", text):
                found.add(symbol)
            continue
        if re.search(rf"(?<![A-Za-z]){re.escape(symbol)}(?:[0-9:.]|\b)", text):
            found.add(symbol)
    return sorted(found)


def _rare_earth_symbols_from_formula_like_text(text: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z0-9:+.\-]+", " ", text)
    found: set[str] = set()
    for token in normalized.split():
        token_upper = token.upper()
        known_uppercase_rare_earth_material = token_upper in {"YAG", "Y2O3"}
        symbols = extract_element_symbols(token)
        if len(symbols) < 2:
            continue
        has_formula_marker = bool(re.search(r"[0-9:+]", token))
        has_condensed_formula_shape = bool(re.search(r"[A-Z][a-z]?[A-Z]", token))
        if token.isupper() and not has_formula_marker and not known_uppercase_rare_earth_material:
            continue
        if not has_formula_marker and not has_condensed_formula_shape:
            continue
        found.update(set(symbols) & set(RARE_EARTH_TERMS))
    return found


def _has_any(text_lower: str, terms: list[str]) -> bool:
    return any(term in text_lower for term in terms)


def _priority(
    score: int,
    contains_manganese: bool,
    phosphor_relevance: bool,
    rare_earth_elements: list[str],
    rare_earth_free_claim: bool,
) -> str:
    if contains_manganese and phosphor_relevance and score >= 10 and not rare_earth_elements:
        return "high"
    if contains_manganese and phosphor_relevance and score >= 5:
        return "medium"
    return "low"


def _review_suggestion(
    priority: str,
    contains_manganese: bool,
    rare_earth_elements: list[str],
    rare_earth_free_claim: bool,
) -> str:
    if priority == "high":
        return "review_first"
    if not contains_manganese or (rare_earth_elements and not rare_earth_free_claim):
        return "likely_exclude"
    return "manual_check"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
