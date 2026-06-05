from __future__ import annotations

from collections.abc import Iterable


PAPER_SCHEMA = [
    "paper_id",
    "doi",
    "title",
    "year",
    "authors",
    "journal",
    "abstract",
    "source_api",
    "openalex_id",
    "semantic_scholar_id",
    "url",
    "keywords",
    "search_query",
    "relevance_notes",
    "collected_at",
]


MATERIAL_SCHEMA = [
    "material_id",
    "paper_id",
    "doi",
    "composition_raw",
    "composition_normalized",
    "host_formula",
    "dopant",
    "dopant_valence",
    "activator",
    "contains_mn",
    "mn_valence",
    "is_rare_earth_free",
    "rare_earth_elements_detected",
    "emission_peak_nm",
    "excitation_peak_nm",
    "fwhm_nm",
    "plqy",
    "lifetime",
    "stability_metric",
    "synthesis_method",
    "structure_info",
    "space_group",
    "notes",
    "data_source",
    "extraction_status",
]


def empty_paper_record(**overrides: object) -> dict[str, object]:
    return _empty_record(PAPER_SCHEMA, overrides)


def empty_material_record(**overrides: object) -> dict[str, object]:
    return _empty_record(MATERIAL_SCHEMA, overrides)


def validate_columns(actual_columns: Iterable[str], expected_columns: Iterable[str]) -> list[str]:
    actual = list(actual_columns)
    expected = list(expected_columns)
    messages: list[str] = []

    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]

    if missing:
        messages.append(f"Missing columns: {', '.join(missing)}")
    if unexpected:
        messages.append(f"Unexpected columns: {', '.join(unexpected)}")
    if not missing and not unexpected and actual != expected:
        messages.append("Columns match the schema names but are in a different order.")

    return messages


def _empty_record(schema: list[str], overrides: dict[str, object]) -> dict[str, object]:
    record: dict[str, object] = {column: "" for column in schema}
    for key, value in overrides.items():
        if key not in record:
            raise KeyError(f"Unknown schema field: {key}")
        record[key] = value
    return record
