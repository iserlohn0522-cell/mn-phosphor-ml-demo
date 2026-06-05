from phosphor_ml.schemas import (
    MATERIAL_SCHEMA,
    PAPER_SCHEMA,
    empty_material_record,
    empty_paper_record,
    validate_columns,
)


def test_paper_schema_contains_required_metadata_columns():
    required = {
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
    }

    assert required.issubset(PAPER_SCHEMA)
    assert set(empty_paper_record()) == set(PAPER_SCHEMA)
    assert validate_columns(PAPER_SCHEMA, PAPER_SCHEMA) == []


def test_material_schema_contains_required_phosphor_columns():
    required = {
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
    }

    assert required.issubset(MATERIAL_SCHEMA)
    assert set(empty_material_record()) == set(MATERIAL_SCHEMA)
