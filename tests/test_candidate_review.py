import csv

from phosphor_ml.literature.candidate_review import (
    REVIEW_SCHEMA_EXTENSION,
    rank_candidate_papers,
    score_candidate_paper,
)


def test_score_candidate_paper_prioritizes_mn4_rare_earth_free_phosphor():
    record = {
        "paper_id": "P1",
        "title": "Surface-modified KNaTaF7:Mn4+ fluoride phosphor for warm WLED applications",
        "abstract": "The rare-earth-free red phosphor shows improved thermal stability and emission.",
        "keywords": "Phosphor; Luminescence; Manganese",
        "search_query": "Mn4+ fluoride phosphor",
    }

    scored = score_candidate_paper(record)

    assert scored["contains_manganese_text"] == "yes"
    assert scored["mentions_mn4_plus"] == "yes"
    assert scored["rare_earth_free_claim"] == "yes"
    assert scored["priority"] == "high"
    assert int(scored["relevance_score"]) >= 10


def test_score_candidate_paper_penalizes_rare_earth_only_luminescence():
    record = {
        "paper_id": "P2",
        "title": "Tb2(DPA)2 green emitter coordination compound luminescence",
        "abstract": "A terbium complex with temperature-dependent luminescence.",
        "keywords": "",
        "search_query": "Mn4+ doped phosphor rare earth free",
    }

    scored = score_candidate_paper(record)

    assert scored["contains_manganese_text"] == "no"
    assert scored["rare_earth_elements_detected"] == "Tb"
    assert scored["priority"] == "low"
    assert scored["review_suggestion"] == "likely_exclude"


def test_score_candidate_paper_detects_rare_earth_symbols_inside_formula_text():
    record = {
        "paper_id": "P3",
        "title": "Enhanced deep-red emission from Mn4+/Mg2+ co-doped CaGdAlO4 phosphors",
        "abstract": "The formula contains gadolinium and should be checked as rare-earth containing.",
        "keywords": "phosphor; emission",
        "search_query": "Mn4+ oxide phosphor",
    }

    scored = score_candidate_paper(record)

    assert "Gd" in scored["rare_earth_elements_detected"]
    assert scored["priority"] != "high"
    assert scored["review_suggestion"] == "likely_exclude"


def test_rank_candidate_papers_sorts_high_priority_first_and_writes_review_csv(tmp_path):
    input_path = tmp_path / "candidate_papers.csv"
    output_path = tmp_path / "candidate_papers_review.csv"
    rows = [
        {
            "paper_id": "P1",
            "doi": "10.1/noise",
            "title": "Mineral commodity summaries 2025",
            "year": "2025",
            "authors": "",
            "journal": "",
            "abstract": "",
            "source_api": "openalex",
            "openalex_id": "",
            "semantic_scholar_id": "",
            "url": "",
            "keywords": "",
            "search_query": "Mn4+ oxide phosphor",
            "relevance_notes": "",
            "collected_at": "",
        },
        {
            "paper_id": "P2",
            "doi": "10.1/mn",
            "title": "K2SiF6:Mn4+ rare-earth-free red phosphor with high PLQY",
            "year": "2024",
            "authors": "",
            "journal": "",
            "abstract": "Emission peak and lifetime are reported for display backlights.",
            "source_api": "openalex",
            "openalex_id": "",
            "semantic_scholar_id": "",
            "url": "",
            "keywords": "phosphor; manganese",
            "search_query": "Mn4+ fluoride phosphor",
            "relevance_notes": "",
            "collected_at": "",
        },
    ]
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = rank_candidate_papers(input_path, output_path)

    assert ranked[0]["paper_id"] == "P2"
    assert ranked[0]["priority"] == "high"
    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as handle:
        output_rows = list(csv.DictReader(handle))

    for column in REVIEW_SCHEMA_EXTENSION:
        assert column in output_rows[0]
