import csv
import json

from phosphor_ml.literature.search_pipeline import run_literature_search
from phosphor_ml.schemas import PAPER_SCHEMA


def test_dry_run_collection_writes_raw_json_and_candidate_csv(tmp_path):
    records = run_literature_search(
        queries=["Mn4+ doped phosphor rare earth free"],
        max_results=2,
        source="all",
        data_dir=tmp_path,
        dry_run=True,
    )

    raw_files = sorted((tmp_path / "raw" / "literature").glob("*.json"))
    csv_path = tmp_path / "interim" / "candidate_papers.csv"

    assert len(raw_files) == 2
    assert csv_path.exists()
    assert len(records) == 2

    with raw_files[0].open(encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    assert "query" in raw_payload

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert list(rows[0].keys()) == PAPER_SCHEMA
    assert rows[0]["paper_id"] == "P000001"
