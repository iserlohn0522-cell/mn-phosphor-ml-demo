from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phosphor_ml.config import Config
from phosphor_ml.literature.crossref_client import CrossrefClient
from phosphor_ml.literature.openalex_client import OpenAlexClient
from phosphor_ml.literature.semantic_scholar_client import SemanticScholarClient
from phosphor_ml.schemas import PAPER_SCHEMA, empty_paper_record
from phosphor_ml.utils.io import ensure_dir, slugify, write_csv, write_json


LOGGER = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "Mn4+ doped phosphor rare earth free",
    "manganese doped phosphor luminescence",
    "Mn-based phosphor display application",
    "rare-earth-free phosphor manganese",
    "Mn4+ fluoride phosphor",
    "Mn4+ oxide phosphor",
    "phosphor emission wavelength PLQY lifetime Mn",
]

SUPPORTED_SOURCES = {"openalex", "semantic_scholar", "crossref", "all"}


def run_literature_search(
    queries: list[str] | None = None,
    max_results: int = 25,
    source: str = "all",
    data_dir: str | Path | None = None,
    config: Config | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source '{source}'. Choose from {sorted(SUPPORTED_SOURCES)}.")

    active_queries = queries or DEFAULT_QUERIES
    active_config = config or Config.from_env()
    output_root = Path(data_dir) if data_dir else active_config.data_dir
    raw_dir = ensure_dir(output_root / "raw" / "literature")
    interim_dir = ensure_dir(output_root / "interim")
    collected_at = datetime.now(timezone.utc).isoformat()

    records: list[dict[str, Any]] = []
    for query in active_queries:
        for source_name in _expand_sources(source):
            try:
                raw_response = _search_source(
                    source_name=source_name,
                    query=query,
                    max_results=max_results,
                    config=active_config,
                    dry_run=dry_run,
                )
            except RuntimeError as exc:
                LOGGER.warning("Skipping %s query %r: %s", source_name, query, exc)
                continue

            payload = {
                "source_api": source_name,
                "query": query,
                "max_results": max_results,
                "dry_run": dry_run,
                "collected_at": collected_at,
                "response": raw_response,
            }
            raw_path = raw_dir / f"{source_name}_{slugify(query)}_{_timestamp_slug(collected_at)}.json"
            write_json(raw_path, payload)

            records.extend(
                _normalize_response(
                    source_name=source_name,
                    query=query,
                    raw_response=raw_response,
                    collected_at=collected_at,
                )
            )

    deduped = _deduplicate_records(records)
    write_csv(interim_dir / "candidate_papers.csv", deduped, PAPER_SCHEMA)
    return deduped


def _expand_sources(source: str) -> list[str]:
    if source == "all":
        return ["openalex", "semantic_scholar"]
    return [source]


def _search_source(
    source_name: str,
    query: str,
    max_results: int,
    config: Config,
    dry_run: bool,
) -> dict:
    if dry_run:
        return _dry_run_response(source_name, query, max_results)

    if source_name == "openalex":
        return OpenAlexClient(
            email=config.openalex_email,
            api_key=config.openalex_api_key,
        ).search(query, max_results=max_results)
    if source_name == "semantic_scholar":
        return SemanticScholarClient(api_key=config.semantic_scholar_api_key).search(
            query,
            max_results=max_results,
        )
    if source_name == "crossref":
        return CrossrefClient(mailto=config.crossref_mailto).search(query, max_results=max_results)
    raise ValueError(f"Unsupported source: {source_name}")


def _dry_run_response(source_name: str, query: str, max_results: int) -> dict:
    if max_results <= 0:
        return {"results": [], "data": [], "message": {"items": []}}

    if source_name == "openalex":
        return {
            "results": [
                {
                    "id": "https://openalex.org/W000000001",
                    "doi": "https://doi.org/10.1000/mn-openalex-demo",
                    "title": "Rare-earth-free Mn4+ fluoride phosphors for display backlights",
                    "publication_year": 2024,
                    "authorships": [
                        {"author": {"display_name": "A. Researcher"}},
                        {"author": {"display_name": "B. Phosphor"}},
                    ],
                    "primary_location": {
                        "landing_page_url": "https://doi.org/10.1000/mn-openalex-demo",
                        "source": {"display_name": "Journal of Luminescent Materials"},
                    },
                    "concepts": [
                        {"display_name": "Phosphor"},
                        {"display_name": "Luminescence"},
                        {"display_name": "Manganese"},
                    ],
                    "abstract_inverted_index": {
                        "Mn4+": [0],
                        "phosphors": [1],
                        "show": [2],
                        "red": [3],
                        "emission.": [4],
                    },
                }
            ]
        }

    if source_name == "semantic_scholar":
        return {
            "data": [
                {
                    "paperId": "S000000001",
                    "externalIds": {"DOI": "10.1000/mn-semantic-demo"},
                    "title": "Manganese activated oxide phosphors with high color purity",
                    "year": 2023,
                    "authors": [{"name": "C. Data"}, {"name": "D. Materials"}],
                    "venue": "Advanced Phosphor Research",
                    "abstract": "A dry-run fixture for Mn phosphor literature triage.",
                    "url": "https://doi.org/10.1000/mn-semantic-demo",
                    "publicationTypes": ["JournalArticle"],
                }
            ]
        }

    if source_name == "crossref":
        return {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/mn-crossref-demo",
                        "title": ["Mn4+ phosphor literature collection fixture"],
                        "published-print": {"date-parts": [[2022]]},
                        "author": [{"given": "E.", "family": "Crossref"}],
                        "container-title": ["Crossref Fixture Journal"],
                        "URL": "https://doi.org/10.1000/mn-crossref-demo",
                    }
                ]
            }
        }

    raise ValueError(f"Unsupported source: {source_name}")


def _normalize_response(
    source_name: str,
    query: str,
    raw_response: dict,
    collected_at: str,
) -> list[dict[str, Any]]:
    if source_name == "openalex":
        return [_normalize_openalex(item, query, collected_at) for item in raw_response.get("results", [])]
    if source_name == "semantic_scholar":
        return [
            _normalize_semantic_scholar(item, query, collected_at)
            for item in raw_response.get("data", [])
        ]
    if source_name == "crossref":
        return [
            _normalize_crossref(item, query, collected_at)
            for item in raw_response.get("message", {}).get("items", [])
        ]
    return []


def _normalize_openalex(item: dict, query: str, collected_at: str) -> dict[str, Any]:
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    authors = [
        authorship.get("author", {}).get("display_name", "")
        for authorship in item.get("authorships", [])
    ]
    keywords = [
        concept.get("display_name", "")
        for concept in item.get("concepts", [])
        if concept.get("display_name")
    ]

    return empty_paper_record(
        doi=normalize_doi(item.get("doi", "")),
        title=item.get("title", "") or "",
        year=item.get("publication_year", "") or "",
        authors="; ".join(author for author in authors if author),
        journal=source.get("display_name", "") or "",
        abstract=_openalex_abstract(item),
        source_api="openalex",
        openalex_id=item.get("id", "") or "",
        url=primary_location.get("landing_page_url") or item.get("id", "") or "",
        keywords="; ".join(keywords),
        search_query=query,
        relevance_notes="",
        collected_at=collected_at,
    )


def _normalize_semantic_scholar(item: dict, query: str, collected_at: str) -> dict[str, Any]:
    external_ids = item.get("externalIds") or {}
    authors = [author.get("name", "") for author in item.get("authors", [])]
    publication_types = item.get("publicationTypes") or []

    return empty_paper_record(
        doi=normalize_doi(external_ids.get("DOI", "")),
        title=item.get("title", "") or "",
        year=item.get("year", "") or "",
        authors="; ".join(author for author in authors if author),
        journal=item.get("venue", "") or "",
        abstract=item.get("abstract", "") or "",
        source_api="semantic_scholar",
        semantic_scholar_id=item.get("paperId", "") or "",
        url=item.get("url", "") or "",
        keywords="; ".join(publication_types),
        search_query=query,
        relevance_notes="",
        collected_at=collected_at,
    )


def _normalize_crossref(item: dict, query: str, collected_at: str) -> dict[str, Any]:
    authors = [
        " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part)
        for author in item.get("author", [])
    ]
    title = _first(item.get("title"))
    journal = _first(item.get("container-title"))

    return empty_paper_record(
        doi=normalize_doi(item.get("DOI", "")),
        title=title,
        year=_crossref_year(item),
        authors="; ".join(author for author in authors if author),
        journal=journal,
        abstract=item.get("abstract", "") or "",
        source_api="crossref",
        url=item.get("URL", "") or "",
        keywords="",
        search_query=query,
        relevance_notes="",
        collected_at=collected_at,
    )


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = _dedupe_key(record)
        if key not in deduped:
            deduped[key] = record
            continue
        _merge_records(deduped[key], record)

    ordered = sorted(
        deduped.values(),
        key=lambda record: (
            str(record.get("year", "")),
            str(record.get("title", "")).lower(),
        ),
        reverse=True,
    )
    for index, record in enumerate(ordered, start=1):
        record["paper_id"] = f"P{index:06d}"
    return ordered


def _dedupe_key(record: dict[str, Any]) -> str:
    if record.get("doi"):
        return f"doi:{record['doi']}"
    if record.get("openalex_id"):
        return f"openalex:{record['openalex_id']}"
    if record.get("semantic_scholar_id"):
        return f"semantic:{record['semantic_scholar_id']}"
    return f"title:{_normalize_title(str(record.get('title', '')))}"


def _merge_records(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key in PAPER_SCHEMA:
        if not existing.get(key) and incoming.get(key):
            existing[key] = incoming[key]

    existing_sources = set(str(existing.get("source_api", "")).split("; "))
    incoming_sources = set(str(incoming.get("source_api", "")).split("; "))
    sources = sorted(source for source in existing_sources | incoming_sources if source)
    existing["source_api"] = "; ".join(sources)


def normalize_doi(value: Any) -> str:
    doi = str(value or "").strip()
    lower = doi.lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if lower.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi.strip().lower()


def _openalex_abstract(item: dict) -> str:
    if item.get("abstract"):
        return item["abstract"]

    inverted = item.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""

    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions:
            positioned_words.append((int(position), word))
    return " ".join(word for _, word in sorted(positioned_words))


def _first(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or "")


def _crossref_year(item: dict) -> str:
    for key in ["published-print", "published-online", "issued"]:
        date_parts = item.get(key, {}).get("date-parts")
        if date_parts and date_parts[0]:
            return str(date_parts[0][0])
    return ""


def _normalize_title(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()


def _timestamp_slug(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("+", "Z")
