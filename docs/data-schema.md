# Data Schema

This demo uses explicit CSV schemas so literature metadata, material records, and downstream feature tables can be validated before modeling. The included material table is synthetic and exists only to exercise the pipeline.

## Paper Metadata

Paper records are defined by `PAPER_SCHEMA` in `src/phosphor_ml/schemas.py`.

| Field | Purpose |
| --- | --- |
| `paper_id` | Stable local identifier assigned after collection and deduplication. |
| `doi` | Normalized DOI when available; useful for deduplication and traceability. |
| `title` | Paper title from a literature source. |
| `year` | Publication year when available. |
| `authors` | Semicolon-delimited author display names from the source API. |
| `journal` | Journal, venue, or source title. |
| `abstract` | Abstract text when the source API provides it. |
| `source_api` | Source used for the record, such as OpenAlex, Semantic Scholar, or Crossref. |
| `openalex_id` | OpenAlex work identifier when applicable. |
| `semantic_scholar_id` | Semantic Scholar paper identifier when applicable. |
| `url` | Landing page or source URL. |
| `keywords` | Concepts, publication types, or other source keywords. |
| `search_query` | Query that produced the record. |
| `relevance_notes` | Manual notes field for later review. |
| `collected_at` | UTC timestamp for reproducibility. |

## Material Records

Material records are defined by `MATERIAL_SCHEMA` in `src/phosphor_ml/schemas.py`.

| Field group | Fields | Why it exists |
| --- | --- | --- |
| Identity and traceability | `material_id`, `paper_id`, `doi`, `data_source`, `extraction_status` | Links each material row back to a paper or source and records whether the row is synthetic, curated, or pending review. |
| Composition | `composition_raw`, `composition_normalized`, `host_formula`, `dopant`, `dopant_valence`, `activator` | Separates human-entered formula text from normalized formula fields used by feature generation. |
| Screening flags | `contains_mn`, `mn_valence`, `is_rare_earth_free`, `rare_earth_elements_detected` | Captures the basic chemical filters needed for Mn phosphor candidate triage. |
| Optical labels | `emission_peak_nm`, `excitation_peak_nm`, `fwhm_nm`, `plqy`, `lifetime`, `stability_metric` | Provides standard columns for supervised learning when validated public or licensed measurements are available. |
| Synthesis and structure | `synthesis_method`, `structure_info`, `space_group` | Preserves context that may later support richer modeling or manual review. |
| Notes | `notes` | Keeps row-level caveats separate from numeric labels. |

## Validation

Run schema validation with:

```bash
python scripts/validate_dataset.py --materials data/examples/synthetic_materials.csv --papers data/interim/candidate_papers.csv
```

Validation checks column presence and ordering. It does not certify scientific correctness; it only confirms that the tables are structurally compatible with the demo workflow.
