# Data Schema

The package keeps paper metadata and material records in explicit CSV schemas. The schemas make each stage easy to check before modeling: collection, review, descriptor generation, and training. The included material table uses synthetic rows.

## Paper Metadata

`src/phosphor_ml/schemas.py` lists the paper fields in `PAPER_SCHEMA`.

| Field | Purpose |
| --- | --- |
| `paper_id` | Local identifier assigned after collection and deduplication. |
| `doi` | Normalized DOI for deduplication and traceability. |
| `title` | Paper title from a literature source. |
| `year` | Publication year. |
| `authors` | Semicolon-delimited author display names from the source API. |
| `journal` | Journal, venue, or source title. |
| `abstract` | Abstract text from the source API. |
| `source_api` | Source used for the record, such as OpenAlex, Semantic Scholar, or Crossref. |
| `openalex_id` | OpenAlex work identifier. |
| `semantic_scholar_id` | Semantic Scholar paper identifier. |
| `url` | Landing page or source URL. |
| `keywords` | Concepts, publication types, or other source keywords. |
| `search_query` | Query that produced the record. |
| `relevance_notes` | Notes from manual review. |
| `collected_at` | UTC timestamp for reproducibility. |

## Material Records

`MATERIAL_SCHEMA` tracks formula fields, screening flags, target columns, and provenance.

| Field group | Fields | Purpose |
| --- | --- | --- |
| Identity and traceability | `material_id`, `paper_id`, `doi`, `data_source`, `extraction_status` | Links each material row to a paper or source and marks the row as synthetic, curated, or pending review. |
| Composition | `composition_raw`, `composition_normalized`, `host_formula`, `dopant`, `dopant_valence`, `activator` | Separates human-entered formula text from normalized formula fields used by feature generation. |
| Screening flags | `contains_mn`, `mn_valence`, `is_rare_earth_free`, `rare_earth_elements_detected` | Records the chemical filters used in Mn phosphor candidate triage. |
| Optical labels | `emission_peak_nm`, `excitation_peak_nm`, `fwhm_nm`, `plqy`, `lifetime`, `stability_metric` | Holds supervised targets once a public or licensed dataset supplies validated measurements. |
| Synthesis and structure | `synthesis_method`, `structure_info`, `space_group` | Keeps context that can support later modeling or manual review. |
| Notes | `notes` | Separates row caveats from numeric labels. |

## Validation

Run schema validation with:

```bash
python scripts/validate_dataset.py --materials data/examples/synthetic_materials.csv --papers data/interim/candidate_papers.csv
```

Validation checks column presence and ordering. Scientific review happens outside this structural check.
