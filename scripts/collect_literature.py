from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phosphor_ml.config import Config
from phosphor_ml.literature.search_pipeline import DEFAULT_QUERIES, run_literature_search
from phosphor_ml.utils.logging_utils import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect candidate Mn-based phosphor papers.")
    parser.add_argument(
        "--query",
        action="append",
        help="Search query. Repeat this option to run multiple queries. Defaults to the Phase 1 query set.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=25,
        help="Maximum records to request per query per source.",
    )
    parser.add_argument(
        "--source",
        choices=["openalex", "semantic_scholar", "crossref", "all"],
        default="all",
        help="Literature source to query.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use local fixtures instead of calling external APIs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_results < 1:
        raise SystemExit("--max-results must be at least 1")

    config = Config.from_env(project_root=PROJECT_ROOT)
    configure_logging(config.log_level)
    queries = args.query or DEFAULT_QUERIES

    records = run_literature_search(
        queries=queries,
        max_results=args.max_results,
        source=args.source,
        config=config,
        dry_run=args.dry_run,
    )

    output_path = config.data_dir / "interim" / "candidate_papers.csv"
    print(f"Collected {len(records)} candidate paper records.")
    print(f"Wrote cleaned metadata to {output_path}")
    print(f"Wrote raw responses under {config.data_dir / 'raw' / 'literature'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
