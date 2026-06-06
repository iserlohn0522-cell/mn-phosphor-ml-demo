from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phosphor_ml.literature.candidate_review import rank_candidate_papers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate papers for phosphor review.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "candidate_papers.csv",
        help="Candidate metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "candidate_papers_review.csv",
        help="Review CSV to write.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ranked = rank_candidate_papers(args.input, args.output)
    counts = {"high": 0, "medium": 0, "low": 0}
    for row in ranked:
        counts[row["priority"]] += 1

    print(f"Scored {len(ranked)} candidate papers.")
    print(f"High: {counts['high']}  Medium: {counts['medium']}  Low: {counts['low']}")
    print(f"Review CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
