from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phosphor_ml.schemas import MATERIAL_SCHEMA, PAPER_SCHEMA, validate_columns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CSV columns against the project schemas.")
    parser.add_argument(
        "--papers",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "candidate_papers.csv",
        help="Candidate paper metadata CSV.",
    )
    parser.add_argument(
        "--materials",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "material_records.csv",
        help="Material-level CSV. Dry-run metadata demos may omit this file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors: list[str] = []

    errors.extend(_validate_csv(args.papers, PAPER_SCHEMA, "paper metadata", required=True))
    errors.extend(_validate_csv(args.materials, MATERIAL_SCHEMA, "material records", required=False))

    if errors:
        for error in errors:
            print(error)
        return 1

    print("Schema check passed.")
    return 0


def _validate_csv(path: Path, expected_schema: list[str], label: str, required: bool) -> list[str]:
    if not path.exists():
        if required:
            return [f"Missing required {label} file: {path}"]
        print(f"Optional {label} file not found: {path}")
        return []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        issues = validate_columns(reader.fieldnames or [], expected_schema)

    return [f"{label}: {issue}" for issue in issues]


if __name__ == "__main__":
    raise SystemExit(main())
