from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phosphor_ml.features.composition_descriptors import (  # noqa: E402
    build_level2_feature_table,
    generate_composition_descriptor_row,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate formula-safe composition descriptors for phosphor rows.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "examples" / "synthetic_materials.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "composition_descriptors.csv",
    )
    parser.add_argument("--descriptor-backend", choices=["basic", "matminer"], default="matminer")
    parser.add_argument(
        "--allow-basic-fallback",
        action="store_true",
        help="Fall back to basic descriptors if Matminer/Pymatgen descriptors are unavailable.",
    )
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def formula_for_row(row: dict[str, str]) -> str:
    return row.get("composition_normalized") or row.get("composition_raw") or row.get("formula") or row.get("Formula") or ""


def generate_descriptor_rows(
    rows: list[dict[str, str]],
    *,
    descriptor_backend: str,
    allow_basic_fallback: bool,
) -> list[dict[str, object]]:
    formulas = [formula_for_row(row) for row in rows]
    feature_names, features = build_level2_feature_table(
        formulas,
        descriptor_backend=descriptor_backend,
        allow_basic_fallback=allow_basic_fallback,
    )
    output_rows: list[dict[str, object]] = []
    for row_index, source_row in enumerate(rows):
        descriptor_row = generate_composition_descriptor_row(formulas[row_index], descriptor_backend=descriptor_backend)
        merged: dict[str, object] = dict(source_row)
        merged.update(descriptor_row)
        for feature_name, value in zip(feature_names, features[row_index]):
            merged[feature_name] = float(value)
        output_rows.append(merged)
    return output_rows


def write_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.input)
    descriptor_rows = generate_descriptor_rows(
        rows,
        descriptor_backend=args.descriptor_backend,
        allow_basic_fallback=args.allow_basic_fallback,
    )
    write_rows(descriptor_rows, args.output)
    print(f"Generated descriptors for {len(descriptor_rows)} rows.")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
