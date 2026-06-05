from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phosphor_ml.training.model_suite import run_model_suite_from_csv, write_model_suite_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare phosphor regression model families.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "examples" / "synthetic_materials.csv",
        help="Canonical material CSV used for demo supervised training.",
    )
    parser.add_argument("--target", default="emission_peak_nm", help="Numeric target column to predict.")
    parser.add_argument("--feature-level", choices=["level1", "level2"], default="level1")
    parser.add_argument(
        "--feature-set",
        choices=["element_fraction", "descriptor_columns"],
        default=None,
        help="Compatibility option: element_fraction maps to Level 1; descriptor_columns reads precomputed descriptor columns.",
    )
    parser.add_argument(
        "--descriptor-backend",
        choices=["basic", "matminer"],
        default="basic",
        help="Descriptor backend for level2. Ignored for level1.",
    )
    parser.add_argument(
        "--allow-basic-fallback",
        action="store_true",
        help="Use basic descriptors if Matminer/Pymatgen are unavailable.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to run.",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--group-column",
        default="doi",
        help="Optional CSV column used for grouped cross-validation. Empty or low-cardinality groups fall back to KFold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "demo_model_suite",
        help="Directory for metrics and report output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_model_suite_from_csv(
        args.input,
        target_column=args.target,
        feature_level="level1" if args.feature_set == "element_fraction" else args.feature_level,
        feature_set=args.feature_set,
        descriptor_backend=args.descriptor_backend,
        allow_basic_fallback=args.allow_basic_fallback,
        group_column=args.group_column,
        model_names=args.models,
        k_folds=args.k_folds,
        random_seed=args.random_seed,
    )
    write_model_suite_outputs(payload, args.output_dir)
    best = payload["best_by_mae"]
    print(f"Trained {len(payload['models'])} model families for {args.target}.")
    print(f"Best by MAE: {best['model']} MAE={best['mae']:.3f} RMSE={best['rmse']:.3f} R2={best['r2']:.3f}")
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
