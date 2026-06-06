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
    parser = argparse.ArgumentParser(description="Compare regression models for the demo material table.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "examples" / "synthetic_materials.csv",
        help="Material CSV for demo training.",
    )
    parser.add_argument("--target", default="emission_peak_nm", help="Numeric target column.")
    parser.add_argument("--feature-level", choices=["level1", "level2"], default="level1")
    parser.add_argument(
        "--feature-set",
        choices=["element_fraction", "descriptor_columns"],
        default=None,
        help="Feature source: element_fraction maps to Level 1; descriptor_columns reads precomputed columns.",
    )
    parser.add_argument(
        "--descriptor-backend",
        choices=["basic", "matminer"],
        default="basic",
        help="Descriptor backend for Level 2 features. Ignored for Level 1.",
    )
    parser.add_argument(
        "--allow-basic-fallback",
        action="store_true",
        help="Use basic descriptors if Matminer/Pymatgen is not installed.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to run.",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--group-column",
        default="doi",
        help="CSV column for grouped cross-validation. Sparse groups use KFold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "demo_model_suite",
        help="Output directory for metrics and reports.",
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
    print(f"Model families: {len(payload['models'])}")
    print(f"Target: {args.target}")
    print(f"Best by MAE: {best['model']} MAE={best['mae']:.3f} RMSE={best['rmse']:.3f} R2={best['r2']:.3f}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
