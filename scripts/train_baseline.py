from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phosphor_ml.training.baseline import train_baseline_from_csv, write_training_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a composition-only ridge baseline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "examples" / "synthetic_materials.csv",
        help="Material CSV for demo training.",
    )
    parser.add_argument("--target", default="emission_peak_nm", help="Numeric target column.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "demo_baseline_emission_peak_nm",
        help="Output directory for metrics, predictions, and report.",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_baseline_from_csv(
        args.input,
        target_column=args.target,
        k_folds=args.k_folds,
        alpha=args.alpha,
        random_seed=args.random_seed,
    )
    write_training_outputs(result, args.output_dir)
    print(f"Baseline target: {args.target}")
    print(f"Labeled rows: {result.summary['n_rows']}")
    print(f"MAE={result.summary['mae']:.3f} RMSE={result.summary['rmse']:.3f} R2={result.summary['r2']:.3f}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
