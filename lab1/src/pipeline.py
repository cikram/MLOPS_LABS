"""Command line entry point:  python -m src.pipeline --month 2024-03"""
import argparse
import sys

from .common import R
from .train import train_and_evaluate
from .validate import validate
from .version import build_datasets, check_reproducibility, write_version


def main():
    parser = argparse.ArgumentParser(description="NYC yellow taxi data pipeline (steps 3 to 6)")
    parser.add_argument("--month", required=True, help="month to process, e.g. 2024-03")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 if the cleaned data fails a validation rule")
    args = parser.parse_args()
    month = args.month

    # Steps 4 and 5 (split) through build_datasets; validation of the raw data first (Step 3)
    raw, clean_df, drop_log, train_df, test_df = build_datasets(month)
    print(f"\n[1/5] validation of the raw data ({month})")
    print(validate(raw, month, R).to_string(index=False))

    print("\n[2/5] cleaning")
    print(drop_log.to_string(index=False))
    print(f"{len(raw):,} -> {len(clean_df):,} rows")

    print("\n[3/5] validation of the cleaned data")
    report_clean = validate(clean_df, month, R)
    print(report_clean.to_string(index=False))
    if (report_clean.status == "FAIL").any():
        print("WARNING: some rules still fail after cleaning")
        if args.strict:
            sys.exit(1)

    print(f"\n[4/5] time-based split: train {len(train_df):,} rows | test {len(test_df):,} rows")
    _, metrics = train_and_evaluate(train_df, test_df)
    print(f"baseline MAE {metrics['baseline_mae_min']} min | model MAE {metrics['test_mae_min']} min")

    print("\n[5/5] versioning")
    manifest, path = write_version(month, clean_df, train_df, test_df, drop_log)
    print(f"manifest written: {path}")
    check_reproducibility(month, manifest, train_df, test_df)
    print("reproducible: rebuilt train/test are identical (values and SHA-256)")


if __name__ == "__main__":
    main()