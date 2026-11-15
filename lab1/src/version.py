"""Step 6 - version the data (manifest + hashes) and prove reproducibility."""
import datetime as dt
import json

import pandas as pd
import sklearn

from .clean import clean
from .common import (ART, FEATURES, PROC, R, RAW, ROWS_IN_FILE, SAMPLE_N, SEED, SPLIT_DAY, TARGET,
                     load_month, sha256)
from .split import split_time


def build_datasets(month, rules=R):
    """raw file -> clean -> time split. Same code, same seed -> same result."""
    raw = load_month(month)
    clean_df, drop_log = clean(raw, month, rules)
    train_df, test_df = split_time(clean_df)
    return raw, clean_df, drop_log, train_df, test_df


def write_version(month, clean_df, train_df, test_df, drop_log, rules=R):
    train_path, test_path = PROC / f"train_{month}.parquet", PROC / f"test_{month}.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    raw_path = RAW / f"yellow_tripdata_{month}.parquet"
    manifest = {
        "dataset": "nyc_yellow_taxi_duration", "version": month,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_file": raw_path.name,
        "source_sha256": sha256(raw_path),
        "rows_in_file": ROWS_IN_FILE[month],
        "sample_size": SAMPLE_N,
        "sample_seed": SEED,
        "cleaning_rules": rules,
        "drop_log": drop_log.to_dict("records"),
        "split_rule": f"pickup_day <= {SPLIT_DAY} -> train, else test",
        "features": FEATURES,
        "target": TARGET,
        "rows_clean": len(clean_df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "sklearn_version": sklearn.__version__,
    }
    manifest_path = ART / f"data_manifest_{month}.json"
    json.dump(manifest, open(manifest_path, "w"), indent=2, default=str)
    return manifest, manifest_path


def check_reproducibility(month, manifest, train_df, test_df):
    """Rebuild everything from the raw file and compare values and SHA-256 with the saved version."""
    _, _, _, train2, test2 = build_datasets(month)
    pd.testing.assert_frame_equal(train_df, train2)
    pd.testing.assert_frame_equal(test_df, test2)
    tmp_train, tmp_test = PROC / f"_check_train_{month}.parquet", PROC / f"_check_test_{month}.parquet"
    try:
        train2.to_parquet(tmp_train)
        test2.to_parquet(tmp_test)
        assert sha256(tmp_train) == manifest["train_sha256"], "train hash differs"
        assert sha256(tmp_test) == manifest["test_sha256"], "test hash differs"
    finally:
        tmp_train.unlink(missing_ok=True)
        tmp_test.unlink(missing_ok=True)