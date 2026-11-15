"""Step 5 (first part) - time-based split: train on the past, test on the future."""
from .common import SPLIT_DAY


def split_time(df, split_day=SPLIT_DAY):
    train_df = df[df["pickup_day"] <= split_day]
    test_df = df[df["pickup_day"] > split_day]
    assert len(train_df) > 0 and len(test_df) > 0, "empty split"
    assert train_df.tpep_pickup_datetime.max() < test_df.tpep_pickup_datetime.min(), "train must be strictly before test"
    return train_df, test_df