"""Shared constants and helpers (the notebook setup cell)."""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# paths are anchored on the project folder (the one containing src/), not on the current directory
ROOT = Path(__file__).resolve().parent.parent
RAW, PROC, ART = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "artifacts"
for p in (RAW, PROC, ART):
    p.mkdir(parents=True, exist_ok=True)

SEED, SAMPLE_N = 42, 200_000          # we sample 200k rows per month to keep the pipeline fast
REF_MONTH = "2024-01"                 # reference month used in the notebook (data version v1)
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
KEEP = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance", "RatecodeID",
        "PULocationID", "DOLocationID", "payment_type", "fare_amount", "tip_amount", "total_amount"]

# thresholds decided in Step 2g
R = dict(MIN_DURATION=1, MAX_DURATION=180,      # minutes
         MIN_DISTANCE=0, MAX_DISTANCE=100,      # miles
         MAX_SPEED=80,                          # mph
         MIN_FARE=0,                            # dollars
         PAX_MIN=1, PAX_MAX=6)                  # passengers

# modelling setup (Step 5)
FEATURES = ["pickup_hour", "pickup_dow", "PULocationID", "DOLocationID", "passenger_count", "VendorID"]
TARGET = "duration_min"
SPLIT_DAY = 24

ROWS_IN_FILE = {}


def month_bounds(ym):
    start = pd.Timestamp(ym + "-01")
    return start, start + pd.offsets.MonthBegin(1)


def download_month(ym):
    path = RAW / f"yellow_tripdata_{ym}.parquet"
    if not path.exists():
        print(f"downloading {ym} ...")
        urllib.request.urlretrieve(BASE_URL.format(ym=ym), path)
    return path


def load_month(ym, n=SAMPLE_N):
    path = download_month(ym)
    names = pq.ParquetFile(path).schema.names
    df = pd.read_parquet(path, columns=[c for c in KEEP if c in names])
    ROWS_IN_FILE[ym] = len(df)
    if n and len(df) > n:
        df = df.sample(n, random_state=SEED)
    return df.sort_values("tpep_pickup_datetime").reset_index(drop=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()