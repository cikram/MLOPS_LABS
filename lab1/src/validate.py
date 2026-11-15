"""Step 3 - validation rules (data tests)."""
import pandas as pd

from .common import month_bounds


def make_rules(ym, R):
    start, end = month_bounds(ym)
    minutes = lambda d: (d.tpep_dropoff_datetime - d.tpep_pickup_datetime).dt.total_seconds() / 60
    speed = lambda d: (d.trip_distance / (minutes(d) / 60)).where(minutes(d) > 0)   # NaN when duration <= 0
    return {
        # name: (function returning a boolean Series, True = VALID row,  max tolerated violation rate)
        "dropoff after pickup": (lambda d: d.tpep_dropoff_datetime > d.tpep_pickup_datetime, 0.001),
        "pickup inside file month": (lambda d: d.tpep_pickup_datetime.ge(start) & d.tpep_pickup_datetime.lt(end), 0.001),
        "duration in range": (lambda d: minutes(d).between(R["MIN_DURATION"], R["MAX_DURATION"]), 0.02),
        "distance in range": (lambda d: d.trip_distance.between(R["MIN_DISTANCE"], R["MAX_DISTANCE"]), 0.001),
        "speed plausible": (lambda d: speed(d).fillna(0) <= R["MAX_SPEED"], 0.001),
        "fare not negative": (lambda d: d.fare_amount >= R["MIN_FARE"], 0.02),
        "passenger_count valid or missing": (lambda d: d.passenger_count.isna() | d.passenger_count.between(R["PAX_MIN"], R["PAX_MAX"]), 0.02),
        "location ids in 1..265": (lambda d: d.PULocationID.between(1, 265) & d.DOLocationID.between(1, 265), 0.0001),
        "payment_type documented": (lambda d: d.payment_type.isin([0, 1, 2, 3, 4, 5, 6]), 0.0001),
        "RatecodeID documented or missing": (lambda d: d.RatecodeID.isna() | d.RatecodeID.isin([1, 2, 3, 4, 5, 6, 99]), 0.0001),
        "VendorID documented": (lambda d: d.VendorID.isin([1, 2, 6, 7]), 0.0001),
        "passenger_count not null": (lambda d: d.passenger_count.notna(), 0.10),
    }


def validate(df, ym, R):
    rows = []
    for name, (fn, max_rate) in make_rules(ym, R).items():
        valid = fn(df).fillna(False)
        v = int((~valid).sum())
        rate = v / len(df)
        rows.append((name, v, round(rate * 100, 3), round(max_rate * 100, 3), "PASS" if rate <= max_rate else "FAIL"))
    return pd.DataFrame(rows, columns=["rule", "violations", "rate_%", "max_rate_%", "status"])