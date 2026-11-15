"""Step 4 - deterministic cleaning with a log of what was removed."""
import numpy as np
import pandas as pd

from .common import KEEP, month_bounds


def clean(df, ym, R):
    d = df[[c for c in KEEP if c in df.columns]].copy()      # only raw columns
    start, end = month_bounds(ym)
    log = []
    d["duration_min"] = (d.tpep_dropoff_datetime - d.tpep_pickup_datetime).dt.total_seconds() / 60

    # order matters: speed needs a valid duration, so it comes after the duration filter
    steps = [("pickup inside file month", lambda x: x.tpep_pickup_datetime.ge(start) & x.tpep_pickup_datetime.lt(end)),
             ("duration in range",        lambda x: x.duration_min.between(R["MIN_DURATION"], R["MAX_DURATION"])),
             ("distance in range",        lambda x: x.trip_distance.between(R["MIN_DISTANCE"], R["MAX_DISTANCE"])),
             ("speed plausible",          lambda x: (x.trip_distance / (x.duration_min / 60)) <= R["MAX_SPEED"]),
             ("fare not negative",        lambda x: x.fare_amount >= R["MIN_FARE"])]
    for name, fn in steps:
        before = len(d)
        d = d[fn(d).fillna(False)]
        log.append((name, before - len(d)))
    d = d.copy()

    # invalid passenger counts become missing (the trip itself is kept)
    pax_bad = d.passenger_count.lt(R["PAX_MIN"]) | d.passenger_count.gt(R["PAX_MAX"])
    d.loc[pax_bad, "passenger_count"] = np.nan

    # time features
    d["pickup_hour"] = d.tpep_pickup_datetime.dt.hour
    d["pickup_dow"] = d.tpep_pickup_datetime.dt.dayofweek      # 0 = Monday
    d["pickup_day"] = d.tpep_pickup_datetime.dt.day            # only used for the split

    return d.reset_index(drop=True), pd.DataFrame(log, columns=["step", "rows_dropped"])