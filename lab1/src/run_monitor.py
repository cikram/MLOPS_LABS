"""Scheduled job:  python -m src.run_monitor [--month 2024-07]
Exit code: 0 = OK or INVESTIGATE, 2 = RETRAIN (cron/Airflow can then raise an alert)."""
import argparse
import datetime as dt
import json
import sys

import pandas as pd                                                      # NEW

from .common import ART
from .monitor import decide, diagnose, load_reference, monitor

REF_MAE = json.load(open(ART / "model_v1_meta.json"))["test_mae_min"]


def previous_month():
    first = dt.date.today().replace(day=1)
    return (first - dt.timedelta(days=1)).strftime("%Y-%m")


def format_alert(ym, row, status, diag):
    psi = row[[c for c in row.index if c.startswith("psi_")]].astype(float)   # CHANGED: .astype(float)
    lines = [f"[{status}] taxi trip duration model, month {ym}",
             f"  max PSI       : {psi.max():.3f} ({psi.idxmax()})",
             f"  MAE ratio     : {row['MAE_min'] / REF_MAE:.2f} (MAE {row['MAE_min']:.2f} min vs reference {REF_MAE:.2f} min)",
             f"  rules failing : {int(row['rules_failing'])}",
             "  Cramer's V    : " + ", ".join(f"{c}={v:.3f}" for c, v in diag["cramers_v"].items()),
             "  worst hours (MAE ratio vs reference):"]
    lines += [f"    {w['hour']:02d}h: {w['mae_cur']:.2f} vs {w['mae_ref']:.2f} min (x{w['ratio']:.2f})"
              for w in diag["worst_hours"]]
    return "\n".join(lines)


def send_alert(ym, row, status, diag):
    message = format_alert(ym, row, status, diag)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(ART / "alerts.log", "a", encoding="utf-8") as f:   # simple alert channel: a log file
        f.write(f"{stamp}\n{message}\n\n")
    print(message)
    # optional: send `message` by email (smtplib) or to a Slack webhook here


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=previous_month(), help="month to monitor (default: previous month)")
    ym = parser.parse_args().month

    row = pd.Series(monitor(ym))                                  # CHANGED: monitor() returns a dict
    status = decide(row, REF_MAE)
    print(f"{ym}: {status}")

    if status != "OK":                                            # INVESTIGATE or RETRAIN: add the diagnosis
        model, _, test_df = load_reference()                      # cached, already loaded by monitor()
        send_alert(ym, row, status, diagnose(test_df, model, ym))

    if status == "RETRAIN":
        sys.exit(2)                                               # non-zero exit code = failed job


if __name__ == "__main__":
    main()