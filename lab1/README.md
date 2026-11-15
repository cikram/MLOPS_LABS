# MLOps Data Lab: NYC Yellow Taxi Trip Duration

A small, reproducible MLOps pipeline that predicts the **duration of a NYC yellow taxi trip (in minutes)**, versions its data and model, and then **monitors the model over time** (data drift, concept drift, retraining decision) with a scheduled Airflow job.

The lab follows one idea from start to finish: a model is only useful if its data is validated, its results are reproducible, and its degradation is detected before users notice it.

## What this project covers

| Stage | What it does |
|---|---|
| Validate | Checks a raw monthly batch against explicit rules (ranges, missing values, consistency) before anything else runs |
| Clean | Applies deterministic cleaning to the validated batch |
| Split | Builds train and test sets in a reproducible way |
| Train | Trains the trip duration model and stores it as a versioned artifact |
| Version | Records the data and model fingerprints (DVC pointer and a SHA-256 manifest) so any result can be traced back to its exact inputs |
| Monitor | Compares a new month to the reference month: feature drift (PSI), categorical drift (chi-square + Cramer's V), error drift (MAE) |
| Decide | Turns the monitoring numbers into one verdict: `OK`, `INVESTIGATE` or `RETRAIN` |
| Schedule and alert | Runs the monitoring every month with Airflow and writes an alert when the verdict is not `OK` |

## Project structure

```
lab1/
|-- src/
|   |-- common.py         # paths, constants (reference month, artifacts folder)
|   |-- validate.py       # validation rules applied to raw batches
|   |-- clean.py          # cleaning step
|   |-- split.py          # train / test split
|   |-- train.py          # model training and evaluation
|   |-- version.py        # manifest and fingerprints
|   |-- pipeline.py       # end-to-end pipeline (optional --monitor flag)
|   |-- monitor.py        # PSI, chi-square, MAE by hour, monitor(), decide()
|   `-- run_monitor.py    # one monitoring cycle: monitor, decide, alert, exit code
|-- dags/
|   `-- taxi_monitor_dag.py   # Airflow DAG (monthly schedule)
|-- airflow/
|   |-- docker-compose.yaml   # Airflow stack (Postgres, Redis, Celery workers)
|   |-- Dockerfile            # Airflow image + project dependencies
|   `-- requirements.txt      # Python dependencies installed in the image
|-- artifacts/                # trained model, metadata, generated alerts
`-- notebook                  # the lab notebook with all steps and answers
```

## Setup

Requirements: Python 3.10+ and, for the scheduled part, Docker Desktop.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install pandas scikit-learn==1.8.0 scipy joblib pyarrow
```

The model is saved with `joblib`, so the same scikit-learn version (1.8.0) must be used to load it. The Docker image pins it for the same reason.

## Run the pipeline

From the `lab1` folder:

```bash
python -m src.pipeline
```

This validates, cleans, splits and trains on the reference month (2024-01) and writes the model and its metadata to `artifacts/`.

To also print a drift report for one or several later months:

```bash
python -m src.pipeline --monitor 2024-02 2024-07
```

## Monitoring

### What is measured

- **PSI (Population Stability Index)** on numeric features (`trip_distance`, `fare_amount`, `pickup_hour`, `passenger_count`) and categorical features (`payment_type`, `PULocationID`). Bins come from the reference month. Rule of thumb: below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 major shift.
- **Chi-square test and Cramer's V** on categorical features. At this sample size the p-value is always close to 0, so Cramer's V (effect size) is what tells a real shift from a tiny one.
- **MAE** of the trained model on the new month, compared to its MAE on the reference test set (`mae_ratio`), and **MAE by pickup hour** to locate where the error grows (concept drift).
- **Validation rules** applied to the raw batch (`rules_failing`).

### Decision rule

| Verdict | Condition |
|---|---|
| `RETRAIN` | `mae_ratio >= 1.25`, or (`max_psi >= 0.25` and `mae_ratio >= 1.10`) |
| `INVESTIGATE` | `mae_ratio >= 1.10`, or `max_psi >= 0.10`, or `rules_failing > 0` |
| `OK` | none of the above |

Reference test MAE: about 5.02 minutes.

### Results

| Month | max PSI | MAE ratio | Verdict |
|---|---|---|---|
| 2024-02 | 0.005 | 1.03 | `OK` |
| 2024-04 | 0.061 | 1.11 | `INVESTIGATE` |
| 2024-07 | n/a | 1.13 | `INVESTIGATE` |
| 2020-04 | 0.603 | 1.32 | `RETRAIN` |

The reference is always January 2024, so the further a month is from it, the more the model degrades. April 2020 (COVID lockdown) is the clearest case: very different trips, two failing rules and the highest error.

### Run one monitoring cycle by hand

```bash
python -m src.run_monitor --month 2024-02
```

Without `--month`, the script monitors the previous calendar month. The NYC TLC publishes files with roughly a two month delay, so for tests it is safer to pass a month explicitly.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | `OK` or `INVESTIGATE` (a warning, the job stays green) |
| 2 | `RETRAIN` (the job is marked as failed on purpose) |
| 1 | unexpected error (missing file, failed download, bug) |

When the verdict is not `OK`, an alert is printed and appended to `artifacts/alerts.log`. It contains the max PSI, the MAE ratio, the number of failing rules, Cramer's V and the hours with the worst MAE ratio.

## Scheduling with Airflow

The DAG `taxi_monitor` (`dags/taxi_monitor_dag.py`) runs `python -m src.run_monitor` through a `BashOperator` with the cron expression `0 6 2 * *`, which means 06:00 UTC on the 2nd of every month. `catchup=False` means that after downtime only the latest missed run is executed, not one per missed month.

### Start Airflow with Docker Compose

```bash
cd airflow
docker compose build
docker compose up airflow-init
docker compose up -d
```

Then open `http://localhost:8080` (user `airflow`, password `airflow`), unpause the `taxi_monitor` DAG and trigger it manually. A `RETRAIN` verdict makes the task fail (return code 2) and the alert appears in the task logs and in `artifacts/alerts.log`.

Without Airflow, the same job can be scheduled with cron (Linux and macOS) or with the Windows Task Scheduler:

```bash
# cron
0 6 2 * * cd /path/to/lab1 && python -m src.run_monitor

# Windows Task Scheduler
schtasks /Create /SC MONTHLY /D 2 /ST 06:00 /TN "taxi_monitor" /TR "..."
```

### Stop and clean up

```bash
docker compose down -v --rmi all
docker builder prune -f
```

## Design choices

- **Deterministic pipeline:** the same raw input always produces the same cleaned data and the same split, so results can be compared across runs.
- **Validation before cleaning:** rules run on the raw batch, so problems in the source data are reported instead of being silently fixed.
- **Fixed reference month:** all later months are compared to January 2024, which keeps the drift numbers comparable over time.
- **Two signals for the decision:** data drift alone does not trigger a retrain, because a shifted feature can still leave the error acceptable. The decision combines drift with the actual MAE.
- **Warning vs failure:** `INVESTIGATE` keeps the Airflow task green, only `RETRAIN` fails it, so the red state always means that action is needed.

## Limitations

- Monitoring is a monthly batch job, not real time. It only runs while the Airflow scheduler is running.
- The alert is written to a log file. Email or Slack delivery can be added inside `send_alert()` in `src/run_monitor.py` without changing the rest of the pipeline.
- A red task does not by itself distinguish a `RETRAIN` from a crash: the task logs do (an alert message for `RETRAIN`, a traceback for a crash).
- Without `--month`, the previous month is targeted, which may not be published yet.