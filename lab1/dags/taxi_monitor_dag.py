from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="taxi_monitor",
    schedule="0 6 2 * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    BashOperator(
        task_id="monitor",
        bash_command="cd /opt/lab1 && python -m src.run_monitor",
    )