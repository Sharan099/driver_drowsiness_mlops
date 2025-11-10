from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess.preprocess_data import preprocess_data
from train.train_model import train_model
from data.process import get_loaders


# -----------------------------
# Define default args
# -----------------------------
default_args = {
    'owner': 'Sharan',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# -----------------------------
# Define DAG
# -----------------------------
dag = DAG(
    'drowsiness_retrain',
    default_args=default_args,
    description='Retrain CNN model when new data is collected',
    schedule_interval=None,  # or None for manual trigger
)

# -----------------------------
# Airflow Tasks
# -----------------------------
task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

task_prepare_loaders = PythonOperator(
    task_id='prepare_loaders',
    python_callable=prepare_loaders,
    provide_context=True,
    dag=dag
)

task_train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

# -----------------------------
# Task dependencies
# -----------------------------
task_preprocess >> task_prepare_loaders >> task_train_model