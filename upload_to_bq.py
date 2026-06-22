"""
upload_to_bq.py — BigQuery Data Loading & Automated ML (ARIMA_PLUS) Training

Description:
This module handles the Data Warehousing and Machine Learning phases of the pipeline.
It reads the staged CSV data and loads it into Google BigQuery using an idempotent 
Write-Truncate strategy. Once the data is centralized, it executes a BigQuery ML 
(BQML) script to natively train an ARIMA_PLUS time-series model, leveraging 
Google's cloud compute rather than local resources. Finally, it generates and 
stores a 24-hour forecast with 90% confidence intervals.

Usage:
    python upload_to_bq.py
"""

import logging
import sys

import pandas as pd
import google.auth.exceptions
from google.cloud import bigquery

from config import (
    CREDENTIALS_PATH,  # noqa: F401 — Has the side effect of setting GOOGLE_APPLICATION_CREDENTIALS
    MODEL_ID,
    RAW_CSV,
    TABLE_FORECAST,
    TABLE_HOURLY,
)

# ── Logger Configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── BigQuery Schema Definition ────────────────────────────────────────────────
# Explicitly defining the schema prevents inference errors during the upload process.
HOURLY_SCHEMA = [
    bigquery.SchemaField("time",           "TIMESTAMP"),
    bigquery.SchemaField("temperature_2m", "FLOAT"),
    bigquery.SchemaField("precipitation",  "FLOAT"),
    bigquery.SchemaField("windspeed_10m",  "FLOAT"),
]

# ── 1. Data Ingestion to BigQuery ─────────────────────────────────────────────
def load_csv(path=RAW_CSV) -> pd.DataFrame:
    """Reads the staged CSV file from the local data directory."""
    if not path.exists():
        log.error("CSV file not found: %s — Please run fetch_data.py first.", path)
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["time"])
    log.info("CSV loaded successfully: %d rows", len(df))
    return df

def upload_to_bq(client: bigquery.Client, df: pd.DataFrame) -> None:
    """
    Uploads the DataFrame to BigQuery.
    
    Engineering Decision: 
    Using 'WRITE_TRUNCATE' ensures idempotency. The pipeline can run multiple 
    times safely without duplicating data, ensuring a fresh 7-day rolling window.
    """
    job_config = bigquery.LoadJobConfig(
        schema=HOURLY_SCHEMA,
        write_disposition="WRITE_TRUNCATE",
    )
    job = client.load_table_from_dataframe(df, TABLE_HOURLY, job_config=job_config)
    job.result()
    log.info("✅ Data successfully loaded into BigQuery → %s (%d rows)", TABLE_HOURLY, len(df))

# ── 2. BQML Model Training & Forecasting ──────────────────────────────────────
# Using ARIMA_PLUS enables automatic hyperparameter tuning, anomaly detection, 
# and seasonality decomposition directly within the data warehouse.
TRAIN_SQL = f"""
CREATE OR REPLACE MODEL `{MODEL_ID}`
OPTIONS(
    model_type                = 'ARIMA_PLUS',
    time_series_timestamp_col = 'time',
    time_series_data_col      = 'temperature_2m',
    auto_arima                = TRUE,
    data_frequency            = 'HOURLY',
    decompose_time_series     = TRUE
) AS
SELECT time, temperature_2m
FROM `{TABLE_HOURLY}`
ORDER BY time
"""

# Generating a 24-hour forecast horizon with a 90% confidence level.
FORECAST_SQL = f"""
CREATE OR REPLACE TABLE `{TABLE_FORECAST}` AS
SELECT *
FROM ML.FORECAST(
    MODEL `{MODEL_ID}`,
    STRUCT(24 AS horizon, 0.9 AS confidence_level)
)
"""

def train_model(client: bigquery.Client) -> None:
    """Executes the BigQuery ML training job."""
    log.info("Training ARIMA_PLUS model in BigQuery (This may take a few minutes)...")
    client.query(TRAIN_SQL).result()
    log.info("✅ Model training completed successfully → %s", MODEL_ID)

def run_forecast(client: bigquery.Client) -> None:
    """Generates the future predictions and saves them to a destination table."""
    log.info("Generating 24-hour forecast data...")
    client.query(FORECAST_SQL).result()
    log.info("✅ Forecast results saved → %s", TABLE_FORECAST)

# ── Main Execution Flow ───────────────────────────────────────────────────────
def main() -> None:
    try:
        client = bigquery.Client()
    except google.auth.exceptions.DefaultCredentialsError as exc:
        log.error("GCP Credentials not found. Please check your config.py setup: %s", exc)
        sys.exit(1)

    df = load_csv()
    upload_to_bq(client, df)
    train_model(client)
    run_forecast(client)
    log.info("🎉 Pipeline execution finished. To launch the dashboard, run: streamlit run dashboard.py")

if __name__ == "__main__":
    main()