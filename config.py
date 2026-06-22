"""
config.py — Centralized Configuration Management

Description:
This module stores environment variables, GCP credentials, and global parameters 
for the ETL pipeline and the Streamlit dashboard. 

SECURITY NOTE: 
Never hardcode actual GCP Service Account JSON paths or API keys here in production.
Use environment variables or Streamlit secrets.
"""

import os
from pathlib import Path

# ── Authentication & Credentials ──────────────────────────────────────────────
# In a local dev environment, set the GOOGLE_APPLICATION_CREDENTIALS env var.
# For GitHub sharing, we use a placeholder path to protect sensitive keys.
CREDENTIALS_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path.home() / "path" / "to" / "your" / "gcp-service-account.json"),
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# ── Google Cloud / BigQuery Architecture ──────────────────────────────────────
# Replace 'YOUR_PROJECT_ID' with your actual Google Cloud Project ID before running locally.
PROJECT_ID      = "YOUR_PROJECT_ID"
DATASET_ID      = "weather_dataset"
TABLE_HOURLY    = f"{PROJECT_ID}.{DATASET_ID}.hourly_data"
TABLE_FORECAST  = f"{PROJECT_ID}.{DATASET_ID}.forecast_results"
MODEL_ID        = f"{PROJECT_ID}.{DATASET_ID}.temp_arima_model"

# ── Open-Meteo API Parameters ─────────────────────────────────────────────────
LOCATION_NAME   = "Eskisehir, Turkey"
LATITUDE        = 39.78
LONGITUDE       = 30.52
PAST_DAYS       = 90          # Historical data window for ARIMA training
FORECAST_DAYS   = 1           # 24-hour prediction horizon

HOURLY_VARIABLES = ["temperature_2m", "precipitation", "windspeed_10m"]

# ── Local File System ─────────────────────────────────────────────────────────
RAW_CSV = Path("data") / "weather_data.csv"