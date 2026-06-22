"""
fetch_data.py — Data Ingestion from Open-Meteo API

Description:
This script acts as the first step in the ETL pipeline. It establishes a robust 
connection to the Open-Meteo API to fetch historical and forecasted weather data. 
To ensure production-level reliability, I implemented caching to reduce API load 
and a retry mechanism with backoff logic to handle transient network failures. 
The extracted data is then structured into a Pandas DataFrame, cleaned, and 
staged locally as a CSV before being uploaded to Google BigQuery.

Usage:
    python fetch_data.py
"""

import logging
import sys
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from config import (
    FORECAST_DAYS,
    HOURLY_VARIABLES,
    LATITUDE,
    LOCATION_NAME,
    LONGITUDE,
    PAST_DAYS,
    RAW_CSV,
)

# ── Logger Configuration ──────────────────────────────────────────────────────
# I use the logging library instead of standard print statements to maintain 
# a professional audit trail of the pipeline's execution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def build_client() -> openmeteo_requests.Client:
    """
    Builds and returns an Open-Meteo client with caching and retry capabilities.
    
    Engineering Decision: 
    - Cache session prevents redundant API calls within a 1-hour window.
    - Retry session ensures the pipeline doesn't break due to temporary timeouts.
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_weather(client: openmeteo_requests.Client) -> pd.DataFrame:
    """
    Fetches raw weather data from the API and transforms it into a clean DataFrame.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": HOURLY_VARIABLES,
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
    }

    log.info("Sending request to Open-Meteo API — Location: %s (%.2f, %.2f)", LOCATION_NAME, LATITUDE, LONGITUDE)

    try:
        responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    except Exception as exc:
        log.error("API request failed: %s", exc)
        sys.exit(1) # Fail fast if the API is unreachable

    # Process the first location (assuming single location query)
    hourly = responses[0].Hourly()

    # Construct the timeseries dataframe
    df = pd.DataFrame(
        {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation":  hourly.Variables(1).ValuesAsNumpy(),
            "windspeed_10m":  hourly.Variables(2).ValuesAsNumpy(),
        }
    )

    # Data Quality Check: Drop missing values before staging
    before_len = len(df)
    df = df.dropna()
    log.info("Extracted %d rows. Removed %d rows containing NaN values.", before_len, before_len - len(df))

    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Saves the cleaned DataFrame to the specified local path (staging area).
    Creates parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Data successfully staged to local storage → %s", path)


def main() -> None:
    client = build_client()
    df = fetch_weather(client)
    save_csv(df, RAW_CSV)
    log.info("✅ fetch_data pipeline completed. Final Data Shape: %s", df.shape)


if __name__ == "__main__":
    main()