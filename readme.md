
# 🌤️ Eskisehir Weather Forecast & Context-Aware AI Assistant

An end-to-end Data Engineering & Machine Learning pipeline that extracts real-time weather data, trains a time-series forecasting model directly in the cloud, and serves a Context-Aware AI assistant using Large Language Models (LLMs).

## 🚀 Project Overview
This project demonstrates a modern, production-grade cloud-native data architecture. Instead of relying solely on static weather APIs, the system builds its own intelligence by:
1. **Extracting** historical and live weather data using robust software practices.
2. **Centralizing** and staging data seamlessly into a cloud data warehouse.
3. **Training** an automated statistical model (`ARIMA_PLUS`) inside Google BigQuery to generate future forecasts.
4. **Visualizing** predictions with a 90% confidence interval via an interactive dashboard.
5. **Augmenting Data** via a RAG (Retrieval-Augmented Generation) pipeline using the **Google Gemini 2.5 Flash** model, enabling context-aware, conversational decision support.
![alt text](image-1.png)
![alt text](image-3.png)
![alt text](image-2.png)
## 🛠️ Tech Stack & MLOps Architecture

- **Language:** Python 3.13
- **Data Ingestion:** Open-Meteo API, `requests-cache`, `retry-requests` (Fault-Tolerant Ingestion)
- **Data Warehouse & ML Engine:** Google Cloud Platform (GCP), Google BigQuery ML (BQML)
- **Generative AI Framework:** Google AI Studio (Gemini 2.5 Flash API)
- **Frontend / UI:** Streamlit, Plotly (Dynamic & Interactive Data Visualization)

### The Modular Pipeline Flow
- `fetch_data.py`: Connects to the weather API with caching/retry logic, validates data quality, and stages the data locally as a clean CSV.
- `upload_to_bq.py`: Ingests the staged CSV into BigQuery using an idempotent `WRITE_TRUNCATE` strategy, triggers the cloud-native `ARIMA_PLUS` training algorithm, and materializes a 24-hour prediction horizon table.
- `dashboard.py`: Connects to live and forecasted BigQuery tables, computes metrics, renders automated time-series visualizations, and wraps the live analytics into the Gemini LLM system prompt for localized, conversational reasoning.
- `config.py`: Manages global environment constants, project tracking identifiers, and secure authentication placeholders.

## 🧠 AI Assistant Demo Questions

Once the dashboard is active, the grounded Gemini Assistant can be tested with context-driven queries such as:
- "I am planning to go for a run tonight at 8 PM. Looking at the wind speed and temperature, is it a good idea?"
- "Should I wear a thick coat tomorrow morning?"
- "Is there a huge difference between the maximum and minimum forecast today?"
![alt text](image-4.png)
## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/nurkarac/weather-forecast.git](https://github.com/nurkarac/weather-forecast.git)
cd weather-forecast

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

**3. Configure Google Cloud Credentials:**

* Set up a Service Account in your GCP Console with BigQuery Admin permissions.
* Download the private authentication key in JSON format.
* Open `config.py` and replace the placeholder with your local file path or initialize the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in your environment.
* Update the `PROJECT_ID` parameter with your unique Google Cloud project ID.

**4. Set up Gemini API Key:**

* Generate a free API key via [Google AI Studio](https://aistudio.google.com/).
* Open `dashboard.py` and insert your key into the `genai.configure(api_key="...")` block, or manage it securely using Streamlit Secrets.

## 🏃‍♂️ Running the Pipeline

Execute the data pipeline and application modules in the following engineering order:

```bash
# Step 1: Ingest the latest weather metrics
python fetch_data.py

# Step 2: Update Data Warehouse & trigger Cloud-Native ML training
python upload_to_bq.py

# Step 3: Launch the interactive analytics dashboard and AI assistant
streamlit run dashboard.py

```

## 📝 License

This project is developed for educational, portfolio, and professional demonstration purposes. Core meteorological data is provided courtesy of the Open-Meteo API.

