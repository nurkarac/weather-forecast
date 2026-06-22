"""
dashboard.py — Eskisehir Weather Forecast & Context-Aware AI Assistant

Description:
In this module, I developed an end-to-end Streamlit dashboard to visualize 
time-series weather data and predictions. I query the historical data and 
ARIMA_PLUS forecast results directly from Google BigQuery. 

To elevate this from a standard dashboard to an AI-driven product, I integrated 
a RAG (Retrieval-Augmented Generation) pipeline using the Google Gemini 2.5 Flash model. 
This allows the dashboard to act as a personal assistant, providing contextual, 
data-backed recommendations for the user's daily plans.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.cloud import bigquery

from config import (
    CREDENTIALS_PATH,  # noqa: F401
    LOCATION_NAME,
    TABLE_FORECAST,
    TABLE_HOURLY,
)

# ── Page Configuration ────────────────────────────────────────────────────────
# I set the layout to wide for better visualization of time-series charts.
st.set_page_config(
    page_title=f"{LOCATION_NAME} · Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS Injection ──────────────────────────────────────────────────────
# I injected custom CSS to create a modern, dark-themed UI that highlights 
# the metrics and aligns with professional dashboard design standards.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Global Background ── */
.stApp {
    background: #0b0f1a;
    font-family: 'DM Sans', sans-serif;
}

/* ── Hero Section ── */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #132044 50%, #0a1628 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e8f4fd;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #7ba7cc;
    margin-top: 0.35rem;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: 0.8rem;
    letter-spacing: 0.5px;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(145deg, #0f1e38, #0d1a30);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.35); }
.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 16px 16px;
}
.metric-card.blue::after   { background: linear-gradient(90deg, #63b3ed, #4299e1); }
.metric-card.orange::after { background: linear-gradient(90deg, #f6ad55, #ed8936); }
.metric-card.teal::after   { background: linear-gradient(90deg, #4fd1c5, #38b2ac); }
.metric-card.purple::after { background: linear-gradient(90deg, #b794f4, #9f7aea); }

.metric-label {
    font-size: 0.75rem;
    color: #6b8caf;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #e8f4fd;
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: #5a7d9e;
    margin-top: 0.4rem;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem;
}
.section-header h3 {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #63b3ed;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 0;
    font-weight: 400;
}
.section-divider {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.3), transparent);
}

/* ── Chart Cards ── */
.chart-card {
    background: #0d1a30;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-size: 0.8rem;
    color: #6b8caf;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── Streamlit Overrides ── */
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }
div[data-testid="stMetric"] { display: none; }
footer { display: none; }

/* ── Data Pills ── */
.data-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #7ba7cc;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── BigQuery Data Retrieval ───────────────────────────────────────────────────
# I implemented st.cache_resource and st.cache_data to optimize BigQuery 
# usage, reducing query costs and significantly improving dashboard load times.

@st.cache_resource
def get_client() -> bigquery.Client:
    return bigquery.Client()

@st.cache_data(ttl=1800, show_spinner=False)
def load_actual() -> pd.DataFrame:
    # Querying the last 168 hours (7 days) of actual weather data
    q = f"""
    SELECT time, temperature_2m, precipitation, windspeed_10m
    FROM `{TABLE_HOURLY}`
    ORDER BY time DESC
    LIMIT 168
    """
    return get_client().query(q).to_dataframe().sort_values("time")

@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast() -> pd.DataFrame:
    # Querying the 24-hour prediction values generated by BigQuery ML (ARIMA_PLUS)
    q = f"""
    SELECT forecast_timestamp, forecast_value,
           prediction_interval_lower_bound,
           prediction_interval_upper_bound
    FROM `{TABLE_FORECAST}`
    ORDER BY forecast_timestamp
    """
    return get_client().query(q).to_dataframe()

# ── Plotly Theming Configuration ──────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#7ba7cc", size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(
        gridcolor="rgba(99,179,237,0.07)",
        linecolor="rgba(99,179,237,0.15)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(99,179,237,0.07)",
        linecolor="rgba(99,179,237,0.15)",
        tickfont=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(13,26,48,0.8)",
        bordercolor="rgba(99,179,237,0.2)",
        borderwidth=1,
        font=dict(size=11, color="#a0c4e0"),
    ),
    hoverlabel=dict(
        bgcolor="#0d1a30",
        bordercolor="#63b3ed",
        font=dict(color="#e8f4fd", family="DM Sans"),
    ),
)

# ── Data Execution ────────────────────────────────────────────────────────────
with st.spinner("Fetching data from BigQuery..."):
    df_actual   = load_actual()
    df_forecast = load_forecast()

# ── Hero Header ───────────────────────────────────────────────────────────────
last_time = df_actual["time"].iloc[-1]
st.markdown(
    f"""
<div class="hero">
    <p class="hero-title">🌤️ {LOCATION_NAME} Weather Forecast</p>
    <p class="hero-subtitle">Powered by BigQuery ML & Gemini 2.5</p>
    <span class="hero-badge">LAST UPDATED · {last_time.strftime('%d %b %Y, %H:%M')}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ── KPI Metrics Computation & Rendering ───────────────────────────────────────
current_temp  = df_actual["temperature_2m"].iloc[-1]
forecast_max  = df_forecast["forecast_value"].max()
forecast_min  = df_forecast["forecast_value"].min()
avg_wind      = df_actual["windspeed_10m"].mean()
total_precip  = df_actual["precipitation"].sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""<div class="metric-card blue">
            <div class="metric-label">Current Temp</div>
            <div class="metric-value">{current_temp:.1f}°C</div>
            <div class="metric-sub">Latest measurement</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""<div class="metric-card orange">
            <div class="metric-label">24h Max Forecast</div>
            <div class="metric-value">{forecast_max:.1f}°C</div>
            <div class="metric-sub">Upper bound included</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""<div class="metric-card teal">
            <div class="metric-label">24h Min Forecast</div>
            <div class="metric-value">{forecast_min:.1f}°C</div>
            <div class="metric-sub">Lower bound included</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""<div class="metric-card purple">
            <div class="metric-label">Avg Wind Speed</div>
            <div class="metric-value">{avg_wind:.1f}</div>
            <div class="metric-sub">km/h · Last 7 days</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

# ── Main Temperature Chart (Actual vs Predicted) ──────────────────────────────
st.markdown(
    """<div class="section-header">
        <h3>Temperature · Actual vs Forecast</h3>
        <div class="section-divider"></div>
    </div>""",
    unsafe_allow_html=True,
)

fig_main = go.Figure()

# Plotting the 90% Confidence Interval generated by ARIMA_PLUS
fig_main.add_trace(
    go.Scatter(
        x=pd.concat(
            [df_forecast["forecast_timestamp"], df_forecast["forecast_timestamp"][::-1]]
        ),
        y=pd.concat(
            [
                df_forecast["prediction_interval_upper_bound"],
                df_forecast["prediction_interval_lower_bound"][::-1],
            ]
        ),
        fill="toself",
        fillcolor="rgba(246,173,85,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Interval (90%)",
        hoverinfo="skip",
    )
)

# Plotting Historical Data
fig_main.add_trace(
    go.Scatter(
        x=df_actual["time"],
        y=df_actual["temperature_2m"],
        name="Actual Temperature",
        line=dict(color="#63b3ed", width=2),
        mode="lines",
    )
)

# Plotting the Forecast Line
fig_main.add_trace(
    go.Scatter(
        x=df_forecast["forecast_timestamp"],
        y=df_forecast["forecast_value"],
        name="ARIMA Forecast",
        line=dict(color="#f6ad55", width=2.5, dash="dot"),
        mode="lines",
    )
)

fig_main.update_layout(
    **PLOT_LAYOUT,
    height=380,
    hovermode="x unified",
    xaxis_title="",
    yaxis_title="Temperature (°C)",
)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

# ── Sub-charts (Precipitation & Wind) ─────────────────────────────────────────
st.markdown(
    """<div class="section-header">
        <h3>Precipitation & Wind · Last 7 Days</h3>
        <div class="section-divider"></div>
    </div>""",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="chart-card"><div class="chart-title">🌧 Hourly Precipitation (mm)</div>', unsafe_allow_html=True)

    fig_rain = go.Figure(
        go.Bar(
            x=df_actual["time"],
            y=df_actual["precipitation"],
            marker=dict(
                color=df_actual["precipitation"],
                colorscale=[[0, "#1a365d"], [0.5, "#2b6cb0"], [1, "#63b3ed"]],
                showscale=False,
            ),
            hovertemplate="%{y:.2f} mm<extra></extra>",
        )
    )
    fig_rain.update_layout(**PLOT_LAYOUT, height=260, yaxis_title="mm")
    st.plotly_chart(fig_rain, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="chart-card"><div class="chart-title">💨 Wind Speed (km/h)</div>', unsafe_allow_html=True)

    fig_wind = go.Figure(
        go.Scatter(
            x=df_actual["time"],
            y=df_actual["windspeed_10m"],
            fill="tozeroy",
            fillcolor="rgba(79,209,197,0.1)",
            line=dict(color="#4fd1c5", width=1.8),
            hovertemplate="%{y:.1f} km/h<extra></extra>",
        )
    )
    fig_wind.update_layout(**PLOT_LAYOUT, height=260, yaxis_title="km/h")
    st.plotly_chart(fig_wind, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── Forecast Data Table ───────────────────────────────────────────────────────
with st.expander("Hourly Forecast Table"):
    df_show = df_forecast.rename(
        columns={
            "forecast_timestamp":              "Time",
            "forecast_value":                  "Forecast (°C)",
            "prediction_interval_lower_bound": "Lower Bound (°C)",
            "prediction_interval_upper_bound": "Upper Bound (°C)",
        }
    ).round(2)
    st.dataframe(df_show, use_container_width=True, hide_index=True)


# ── AI ASSISTANT (RAG / CONTEXT-AWARE PIPELINE) ───────────────────────────────
# In this section, I integrate the LLM. Rather than serving as a standard chatbot, 
# I structured it as a context-aware system by injecting live BigQuery ML metrics 
# directly into the system prompt. This ensures the model grounds its responses 
# on statistical truth.

import google.generativeai as genai

st.markdown(
    """<div class="section-header">
        <h3>AI Weather Assistant</h3>
        <div class="section-divider"></div>
    </div>""",
    unsafe_allow_html=True,
)

# Securely fetching the API key. 
# NOTE: In a production environment, I use st.secrets. Do not hardcode keys.
GEMINI_API_KEY = "AQ.Ab8RN6KDBwclO5hAQxtBJjz9_n5m84iuwzrqtMCcvJxSCM3uBQ"  # Replace with actual key or st.secrets implementation
genai.configure(api_key=GEMINI_API_KEY)

# I opted for the 'gemini-2.5-flash' model for its ultra-low latency and 
# high capability in executing instruction-following tasks.
model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize session state to maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("Ask for recommendations (e.g., Should I wear a jacket tonight?)"):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Compiling the structured data into natural language for the LLM context (Data Augmentation)
    context = f"Current temperature is {current_temp:.1f}°C. Over the next 24 hours, the forecast predicts a maximum of {forecast_max:.1f}°C and a minimum of {forecast_min:.1f}°C. Average wind speed is {avg_wind:.1f} km/h."
    
    # Defining system behavior and grounding constraints
    system_prompt = f"""
    You are a concise, helpful, and friendly AI weather assistant for our current location.
    Using the CURRENT DATA below, answer the user's question directly. 
    Provide actionable advice and recommendations; do not simply repeat the data like a robot.

    CURRENT DATA: {context}
    USER QUESTION: {prompt}
    """

    with st.chat_message("assistant"):
        with st.spinner("Assistant is thinking..."):
            try:
                # Request generation from the Gemini API
                response = model.generate_content(system_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"API Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="margin-top:3rem; padding:1.2rem 0; border-top:1px solid rgba(99,179,237,0.1);
     display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:0.75rem; color:#3a5f80;">
        Data Source: <strong style="color:#4a7fa0">Open-Meteo API</strong>
        &nbsp;·&nbsp;
        Model: <strong style="color:#4a7fa0">BigQuery ML ARIMA_PLUS</strong>
    </span>
    <span style="font-size:0.75rem; color:#3a5f80;">
        github.com/<strong style="color:#4a7fa0">nurkarac</strong>/weather-forecast
    </span>
</div>
""",
    unsafe_allow_html=True,
)


