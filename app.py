import streamlit as st
import pandas as pd
import numpy as np
import subprocess

# Set professional page configuration
st.set_page_config(page_title="AgriWater-Iberia Analytics", page_icon="💧", layout="wide")

st.title("💧 AgriWater-Iberia AI-Powered Management Dashboard")
st.markdown("### Production-Grade Resource Optimization & Environmental Telemetry (Algarve, Portugal)")

# --- 1. PIPELINE AUTOMATION (Refresh Button) ---
st.markdown("#### 🔄 Live Pipeline Control")
if st.button("🔄 Fetch Latest API Data & Re-train AI Model"):
    with st.spinner("Connecting to OpenWeather API and updating Random Forest forecasts..."):
        try:
            # Run data collection and machine learning scripts sequentially
            subprocess.run(["python", "data_collection.py"], check=True)
            subprocess.run(["python", "ml_forecasting.py"], check=True)
            st.success("🎉 Data pipeline executed successfully! Latest weather records loaded and predictions updated.")
        except Exception as e:
            st.error(f"❌ Pipeline Error: {e}")

st.markdown("---")

# Load the enhanced predictive data
try:
    df = pd.read_csv("live_weather_forecast.csv")
except FileNotFoundError:
    st.error("❌ Error: live_weather_forecast.csv not found! Please trigger the data pipeline button above.")
    st.stop()

# --- 2. SMART UI/UX: KPI METRICS CARDS ---
col1, col2, col3 = st.columns(3)

with col1:
    max_temp = df['temperature'].max()
    st.metric(label="🌡️ Max Expected Temp", value=f"{round(max_temp, 2)} °C", delta="High Evaporation Risk" if max_temp > 18 else "Stable")

with col2:
    avg_hum = df['humidity'].mean()
    st.metric(label="💧 Average Humidity", value=f"{round(avg_hum, 1)} %", delta="- Dry Air" if avg_hum < 75 else "Good Moisture", delta_color="inverse")

with col3:
    total_water = df['predicted_water_demand'].sum()
    st.metric(label="🚰 Total Predicted Water Demand", value=f"{round(total_water, 1)} m³", delta="Action Required" if total_water > 500 else "Normal")

st.markdown("---")

# --- 3. INTELLIGENT ALERTS SYSTEM ---
critical_events = df[df['predicted_water_demand'] > 25]

if not critical_events.empty:
    st.error(f"⚠️ **Critical Alert:** High Resource Demand Detected! There are {len(critical_events)} forecasted intervals where water demand exceeds 25 m³. Optimize hydraulic pressure immediately.")
    with st.expander("🔍 View Critical High-Demand Hours"):
        # Explicitly check for 'hour' column to avoid display issues
        alert_cols = ['datetime', 'hour', 'temperature', 'humidity', 'predicted_water_demand'] if 'hour' in df.columns else ['datetime', 'temperature', 'humidity', 'predicted_water_demand']
        st.dataframe(critical_events[alert_cols])
else:
    st.success("✅ **System Status:** Resource consumption forecasts are within safe sustainable thresholds for the next few days.")

# --- 4. ADVANCED VISUALIZATION ---
st.subheader("📊 Environmental Factors vs. AI Predictions")
available_features = ['temperature', 'humidity', 'wind_speed', 'pressure']
if 'hour' in df.columns:
    available_features.append('hour')  # Dynamically add the engineered time-feature

feature_choice = st.selectbox("Select a feature to contrast with AI Predicted Water Demand:", available_features)

# Prepare chart data
chart_data = df.set_index('datetime')[['predicted_water_demand', feature_choice]]
st.line_chart(chart_data)

# --- 5. DATA TABLE VIEW ---
if st.checkbox("Show Combined Feature & Prediction Table"):
    st.subheader("📋 Forecast and AI Demand Output")
    display_cols = ['datetime', 'temperature', 'humidity', 'wind_speed', 'predicted_water_demand']
    if 'hour' in df.columns:
        display_cols.insert(1, 'hour')  # Insert 'hour' right after 'datetime' for clear structure
    st.dataframe(df[display_cols])