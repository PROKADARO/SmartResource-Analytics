import streamlit as st
import pandas as pd
import os
# Import the data collection function directly to resolve cloud environment path conflicts
from data_collection import main as fetch_weather_data

# Configure the core deployment settings for the dashboard interface
st.set_page_config(
    page_title="AgriWater-Iberia AI Dashboard",
    page_icon="💧",
    layout="wide"
)

# Main architectural headers for the AgriWater system
st.title("💧 AgriWater-Iberia AI-Powered Management Dashboard")
st.markdown("### Production-Grade Resource Optimization & Environmental Telemetry (Algarve, Portugal)")

st.sidebar.header("⚙️ Dashboard Controls")

# Interactive toggle to surface live backend data automation tools
live_pipeline = st.sidebar.checkbox("Live Pipeline Control", value=False)

if live_pipeline:
    st.sidebar.subheader("🔄 Execute Data Pipeline")
    if st.sidebar.button("Fetch Latest API Data & Re-train AI Model"):
        try:
            with st.spinner("Connecting to OpenWeather API & executing data pipeline..."):
                # Run the backend execution path synchronously within the active virtual env
                fetch_weather_data()
            st.sidebar.success("Pipeline executed successfully! Fresh weather records saved.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Pipeline Error: {e}")

# Validate physical existence of runtime target artifact
data_file = 'live_weather_forecast.csv'

if not os.path.exists(data_file):
    st.error(f"❌ Error: '{data_file}' not found! Please trigger the data pipeline button in the sidebar.")
else:
    # Read and render live telemetric state once file serialization completes
    df = pd.read_csv(data_file)
    
    st.success("✅ Weather data pipeline is healthy and active!")
    
    # Segment operational layout into structured data visualization metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="📍 Target Location", value="Algarve, PT")
    with col2:
        latest_temp = df['temperature'].iloc[-1] if 'temperature' in df.columns else "N/A"
        st.metric(label="🌡️ Current Temperature", value=f"{latest_temp} °C")
    with col3:
        latest_humidity = df['humidity'].iloc[-1] if 'humidity' in df.columns else "N/A"
        st.metric(label="💧 Humidity Level", value=f"{latest_humidity} %")
        
    st.markdown("---")
    st.subheader("📊 Historical & Forecasted Telemetry Data")
    st.dataframe(df, use_container_width=True)