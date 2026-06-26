import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🌲 Starting Advanced Machine Learning Pipeline (Random Forest)...")

# 1. Load the structured environmental data
try:
    df = pd.read_csv("live_weather_forecast.csv")
except FileNotFoundError:
    print("❌ Error: live_weather_forecast.csv not found! Run data_collection.py first.")
    exit()

# --- NEW: FEATURE ENGINEERING (Extracting Time Features) ---
# Convert datetime string to pandas datetime object
df['datetime'] = pd.to_datetime(df['datetime'])
# Extract the hour of the day (0 - 23)
df['hour'] = df['datetime'].dt.hour

# 2. Engineering Target Variable (Simulate smart irrigation demand)
np.random.seed(42)
# Added a mathematical effect for hour: higher consumption during peak daylight hours (12 PM - 4 PM)
daylight_effect = np.where((df['hour'] >= 12) & (df['hour'] <= 16), 5.0, 0.0)

df['target_water_demand'] = (df['temperature'] * 1.6) - (df['humidity'] * 0.15) + (df['wind_speed'] * 0.4) + daylight_effect + np.random.normal(10, 1.5, len(df))
df['target_water_demand'] = df['target_water_demand'].clip(lower=5)

# 3. Features (X) including the new 'hour' feature and Target (y) Selection
X = df[['temperature', 'humidity', 'wind_speed', 'pressure', 'hour']]
y = df['target_water_demand']

# 4. Split Data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- UPGRADED MODEL: RANDOM FOREST REGRESSOR ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Upgraded Model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"✔️ Random Forest Model Trained Successfully!")
print(f"📊 Upgraded Training Accuracy (R2): {round(train_score * 100, 2)}%")
print(f"📊 Upgraded Testing Accuracy (R2): {round(test_score * 100, 2)}%")

# 6. Generate Predictions & Save Assets
df['predicted_water_demand'] = model.predict(X)

# Return datetime back to string format for Streamlit rendering stability
df['datetime'] = df['datetime'].astype(str)

df.to_csv("live_weather_forecast.csv", index=False)
joblib.dump(model, "resource_demand_model.pkl")
print("💾 Upgraded model saved as 'resource_demand_model.pkl' with advanced time-features!")