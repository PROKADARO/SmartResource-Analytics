import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

print("🤖 Starting Machine Learning Pipeline...")

# 1. Load the structured environmental data
try:
    df = pd.read_csv("live_weather_forecast.csv")
except FileNotFoundError:
    print("❌ Error: live_weather_forecast.csv not found! Run data_collection.py first.")
    exit()

# 2. Engineering Features: Simulate target variable (Smart Resource Demand in m3)
# Based on environmental physics: Higher temperature & lower humidity = More irrigation needed
np.random.seed(42)
df['target_water_demand'] = (df['temperature'] * 1.5) - (df['humidity'] * 0.2) + (df['wind_speed'] * 0.5) + np.random.normal(10, 2, len(df))
df['target_water_demand'] = df['target_water_demand'].clip(lower=5)  # Ensure no negative water consumption

# 3. Features (X) and Target (y) Selection
X = df[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = df['target_water_demand']

# 4. Split Data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate the Model Stability
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"✔️ Model Trained Successfully!")
print(f"📊 Training Accuracy (R2): {round(train_score * 100, 2)}%")
print(f"📊 Testing Accuracy (R2): {round(test_score * 100, 2)}%")

# 7. Generate Predictions for the upcoming days
df['predicted_water_demand'] = model.predict(X)

# Save the updated data with predictions and the trained model
df.to_csv("live_weather_forecast.csv", index=False)
joblib.dump(model, "resource_demand_model.pkl")
print("💾 Model saved as 'resource_demand_model.pkl' and predictions appended to CSV!")