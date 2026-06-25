import requests
import pandas as pd
from datetime import datetime

# API Configuration
API_KEY = "c82c0cf8271fad23cf9cd1dcedacbbc5"
CITY = "Algarve"  
URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

def fetch_weather_data():
    print(f"🔄 Fetching weather forecast data for {CITY}...")
    response = requests.get(URL)
    
    if response.status_code == 200:
        data = response.json()
        forecast_list = data['list']
        
        parsed_data = []
        for item in forecast_list:
            weather_record = {
                "datetime": item['dt_txt'],
                "temperature": item['main']['temp'],
                "humidity": item['main']['humidity'],
                "pressure": item['main']['pressure'],
                "wind_speed": item['wind']['speed'],
                "weather_description": item['weather'][0]['description']
            }
            parsed_data.append(weather_record)
        
        df = pd.DataFrame(parsed_data)
        print("✅ Data successfully fetched and structured!")
        return df
    else:
        print(f"❌ Failed to fetch data. Status Code: {response.status_code}")
        return None

if __name__ == "__main__":
    weather_df = fetch_weather_data()
    if weather_df is not None:
        print("\n--- Sample of Retrieved Forecast Data ---")
        print(weather_df.head())
        
        weather_df.to_csv("live_weather_forecast.csv", index=False)
        print("\n💾 Data saved successfully to: live_weather_forecast.csv")