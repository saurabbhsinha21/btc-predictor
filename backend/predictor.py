import requests
from datetime import datetime, timedelta
import time

def fetch_current_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url).json()
    return float(response['price'])

def predict_price_direction(target_price, target_time):
    current_price = fetch_current_price()
    current_time = datetime.utcnow()

    # If target time is in the past, return N/A
    if target_time <= current_time:
        return {"prediction": "Invalid Target Time", "confidence": 0}

    # Placeholder logic: assume price movement directionally follows momentum
    # Can be replaced by ML later
    time_left = (target_time - current_time).total_seconds()
    predicted_price = current_price * (1 + 0.0001 * (time_left // 60))  # Simple extrapolation

    prediction = "Above" if predicted_price > target_price else "Below"
    confidence = round(abs(predicted_price - target_price) / target_price * 100, 2)

    return {
        "current_price": current_price,
        "predicted_price": round(predicted_price, 2),
        "target_price": target_price,
        "target_time": target_time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": prediction,
        "confidence": confidence
    }
