from flask import Flask, request, jsonify
from predictor import predict_price_direction
from datetime import datetime

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    target_price = float(data.get("target_price"))
    target_time_str = data.get("target_time")  # Format: YYYY-MM-DD HH:MM

    try:
        target_time = datetime.strptime(target_time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return jsonify({"error": "Invalid time format. Use YYYY-MM-DD HH:MM"}), 400

    result = predict_price_direction(target_price, target_time)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
