"""
app.py — Flask backend API for House Price Prediction.

Endpoints:
    GET  /                -> health check
    POST /predict         -> {"area": 2200, "rooms": 3, "age": 5} -> {"predicted_price": ...}
    GET  /model-info       -> basic info about the loaded model

Run with:
    python app.py
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "house_price_model.pkl")

app = Flask(__name__)
CORS(app)

# Load trained model once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "House Price Prediction API is running."})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "Linear Regression",
        "features": ["area", "rooms", "age"],
        "coefficients": dict(zip(["area", "rooms", "age"], model.coef_.tolist())),
        "intercept": model.intercept_,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        area = float(data["area"])
        rooms = float(data["rooms"])
        age = float(data["age"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide numeric 'area', 'rooms', and 'age' fields."}), 400

    input_df = pd.DataFrame([[area, rooms, age]], columns=["area", "rooms", "age"])
    prediction = float(model.predict(input_df)[0])

    return jsonify({
        "input": {"area": area, "rooms": rooms, "age": age},
        "predicted_price": round(prediction, 2)
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
