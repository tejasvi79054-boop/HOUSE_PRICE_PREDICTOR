"""
streamlit_app.py — Frontend for House Price Prediction.

Can run in two modes:
1. Standalone: loads the .pkl model directly (no backend needed).
2. API mode: calls the Flask backend's /predict endpoint (set USE_API = True).

Run with:
    streamlit run streamlit_app.py
"""

import os
import pickle
import requests
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "house_price_model.pkl")

USE_API = False  # set True if the Flask backend (app.py) is running on localhost:5000
API_URL = "http://localhost:5000/predict"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction")
st.write("Enter the property details below to get an estimated price.")

col1, col2, col3 = st.columns(3)
with col1:
    area = st.number_input("Area (sq. ft.)", min_value=200, max_value=10000, value=2000, step=50)
with col2:
    rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3, step=1)
with col3:
    age = st.number_input("Age of property (years)", min_value=0, max_value=100, value=5, step=1)

if st.button("Predict Price", type="primary"):
    if USE_API:
        try:
            response = requests.post(API_URL, json={"area": area, "rooms": rooms, "age": age}, timeout=5)
            response.raise_for_status()
            result = response.json()
            price = result["predicted_price"]
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend API: {e}")
            price = None
    else:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        input_df = pd.DataFrame([[area, rooms, age]], columns=["area", "rooms", "age"])
        price = round(float(model.predict(input_df)[0]), 2)

    if price is not None:
        st.success(f"### Estimated Price: ₹ {price:,.2f}")

st.divider()
st.caption("Model: Linear Regression | Features: area, rooms, age")
