"""
train_model.py
Trains a Linear Regression model on the house price dataset and saves it to disk.

Run with:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "house_price.csv")
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def train():
    df = load_data()

    X = df[["area", "rooms", "age"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model training complete.")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R2   : {r2:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_PATH}")
    return model


def predict_price(model, area, rooms, age):
    input_df = pd.DataFrame([[area, rooms, age]], columns=["area", "rooms", "age"])
    return round(float(model.predict(input_df)[0]), 2)


if __name__ == "__main__":
    trained_model = train()
    sample = predict_price(trained_model, area=2200, rooms=3, age=5)
    print(f"\nSample prediction (2200 sqft, 3 rooms, 5 yrs old): Rs. {sample:,.2f}")
