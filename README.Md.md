# 🏠 House Price Prediction

A Machine Learning project that predicts house prices using **Linear Regression** based on property characteristics such as **area, number of rooms, and property age**.

The project includes a complete ML workflow:

**Dataset → Model Training → Model Evaluation → Model Saving → Flask REST API → Streamlit Web App**

---

## 📌 Project Overview

House price prediction is a real-world regression problem in which property characteristics are used to estimate the market price of a house.

In this project, a **Linear Regression** model is trained using three input features:

* 📐 Area of the house (sq. ft.)
* 🚪 Number of rooms
* 🏠 Age of the property (years)

The trained model is saved using **Pickle** and can be used through either a Flask API or a Streamlit web interface.

---

## ✨ Features

* 📊 Exploratory Data Analysis
* 🤖 Linear Regression machine learning model
* 📈 Model evaluation using:

  * MAE
  * RMSE
  * R² Score
* 💾 Trained model saved as `.pkl`
* 🔌 Flask REST API
* 🖥️ Streamlit user interface
* 🔄 Two prediction modes:

  * Direct model prediction
  * Flask API-based prediction
* 🌐 CORS enabled for API access

The training script uses an 80/20 train-test split and evaluates the model using MAE, RMSE and R².

---

## 🛠️ Technologies Used

| Technology       | Purpose                  |
| ---------------- | ------------------------ |
| Python           | Programming Language     |
| Pandas           | Data Processing          |
| NumPy            | Numerical Operations     |
| Matplotlib       | Data Visualization       |
| Seaborn          | Data Visualization       |
| Scikit-learn     | Machine Learning         |
| Flask            | Backend REST API         |
| Flask-CORS       | Cross-Origin Requests    |
| Streamlit        | Frontend Web Application |
| Pickle           | Model Persistence        |
| Jupyter Notebook | Development / Analysis   |

The required package versions are provided in `requirements.txt`.

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   └── house_price.csv
│
├── model/
│   ├── train_model.py
│   └── house_price_model.pkl
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

### File Description

| File                          | Description                           |
| ----------------------------- | ------------------------------------- |
| `data/house_price.csv`        | House price dataset                   |
| `model/train_model.py`        | Trains and evaluates the ML model     |
| `model/house_price_model.pkl` | Saved trained Linear Regression model |
| `backend/app.py`              | Flask REST API                        |
| `frontend/streamlit_app.py`   | Streamlit web interface               |
| `requirements.txt`            | Python dependencies                   |
| `README.md`                   | Project documentation                 |

The project report describes the same separation between model, backend and frontend components.

---

## 📊 Dataset

The dataset contains **300 records** with the following columns:

| Feature | Description                                 |
| ------- | ------------------------------------------- |
| `area`  | Built-up area of the house in square feet   |
| `rooms` | Number of rooms                             |
| `age`   | Age of the property in years                |
| `price` | Market price of the house — target variable |

---

## 🧠 Machine Learning Model

The project uses **Linear Regression**.

### Input Features

```text
area
rooms
age
```

### Target

```text
price
```

The data is divided into:

`
