# expense_prediction_api_with_lstm_max.py

import os
import json
import warnings
from datetime import datetime, timedelta
from functools import wraps
from typing import List, Dict, Any, Optional

from flask_cors import CORS
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# TensorFlow optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CONFIG = {
    "ARIMA_ORDERS": [(1,1,0),(0,1,1),(1,1,1)],
    "ML_LOOKBACK_STEPS": 6,
    "LSTM_LOOKBACK": 6,
    "LSTM_EPOCHS": 20,
    "LSTM_BATCH": 8
}

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

db = None
model_cache = {}

# ---------------- FIREBASE ----------------
def initialize_firebase():
    global db
    try:
        if firebase_admin._apps:
            db = firestore.client()
            return True

        firebase_key = os.environ.get("FIREBASE_KEY")
        if not firebase_key:
            print("⚠️ Running in MOCK mode")
            return False

        cred = credentials.Certificate(json.loads(firebase_key))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase connected")
        return True

    except Exception as e:
        print("Firebase Error:", e)
        return False

initialize_firebase()

def ensure_db():
    global db
    if db is None:
        return initialize_firebase()
    return True

# ---------------- AUTH ----------------
def verify_firebase_token(token):
    try:
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except:
        return None

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "No token"}), 401

        token = auth_header.split("Bearer ")[-1]
        uid = verify_firebase_token(token)

        if not uid:
            return jsonify({"error": "Invalid token"}), 401

        return f(uid, *args, **kwargs)
    return wrapper

# ---------------- DATA ----------------
def get_user_transactions(uid):
    if not ensure_db():
        np.random.seed(abs(hash(uid)) % 10000)
        data = []
        base = datetime.now()

        for i in range(24):
            date = (base - pd.DateOffset(months=i)).replace(day=1)
            amt = float(np.random.randint(3000,8000))
            data.append({"amount": amt, "date": date})

        return data

    try:
        ref = db.collection("users").document(uid).collection("transactions")
        docs = ref.stream()

        data = []
        for d in docs:
            x = d.to_dict()
            if "amount" in x and "date" in x:
                data.append({
                    "amount": float(x["amount"]),
                    "date": pd.to_datetime(x["date"])
                })

        return data

    except Exception as e:
        print("Fetch error:", e)
        return []

def prepare_monthly_data(tx):
    if not tx:
        return None

    df = pd.DataFrame(tx)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    monthly = df["amount"].resample("MS").sum()
    return monthly

# ---------------- MODELS ----------------
def arima_pred(series):
    try:
        model = ARIMA(series, order=(1,1,1)).fit()
        return float(model.forecast()[0])
    except:
        return None

def rf_pred(series):
    vals = series.values
    if len(vals) < 6:
        return None

    X, y = [], []
    for i in range(3, len(vals)):
        X.append(vals[i-3:i])
        y.append(vals[i])

    model = RandomForestRegressor()
    model.fit(X, y)

    return float(model.predict([vals[-3:]])[0])

def trend_pred(series):
    X = np.arange(len(series)).reshape(-1,1)
    y = series.values

    model = LinearRegression().fit(X,y)
    return float(model.predict([[len(series)]])[0])

def custom_pred(series):
    return float(series.tail(6).mean()*1.05)

def lstm_pred(series):
    if not TF_AVAILABLE or len(series)<6:
        return None

    try:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(series.values.reshape(-1,1))

        X, y = [], []
        for i in range(3,len(data)):
            X.append(data[i-3:i])
            y.append(data[i])

        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(32, input_shape=(X.shape[1],1)),
            Dense(1)
        ])
        model.compile("adam","mse")
        model.fit(X,y,epochs=10,verbose=0)

        pred = model.predict(data[-3:].reshape(1,3,1))
        return float(scaler.inverse_transform(pred)[0][0])

    except:
        return None

# ---------------- ENGINE ----------------
def predict_all(series):
    preds = {
        "arima": arima_pred(series),
        "rf": rf_pred(series),
        "trend": trend_pred(series),
        "custom": custom_pred(series),
        "lstm": lstm_pred(series)
    }

    valid = [v for v in preds.values() if v and v>0]

    return max(valid) if valid else None

# ---------------- SAVE ----------------
def save_prediction(uid, val):
    if not ensure_db():
        return False

    try:
        db.collection("users").document(uid)\
        .collection("prediction").document("next_month")\
        .set({
            "value": round(val,2),
            "created": datetime.now()
        })
        return True
    except:
        return False

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"msg":"API running"})

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "firebase": db is not None,
        "tf": TF_AVAILABLE
    })

@app.route("/predict", methods=["POST"])
@require_auth
def predict(uid):
    tx = get_user_transactions(uid)

    if not tx:
        return jsonify({"error":"no data"}),400

    series = prepare_monthly_data(tx)

    pred = predict_all(series)

    if not pred:
        return jsonify({"error":"prediction failed"}),400

    saved = save_prediction(uid,pred)

    return jsonify({
        "prediction": round(pred,2),
        "saved": saved
    })

# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":
    print("Running locally...")
    app.run(host="0.0.0.0", port=5000)