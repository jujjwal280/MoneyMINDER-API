import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_caching import Cache
from marshmallow import Schema, fields, ValidationError

import firebase_admin
from firebase_admin import credentials, firestore, auth

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = {
    "DEBUG": os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": "model_cache_bin",
    "CACHE_DEFAULT_TIMEOUT": 86400,
}

app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)

# Global Firebase
db = None

# --- DATA VALIDATION SCHEMA ---
class PredictionRequestSchema(Schema):
    force_train = fields.Bool(load_default=False)

# --- FIREBASE INITIALIZATION ---
def initialize_firebase():
    global db
    try:
        if not firebase_admin._apps:
            firebase_key = os.environ.get('FIREBASE_KEY')
            if firebase_key:
                service_account_info = json.loads(firebase_key)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                logger.info("✅ Firebase initialized successfully")
                return True
        else:
            db = firestore.client()
            return True
    except Exception as e:
        logger.error(f"⚠️ Firebase error: {e}")
        return False
    return False

def verify_firebase_token(token: str) -> Optional[str]:
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'success': False, 'error': 'Authorization required'}), 401
        token = auth_header.split('Bearer ')[1]
        user_uid = verify_firebase_token(token)
        if not user_uid:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        return f(user_uid, *args, **kwargs)
    return decorated

# --- DATA FUNCTIONS ---
def get_user_transactions(uid: str) -> List[Dict[str, Any]]:
    global db
    if not db:
        initialize_firebase()
        if not db:
            return []
    try:
        transactions_ref = db.collection('users').document(uid).collection('transactions')
        docs = transactions_ref.stream()
        transactions = []
        for doc in docs:
            data = doc.to_dict()
            if 'amount' in data and 'date' in data:
                date = data['date']
                if hasattr(date, 'seconds'):
                    date = datetime.fromtimestamp(date.seconds)
                elif isinstance(date, str):
                    date = pd.to_datetime(date)
                transactions.append({
                    "id": doc.id,
                    "amount": float(data['amount']),
                    "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    "category": data.get('category', 'Other'),
                    "description": data.get('description', ''),
                    "day": data.get('day', ''),
                    "month": data.get('month', ''),
                    "week": data.get('week', '')
                })
        logger.info(f"📊 Found {len(transactions)} transactions for user {uid}")
        return transactions
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return []

def prepare_monthly_data(transactions: List[Dict[str, Any]]) -> Optional[pd.Series]:
    if not transactions:
        return None
    try:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        monthly_series = df['amount'].resample('ME').sum().sort_index()
        monthly_series = monthly_series[monthly_series > 0]
        return monthly_series
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None

def get_ensemble_prediction(monthly_data: pd.Series) -> Optional[float]:
    if monthly_data is None or len(monthly_data) == 0:
        return None
    
    predictions = []
    recent_avg = float(monthly_data.tail(3).mean()) if len(monthly_data) >= 3 else float(monthly_data.mean())
    
    # ARIMA
    try:
        if len(monthly_data) >= 4:
            model = ARIMA(monthly_data, order=(1,1,0))
            fitted = model.fit()
            predictions.append(float(fitted.forecast(steps=1).iloc[0]))
    except:
        pass
    
    # Random Forest
    try:
        if len(monthly_data) >= 6:
            values = monthly_data.values
            X, y = [], []
            for i in range(3, len(values)):
                X.append(values[i-3:i])
                y.append(values[i])
            if len(X) >= 3:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(np.array(X), np.array(y))
                predictions.append(float(model.predict([values[-3:]])[0]))
    except:
        pass
    
    # Trend
    try:
        if len(monthly_data) >= 3:
            X = np.arange(len(monthly_data)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, monthly_data.values)
            predictions.append(float(model.predict([[len(monthly_data)]])[0]))
    except:
        pass
    
    if predictions:
        ensemble_pred = (sum(predictions) + recent_avg * 2) / (len(predictions) + 2)
    else:
        ensemble_pred = recent_avg
    
    ensemble_pred = max(recent_avg * 0.5, min(recent_avg * 2.5, ensemble_pred))
    return float(ensemble_pred)

def store_prediction_to_firestore(user_uid: str, predicted_expense: float) -> bool:
    global db
    if not db or predicted_expense <= 0:
        return False
    try:
        prediction_data = {
            'predicted_expense': round(predicted_expense, 2),
            'created_at': firestore.SERVER_TIMESTAMP,
            'month': (datetime.now() + timedelta(days=30)).strftime('%B %Y'),
        }
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        prediction_ref.set(prediction_data)
        logger.info(f"✅ Stored prediction {predicted_expense:.2f} for user {user_uid}")
        return True
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")
        return False

# --- API ROUTES ---
@app.route('/', methods=['GET', 'POST'])
@require_auth
def home(user_uid):
    if request.method == 'POST':
        return predict(user_uid)
    return jsonify({"message": "Expense Prediction API", "user": user_uid, "status": "ready"})

@app.route('/transactions', methods=['GET'])
@require_auth
def get_transactions_route(user_uid):
    transactions = get_user_transactions(user_uid)
    return jsonify({"success": True, "count": len(transactions), "transactions": transactions})

@app.route('/predictions', methods=['GET'])
@require_auth
def get_predictions_route(user_uid):
    global db
    if not db:
        initialize_firebase()
    try:
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        doc = prediction_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return jsonify({"success": True, "predicted_expense": data.get('predicted_expense'), "has_prediction": True})
        return jsonify({"success": True, "predicted_expense": None, "has_prediction": False})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
@require_auth
def predict(user_uid):
    try:
        logger.info(f"🔥 PREDICTION REQUEST for user: {user_uid}")
        transactions = get_user_transactions(user_uid)
        if not transactions:
            return jsonify({"success": False, "error": "No transactions found"}), 400
        
        monthly_data = prepare_monthly_data(transactions)
        if monthly_data is None or len(monthly_data) == 0:
            return jsonify({"success": False, "error": "Insufficient data"}), 400
        
        predicted_expense = get_ensemble_prediction(monthly_data)
        if predicted_expense is None or predicted_expense <= 0:
            return jsonify({"success": False, "error": "Prediction failed"}), 400
        
        stored = store_prediction_to_firestore(user_uid, predicted_expense)
        
        return jsonify({
            "success": True,
            "predicted_expense": round(predicted_expense, 2),
            "stored_to_firestore": stored,
            "data_points": len(monthly_data)
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "firebase_connected": db is not None, "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    logger.info("🚀 Starting Expense Prediction API...")
    initialize_firebase()
    app.run(host='0.0.0.0', port=5000)
