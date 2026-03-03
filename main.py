# expense_prediction_api_with_lstm_max.py
# Advanced Expense Prediction API (v2.4) - Max Prediction Merge
# Features:
#  - ARIMA, RandomForest, Trend, LSTM (optional), Custom Model
#  - Model caching per-user+data-hash
#  - Mock-data fallback when Firestore not configured
#  - Max value prediction across all models
#  - Endpoints: /, /transactions, /train, /predict, /health

import os
import json
import warnings
from datetime import datetime, timedelta
from functools import wraps
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

# Try importing TensorFlow / Keras. If unavailable, LSTM will be skipped gracefully.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings('ignore')

# -------------------------- Configuration --------------------------
CONFIG = {
    "ARIMA_ORDERS": [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2)],
    "ML_LOOKBACK_STEPS": 6,
    "LSTM_LOOKBACK": 6,
    "LSTM_EPOCHS": 30,
    "LSTM_BATCH": 8
}

# -------------------------- Flask + Globals --------------------------
app = Flask(__name__)
db: Optional[firestore.Client] = None
model_cache: Dict[str, Any] = {}

# -------------------------- Firebase helpers --------------------------
def initialize_firebase() -> bool:
    global db
    try:
        if firebase_admin._apps:
            # Already initialized — just make sure db client is set
            if db is None:
                db = firestore.client()
                print("✅ Firebase db client recovered from existing app")
            return True
        
        firebase_key = os.environ.get('FIREBASE_KEY')
        if not firebase_key:
            print("⚠️ FIREBASE_KEY not found. Running in mock-data mode.")
            return False
        service_account_info = json.loads(firebase_key)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Firebase init error — {type(e).__name__}: {e}")
        return False

def ensure_db():
    global db
    if db is None:
        initialize_firebase()
    return db is not None

def verify_firebase_token(token: str) -> Optional[str]:
    try:
        decoded = auth.verify_id_token(token)
        return decoded.get('uid')
    except Exception as e:
        print(f"Token verification error: {e}")
        return None

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'success': False, 'error': 'Authorization header required'}), 401
        token = auth_header.split('Bearer ')[1]
        uid = verify_firebase_token(token)
        if not uid:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        return f(uid, *args, **kwargs)
    return wrapper

# -------------------------- Data retrieval & preparation --------------------------
def get_user_transactions(uid: str) -> List[Dict[str, Any]]:
    global db
    if not ensure_db():  # deterministic mock based on uid
        np.random.seed(int(uid.__hash__() & 0xffffffff))
        mock = []
        base = datetime.now()
        for i in range(24):
            date = (base - pd.DateOffset(months=i)).replace(day=1)
            amount = float(max(500, np.random.randint(3000, 8000) + np.random.normal(0, 600)))
            mock.append({
                'amount': amount,
                'date': date,
                'category': np.random.choice(['Groceries', 'Transport', 'Entertainment', 'Utilities'])
            })
        print(f"🔧 Returning {len(mock)} mock transactions for user {uid}")
        return mock
    try:
        ref = db.collection('users').document(uid).collection('transactions')
        docs = ref.stream()
        transactions = []
        for d in docs:
            data = d.to_dict()
            if 'amount' in data and 'date' in data:
                date = data['date']
                if hasattr(date, 'seconds'):
                    date = datetime.fromtimestamp(date.seconds)
                elif isinstance(date, str):
                    date = pd.to_datetime(date)
                transactions.append({'amount': float(data['amount']), 'date': date, 'category': data.get('category', 'Other')})
        print(f"📊 Retrieved {len(transactions)} transactions for user {uid}")
        return transactions
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return []

def prepare_monthly_data(transactions: List[Dict[str, Any]]) -> Optional[pd.Series]:
    if not transactions:
        return None
    try:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        monthly = df['amount'].resample('MS').sum().sort_index()
        monthly = monthly[monthly > 0]
        if monthly.empty:
            return None
        return monthly
    except Exception as e:
        print(f"Error preparing monthly data: {e}")
        return None

# -------------------------- Prediction helpers --------------------------
def _best_arima_forecast(series: pd.Series) -> Optional[float]:
    if series is None or len(series) < 3:
        return None
    best_aic = float('inf')
    best_pred = None
    for order in CONFIG['ARIMA_ORDERS']:
        try:
            m = ARIMA(series, order=order).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_pred = float(m.forecast(steps=1).iloc[0])
        except Exception:
            continue
    return best_pred

# -------------------------- Prediction Engine --------------------------
class PredictionEngine:
    def __init__(self, monthly_data: pd.Series, user_uid: str):
        self.monthly_data = monthly_data
        self.user_uid = user_uid
        self.data_hash = int(pd.util.hash_pandas_object(monthly_data).sum())
        self.cache_key = f"{user_uid}_{self.data_hash}"

    def _cached(self, name: str):
        return model_cache.get(self.cache_key, {}).get(name)

    def _set_cache(self, name: str, value: Any):
        if self.cache_key not in model_cache:
            model_cache[self.cache_key] = {}
        model_cache[self.cache_key][name] = value

    def get_arima_prediction(self) -> Optional[float]:
        cached = self._cached('arima')
        if cached is not None:
            return cached
        pred = _best_arima_forecast(self.monthly_data)
        if pred is not None:
            self._set_cache('arima', pred)
        return pred

    def get_ml_prediction(self) -> Optional[float]:
        cached = self._cached('rf')
        values = self.monthly_data.values
        if len(values) < 6:
            return None
        n_steps = min(CONFIG['ML_LOOKBACK_STEPS'], len(values)-1)
        X, y = [], []
        for i in range(n_steps, len(values)):
            X.append(values[i-n_steps:i])
            y.append(values[i])
        if len(X) < 3:
            return None
        def fe(x): return np.concatenate([x, [x.mean()], [x.std()], [x[-1]-x[-2] if len(x)>1 else 0]])
        Xf = np.array([fe(xi) for xi in X])
        y = np.array(y)
        if cached is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(Xf, y)
            self._set_cache('rf', model)
        else:
            model = cached
        last = fe(values[-n_steps:])
        try:
            return float(model.predict([last])[0])
        except Exception:
            return None

    def get_trend_prediction(self) -> Optional[float]:
        cached = self._cached('trend')
        if cached is not None:
            model = cached
        else:
            X = np.arange(len(self.monthly_data)).reshape(-1,1)
            y = self.monthly_data.values
            model = LinearRegression().fit(X, y)
            self._set_cache('trend', model)
        next_idx = len(self.monthly_data)
        try:
            return float(model.predict([[next_idx]])[0])
        except Exception:
            return None

    def get_lstm_prediction(self) -> Optional[float]:
        if not TF_AVAILABLE or len(self.monthly_data) < 6:
            return None
        cached_model = self._cached('lstm')
        cached_scaler = self._cached('scaler')
        values = self.monthly_data.values.reshape(-1,1).astype('float32')
        if cached_model is None or cached_scaler is None:
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(values)
            n_steps = min(6, len(scaled)-1)
            X, y = [], []
            for i in range(n_steps, len(scaled)):
                X.append(scaled[i-n_steps:i,0])
                y.append(scaled[i,0])
            if len(X)<3:
                return None
            X = np.array(X).reshape((len(X), n_steps,1))
            y = np.array(y)
            try:
                tf.keras.backend.clear_session()
                model = Sequential()
                model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
                model.add(Dropout(0.2))
                model.add(Dense(16, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                es = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True, verbose=0)
                model.fit(X, y, epochs=CONFIG['LSTM_EPOCHS'], batch_size=CONFIG['LSTM_BATCH'], verbose=0, callbacks=[es])
                self._set_cache('lstm', model)
                self._set_cache('scaler', scaler)
                cached_model = model
                cached_scaler = scaler
            except Exception as e:
                print(f"⚠️ LSTM failed: {e}")
                return None
        try:
            scaled = cached_scaler.transform(values)
            n_steps = min(6, len(scaled)-1)
            last_seq = scaled[-n_steps:,0].reshape((1,n_steps,1))
            pred_scaled = cached_model.predict(last_seq, verbose=0)[0,0]
            pred = cached_scaler.inverse_transform(np.array(pred_scaled).reshape(-1,1))[0,0]
            return float(pred)
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}")
            return None

    # --- Custom model ---
    def get_custom_model_prediction(self) -> Optional[float]:
        if len(self.monthly_data) < 3:
            return None
        return float(self.monthly_data.tail(6).mean() * 1.05)  # example custom model

    # --- Max prediction across all models ---
    def get_max_prediction(self) -> Optional[float]:
        preds = {
            'arima': self.get_arima_prediction(),
            'ml': self.get_ml_prediction(),
            'lstm': self.get_lstm_prediction(),
            'trend': self.get_trend_prediction(),
            'custom': self.get_custom_model_prediction()
        }
        valid = [v for v in preds.values() if v is not None and v>0]
        if not valid:
            return None
        final = max(valid)
        print(f"🎯 Max prediction: {final} (from models: {list(preds.keys())})")
        return final

# -------------------------- Firestore saving --------------------------
def store_prediction_to_firestore(user_uid: str, predicted_expense: float) -> bool:
    global db
    if not db or predicted_expense <= 0:
        return False
    try:
        data = {
            'predicted_expense': round(predicted_expense, 2),
            'created_at': datetime.now(),
            'month': (datetime.now() + timedelta(days=30)).strftime('%B %Y')
        }
        ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        ref.set(data)
        print(f"💾 Stored prediction for {user_uid}: {predicted_expense:.2f}")
        return True
    except Exception as e:
        print(f"Error storing prediction: {e}")
        return False

# -------------------------- API routes --------------------------
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Advanced Expense Prediction API (v2.4 with Max Prediction)',
        'features': ['ARIMA', 'RandomForest', 'LSTM (optional)', 'Trend', 'Custom Model', 'Max Prediction', 'Model Caching']
    })

@app.route('/transactions', methods=['GET'])
@require_auth
def api_transactions(user_uid):
    transactions = get_user_transactions(user_uid)
    return jsonify({'success': True, 'count': len(transactions), 'transactions': transactions})

@app.route('/train', methods=['GET'])
@require_auth
def train_endpoint(user_uid):
    tx = get_user_transactions(user_uid)
    monthly = prepare_monthly_data(tx)
    if monthly is None or len(monthly) < 3:
        return jsonify({'success': False, 'error': 'Not enough data to train models.'}), 400
    engine = PredictionEngine(monthly, user_uid)
    _ = engine.get_arima_prediction()
    _ = engine.get_ml_prediction()
    _ = engine.get_trend_prediction()
    _ = engine.get_lstm_prediction()
    _ = engine.get_custom_model_prediction()
    return jsonify({'success': True, 'message': 'Models trained/cached for user', 'data_points': len(monthly)})

@app.route('/predict', methods=['POST'])
@require_auth
def predict_expense(user_uid):
    try:
        transactions = get_user_transactions(user_uid)
        if not transactions:
            return jsonify({'success': False, 'error': 'No transaction data found.'}), 400
        monthly = prepare_monthly_data(transactions)
        if monthly is None or monthly.empty:
            return jsonify({'success': False, 'error': 'Insufficient monthly data.'}), 400
        engine = PredictionEngine(monthly, user_uid)
        pred = engine.get_max_prediction()
        if pred is None or pred <= 0:
            return jsonify({'success': False, 'error': 'Could not produce a reliable prediction.'}), 400
        stored = store_prediction_to_firestore(user_uid, pred)
        return jsonify({'success': True, 'predicted_expense': round(pred,2), 'stored_to_firestore': stored, 'data_points_used': len(monthly)})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'firebase_connected': db is not None,
        'tensorflow_available': TF_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

# -------------------------- Main --------------------------
if __name__ == '__main__':
    print('Starting Advanced Expense Prediction API (v2.4 with Max Prediction)')
    initialize_firebase()
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=debug_mode)
