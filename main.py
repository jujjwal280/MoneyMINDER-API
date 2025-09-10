# ==============================================================================
#  Personal Finance Prediction API with Advanced ML Models
#  Enhanced Ensemble Prediction System
# ==============================================================================

from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import json
import os
from typing import List, Dict, Any, Optional
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for Firebase and ML models
db = None
ensemble_models = {}

def initialize_firebase():
    """Initialize Firebase Admin SDK using secret key"""
    global db
    try:
        if not firebase_admin._apps:
            firebase_key = os.environ.get('FIREBASE_KEY')
            if firebase_key:
                service_account_info = json.loads(firebase_key)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                print("✅ Firebase initialized successfully with secret key")
                return True
            else:
                print("⚠️ FIREBASE_KEY secret not found. Using development mode.")
                return False
        else:
            db = firestore.client()
            print("✅ Firebase already initialized")
            return True
    except Exception as e:
        print(f"⚠️ Error initializing Firebase: {e}")
        return False

def verify_firebase_token(token: str) -> Optional[str]:
    """Verify Firebase ID token and return user UID"""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        print(f"Token verification error: {e}")
        return None

def require_auth(f):
    """Decorator to require Firebase authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'success': False, 'error': 'Authorization header required'}), 401
        
        token = auth_header.split('Bearer ')[1]
        user_uid = verify_firebase_token(token)
        if not user_uid:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        
        return f(user_uid, *args, **kwargs)
    return decorated_function

def get_user_transactions(uid: str) -> List[Dict[str, Any]]:
    """Fetch all transactions for a specific user"""
    global db
    if not db:
        if not initialize_firebase():
            # Generate realistic mock data for development
            print(f"🔧 Using mock data for user {uid}")
            mock_data = []
            base_date = datetime.now()
            for i in range(24):  # 24 months of data
                date = base_date - timedelta(days=30 * i)
                # Simulate realistic spending patterns
                amount = np.random.randint(3000, 8000) + np.random.normal(0, 500)
                amount = max(1000, amount)  # Ensure minimum spending
                mock_data.append({
                    "amount": float(amount),
                    "date": date,
                    "category": np.random.choice(["Groceries", "Transport", "Entertainment", "Utilities"])
                })
            return mock_data
        if not db:
            return []
    
    try:
        transactions_ref = db.collection('users').document(uid).collection('transactions')
        docs = transactions_ref.stream()
        
        transactions = []
        for doc in docs:
            transaction_data = doc.to_dict()
            if 'amount' in transaction_data and 'date' in transaction_data:
                # Convert Firestore timestamp to datetime if needed
                date = transaction_data['date']
                if hasattr(date, 'seconds'):  # Firestore timestamp
                    date = datetime.fromtimestamp(date.seconds)
                elif isinstance(date, str):
                    date = pd.to_datetime(date)
                
                transactions.append({
                    "amount": float(transaction_data['amount']),
                    "date": date,
                    "category": transaction_data.get('category', 'Other')
                })
        
        print(f"📊 Retrieved {len(transactions)} transactions for user {uid}")
        return transactions
    except Exception as e:
        print(f"Error retrieving transactions: {e}")
        return []

def prepare_monthly_data(transactions: List[Dict[str, Any]]) -> Optional[pd.Series]:
    """Aggregate transactions into monthly totals for time-series forecasting"""
    if not transactions:
        return None
    
    try:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Resample to monthly totals and sort chronologically
        monthly_series = df['amount'].resample('ME').sum().sort_index()
        
        # Remove months with zero spending (incomplete data)
        monthly_series = monthly_series[monthly_series > 0]
        
        print(f"📈 Prepared {len(monthly_series)} months of data")
        return monthly_series
    except Exception as e:
        print(f"Error preparing monthly data: {e}")
        return None

def get_arima_prediction(monthly_data: pd.Series) -> Optional[float]:
    """ARIMA statistical model for time series prediction"""
    if monthly_data is None or len(monthly_data) < 3:
        return None
    
    try:
        # Try different ARIMA orders and pick the best one
        best_aic = float('inf')
        best_prediction = None
        
        orders = [(1,1,0), (0,1,1), (1,1,1), (2,1,0), (0,1,2)]
        
        for order in orders:
            try:
                model = ARIMA(monthly_data, order=order)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_prediction = float(fitted_model.forecast(steps=1).iloc[0])
            except:
                continue
        
        print(f"🔮 ARIMA prediction: {best_prediction}")
        return best_prediction
    except Exception as e:
        print(f"ARIMA error: {e}")
        return None

def get_ml_prediction(monthly_data: pd.Series) -> Optional[float]:
    """Enhanced ML prediction using Random Forest"""
    if monthly_data is None or len(monthly_data) < 6:
        return None
    
    try:
        # Create features from time series
        values = monthly_data.values
        n_steps = min(6, len(values) - 1)
        
        X, y = [], []
        for i in range(n_steps, len(values)):
            X.append(values[i-n_steps:i])
            y.append(values[i])
        
        if len(X) < 3:
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        # Add additional features
        X_enhanced = []
        for i in range(len(X)):
            features = list(X[i])  # Historical values
            features.append(float(np.mean(X[i])))  # Mean of recent values
            features.append(float(np.std(X[i])))   # Standard deviation
            features.append(X[i][-1] - X[i][-2] if len(X[i]) > 1 else 0)  # Trend
            X_enhanced.append(features)
        
        X_enhanced = np.array(X_enhanced)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_enhanced, y)
        
        # Make prediction
        last_sequence = values[-n_steps:]
        last_features = list(last_sequence)
        last_features.append(float(np.mean(last_sequence)))
        last_features.append(float(np.std(last_sequence)))
        last_features.append(last_sequence[-1] - last_sequence[-2] if len(last_sequence) > 1 else 0)
        
        prediction = model.predict([last_features])[0]
        print(f"🤖 ML prediction: {prediction}")
        return float(prediction)
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None

def get_trend_prediction(monthly_data: pd.Series) -> Optional[float]:
    """Simple trend-based prediction"""
    if monthly_data is None or len(monthly_data) < 3:
        return None
    
    try:
        # Use linear regression on time series
        X = np.arange(len(monthly_data)).reshape(-1, 1)
        y = monthly_data.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next month
        next_month = len(monthly_data)
        prediction = model.predict([[next_month]])[0]
        
        print(f"📊 Trend prediction: {prediction}")
        return float(prediction)
    except Exception as e:
        print(f"Trend prediction error: {e}")
        return None

def get_ensemble_prediction(monthly_data: pd.Series) -> Optional[float]:
    """Advanced ensemble prediction combining multiple models"""
    if monthly_data is None or len(monthly_data) == 0:
        return None
    
    predictions = {}
    weights = {}
    
    # Get predictions from different models
    arima_pred = get_arima_prediction(monthly_data)
    ml_pred = get_ml_prediction(monthly_data)
    trend_pred = get_trend_prediction(monthly_data)
    
    # Calculate recent average as baseline
    recent_avg = float(monthly_data.tail(3).mean()) if len(monthly_data) >= 3 else float(monthly_data.mean())
    
    # Assign weights based on data availability and model reliability
    if len(monthly_data) >= 12:  # Good amount of data
        if arima_pred is not None:
            predictions['arima'] = arima_pred
            weights['arima'] = 0.4
        if ml_pred is not None:
            predictions['ml'] = ml_pred
            weights['ml'] = 0.4
        if trend_pred is not None:
            predictions['trend'] = trend_pred
            weights['trend'] = 0.15
        predictions['avg'] = recent_avg
        weights['avg'] = 0.05
    elif len(monthly_data) >= 6:  # Moderate data
        if arima_pred is not None:
            predictions['arima'] = arima_pred
            weights['arima'] = 0.5
        if trend_pred is not None:
            predictions['trend'] = trend_pred
            weights['trend'] = 0.3
        predictions['avg'] = recent_avg
        weights['avg'] = 0.2
    else:  # Limited data
        if trend_pred is not None:
            predictions['trend'] = trend_pred
            weights['trend'] = 0.7
        predictions['avg'] = recent_avg
        weights['avg'] = 0.3
    
    # Calculate weighted ensemble prediction
    if not predictions:
        return recent_avg
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        return recent_avg
    
    ensemble_pred = sum(predictions[model] * weights[model] for model in predictions) / total_weight
    
    # Apply safety bounds
    max_reasonable = recent_avg * 3.0
    min_reasonable = recent_avg * 0.3
    ensemble_pred = max(min_reasonable, min(max_reasonable, ensemble_pred))
    
    print(f"🎯 Ensemble prediction: {ensemble_pred} (from {list(predictions.keys())})")
    return float(ensemble_pred)

def store_prediction_to_firestore(user_uid: str, predicted_expense: float) -> bool:
    """Store user's prediction to Firestore"""
    global db
    
    if not db or predicted_expense <= 0:
        return False
    
    try:
        prediction_data = {
            'predicted_expense': round(predicted_expense, 2),
            'created_at': datetime.now(),
            'month': (datetime.now() + timedelta(days=30)).strftime('%B %Y')
        }
        
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        prediction_ref.set(prediction_data)
        
        print(f"💾 Stored prediction {predicted_expense} for user {user_uid}")
        return True
    except Exception as e:
        print(f"Error storing prediction: {e}")
        return False

# API Routes

@app.route('/', methods=['GET', 'POST'])
@require_auth
def home(user_uid):
    """Home endpoint that handles both GET and POST requests"""
    if request.method == 'POST':
        return predict_expense(user_uid)
    
    return jsonify({
        "message": "🚀 Advanced Expense Prediction API",
        "version": "2.0",
        "user": user_uid,
        "features": [
            "ARIMA statistical forecasting",
            "Random Forest ML prediction", 
            "Trend analysis",
            "Ensemble model combining all approaches"
        ],
        "endpoints": [
            "GET / - API information",
            "POST / or POST /predict - Generate prediction",
            "GET /health - Health check"
        ]
    })

@app.route('/predict', methods=['POST'])
@require_auth
def predict_expense(user_uid):
    """Generate advanced expense prediction using ensemble of models"""
    try:
        if not db and not initialize_firebase():
            print("🔧 Running in development mode")
        
        # Get user's transaction data
        transactions = get_user_transactions(user_uid)
        if not transactions:
            return jsonify({
                "success": False,
                "error": "No transaction data found. Please add some expenses first."
            }), 400
        
        # Prepare monthly aggregated data
        monthly_data = prepare_monthly_data(transactions)
        if monthly_data is None or len(monthly_data) == 0:
            return jsonify({
                "success": False,
                "error": "Insufficient data for prediction."
            }), 400
        
        # Generate ensemble prediction
        predicted_expense = get_ensemble_prediction(monthly_data)
        
        if predicted_expense is None or predicted_expense <= 0:
            return jsonify({
                "success": False,
                "error": "Could not generate a reliable prediction."
            }), 400
        
        # Store prediction to Firestore
        stored = store_prediction_to_firestore(user_uid, predicted_expense)
        
        return jsonify({
            "success": True,
            "message": "Advanced prediction generated successfully",
            "user_uid": user_uid,
            "predicted_expense": round(predicted_expense, 2),
            "stored_to_firestore": stored,
            "data_points": len(monthly_data)
        })
        
    except Exception as e:
        print(f"🔥 Prediction error for user {user_uid}: {e}")
        return jsonify({
            "success": False,
            "error": "An internal server error occurred."
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    firebase_status = db is not None
    
    return jsonify({
        "status": "healthy",
        "firebase_connected": firebase_status,
        "models_available": ["ARIMA", "RandomForest", "Trend", "Ensemble"],
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Advanced Expense Prediction API...")
    initialize_firebase()
    app.run(host='0.0.0.0', port=5000, debug=True)