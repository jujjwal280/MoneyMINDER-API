from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
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

# Global variables
db = None

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
                print("✅ Firebase initialized successfully")
                return True
            else:
                print("⚠️ FIREBASE_KEY not found")
                return False
        else:
            db = firestore.client()
            return True
    except Exception as e:
        print(f"⚠️ Firebase error: {e}")
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
            return jsonify({'success': False, 'error': 'Authorization required'}), 401
        
        token = auth_header.split('Bearer ')[1]
        user_uid = verify_firebase_token(token)
        if not user_uid:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        
        return f(user_uid, *args, **kwargs)
    return decorated_function

def get_user_transactions(uid: str) -> List[Dict[str, Any]]:
    """Get user transactions from Firestore"""
    global db
    if not db:
        if not initialize_firebase():
            return []
        if not db:
            return []
    
    try:
        transactions_ref = db.collection('users').document(uid).collection('transactions')
        docs = transactions_ref.stream()
        
        transactions = []
        for doc in docs:
            data = doc.to_dict()
            if 'amount' in data and 'date' in data:
                # Handle Firestore timestamp
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
        
        print(f"📊 Found {len(transactions)} transactions for user {uid}")
        return transactions
    except Exception as e:
        print(f"❌ Error getting transactions: {e}")
        return []

def prepare_monthly_data(transactions: List[Dict[str, Any]]) -> Optional[pd.Series]:
    """Convert transactions to monthly spending totals"""
    if not transactions:
        return None
    
    try:
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Group by month and sum amounts
        monthly_series = df['amount'].resample('ME').sum().sort_index()
        monthly_series = monthly_series[monthly_series > 0]
        
        print(f"📈 Prepared {len(monthly_series)} months of data")
        return monthly_series
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return None

def get_ensemble_prediction(monthly_data: pd.Series) -> Optional[float]:
    """Generate prediction using multiple models"""
    if monthly_data is None or len(monthly_data) == 0:
        return None
    
    predictions = []
    recent_avg = float(monthly_data.tail(3).mean()) if len(monthly_data) >= 3 else float(monthly_data.mean())
    
    # ARIMA prediction
    try:
        if len(monthly_data) >= 4:
            model = ARIMA(monthly_data, order=(1,1,0))
            fitted = model.fit()
            arima_pred = float(fitted.forecast(steps=1).iloc[0])
            predictions.append(arima_pred)
            print(f"🔮 ARIMA: {arima_pred}")
    except:
        pass
    
    # Random Forest prediction  
    try:
        if len(monthly_data) >= 6:
            values = monthly_data.values
            X, y = [], []
            for i in range(3, len(values)):
                X.append(values[i-3:i])
                y.append(values[i])
            
            if len(X) >= 3:
                X = np.array(X)
                y = np.array(y)
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X, y)
                last_seq = values[-3:]
                rf_pred = model.predict([last_seq])[0]
                predictions.append(float(rf_pred))
                print(f"🤖 Random Forest: {rf_pred}")
    except:
        pass
    
    # Trend prediction
    try:
        if len(monthly_data) >= 3:
            X = np.arange(len(monthly_data)).reshape(-1, 1)
            y = monthly_data.values
            model = LinearRegression()
            model.fit(X, y)
            trend_pred = model.predict([[len(monthly_data)]])[0]
            predictions.append(float(trend_pred))
            print(f"📊 Trend: {trend_pred}")
    except:
        pass
    
    # Calculate final prediction
    if predictions:
        # Weight recent average and model predictions
        ensemble_pred = (sum(predictions) + recent_avg * 2) / (len(predictions) + 2)
    else:
        ensemble_pred = recent_avg
    
    # Apply safety bounds
    max_bound = recent_avg * 2.5
    min_bound = recent_avg * 0.5
    ensemble_pred = max(min_bound, min(max_bound, ensemble_pred))
    
    print(f"🎯 Final prediction: {ensemble_pred}")
    return float(ensemble_pred)

def store_prediction_to_firestore(user_uid: str, predicted_expense: float) -> bool:
    """Store prediction to Firestore - CRITICAL FUNCTION"""
    global db
    
    if not db:
        print("❌ No Firebase connection for storing prediction")
        return False
    
    if predicted_expense <= 0:
        print("❌ Invalid prediction amount")
        return False
    
    try:
        # Create prediction data
        prediction_data = {
            'predicted_expense': round(predicted_expense, 2),
            'created_at': firestore.SERVER_TIMESTAMP,
            'month': (datetime.now() + timedelta(days=30)).strftime('%B %Y'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store to Firestore
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        prediction_ref.set(prediction_data)
        
        print(f"✅ STORED prediction {predicted_expense:.2f} for user {user_uid}")
        
        # Verify it was stored
        stored_doc = prediction_ref.get()
        if stored_doc.exists:
            print(f"✅ VERIFIED: Prediction stored successfully")
            return True
        else:
            print(f"❌ FAILED: Could not verify storage")
            return False
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR storing prediction: {e}")
        return False

# API Routes

@app.route('/', methods=['GET', 'POST'])
@require_auth
def home(user_uid):
    """Handle both GET and POST requests"""
    if request.method == 'POST':
        return predict_expense(user_uid)
    
    return jsonify({
        "message": "🚀 Expense Prediction API - WORKING VERSION",
        "user": user_uid,
        "status": "ready",
        "endpoints": [
            "GET /transactions - Get user transactions",
            "POST /predict - Generate prediction", 
            "GET /predictions - Get stored predictions",
            "GET /health - Health check"
        ]
    })

@app.route('/transactions', methods=['GET'])
@require_auth  
def get_transactions(user_uid):
    """Get user's transactions - FOR FLUTTER APP"""
    try:
        print(f"🔍 FETCHING transactions for user: {user_uid}")
        
        # Ensure Firebase is connected
        if not db:
            if not initialize_firebase():
                return jsonify({
                    "success": False,
                    "error": "Firebase connection failed"
                }), 503
        
        # Get user transactions
        transactions = get_user_transactions(user_uid)
        
        print(f"📊 Returning {len(transactions)} transactions")
        
        return jsonify({
            "success": True,
            "user_uid": user_uid,
            "count": len(transactions),
            "transactions": transactions
        })
        
    except Exception as e:
        print(f"❌ Error fetching transactions for {user_uid}: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to fetch transactions: {str(e)}"
        }), 500

@app.route('/predictions', methods=['GET'])
@require_auth
def get_predictions(user_uid):
    """Get user's stored predictions - FOR FLUTTER APP"""
    try:
        print(f"🔍 FETCHING predictions for user: {user_uid}")
        
        if not db:
            if not initialize_firebase():
                return jsonify({
                    "success": False,
                    "error": "Firebase connection failed"
                }), 503
        
        # Get user's prediction
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        doc = prediction_ref.get()
        
        if doc.exists:
            prediction_data = doc.to_dict()
            predicted_expense = prediction_data.get('predicted_expense', 0)
            
            print(f"📊 Found prediction: {predicted_expense}")
            
            return jsonify({
                "success": True,
                "user_uid": user_uid,
                "predicted_expense": predicted_expense,
                "month": prediction_data.get('month', ''),
                "created_at": prediction_data.get('timestamp', ''),
                "has_prediction": True
            })
        else:
            print(f"❌ No prediction found for user {user_uid}")
            return jsonify({
                "success": True,
                "user_uid": user_uid,
                "predicted_expense": None,
                "has_prediction": False,
                "message": "No predictions found for this user"
            })
            
    except Exception as e:
        print(f"❌ Error fetching predictions for {user_uid}: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to fetch predictions: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
@require_auth
def predict_expense(user_uid):
    """Generate prediction and store to Firestore - FOR FLUTTER APP"""
    try:
        print(f"🔥 PREDICTION REQUEST for user: {user_uid}")
        
        # Ensure Firebase is connected
        if not db:
            if not initialize_firebase():
                return jsonify({
                    "success": False,
                    "error": "Firebase connection failed"
                }), 503
        
        # Get user transactions
        transactions = get_user_transactions(user_uid)
        if not transactions:
            print(f"❌ No transactions found for user {user_uid}")
            return jsonify({
                "success": False,
                "error": "No transaction data found. Please add expenses first."
            }), 400
        
        # Prepare data
        monthly_data = prepare_monthly_data(transactions)
        if monthly_data is None or len(monthly_data) == 0:
            print(f"❌ No monthly data for user {user_uid}")
            return jsonify({
                "success": False,
                "error": "Insufficient data for prediction"
            }), 400
        
        # Generate prediction
        predicted_expense = get_ensemble_prediction(monthly_data)
        if predicted_expense is None or predicted_expense <= 0:
            print(f"❌ Prediction failed for user {user_uid}")
            return jsonify({
                "success": False,
                "error": "Could not generate prediction"
            }), 400
        
        # **CRITICAL: Store to Firestore**
        print(f"🔥 ATTEMPTING TO STORE prediction {predicted_expense:.2f}")
        stored = store_prediction_to_firestore(user_uid, predicted_expense)
        
        if not stored:
            print(f"❌ STORAGE FAILED for user {user_uid}")
            return jsonify({
                "success": False,
                "error": "Failed to store prediction"
            }), 500
        
        print(f"✅ SUCCESS: Prediction stored for user {user_uid}")
        
        return jsonify({
            "success": True,
            "message": "Prediction generated and stored successfully",
            "user_uid": user_uid,
            "predicted_expense": round(predicted_expense, 2),
            "stored_to_firestore": stored,
            "data_points": len(monthly_data)
        })
        
    except Exception as e:
        print(f"🔥 CRITICAL ERROR for user {user_uid}: {e}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "firebase_connected": db is not None,
        "timestamp": datetime.now().isoformat(),
        "endpoints_available": [
            "GET /transactions",
            "POST /predict", 
            "GET /predictions",
            "GET /health"
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting WORKING Expense Prediction API...")
    initialize_firebase()
    app.run(host='0.0.0.0', port=5000, debug=True)