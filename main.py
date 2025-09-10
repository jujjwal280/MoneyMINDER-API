from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any
from functools import wraps

app = Flask(__name__)

# Global variables for Firebase and ML model
db = None
expense_model = None
category_encoder = None

def initialize_firebase():
    """Initialize Firebase Admin SDK using secret key"""
    global db
    try:
        if not firebase_admin._apps:
            # Get Firebase credentials from environment variable
            firebase_key = os.environ.get('FIREBASE_KEY')
            if firebase_key:
                # Parse the JSON string from the secret
                service_account_info = json.loads(firebase_key)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                print("Firebase initialized successfully with secret key")
                return True
            else:
                print("Warning: FIREBASE_KEY secret not found. Firebase features will be unavailable.")
                print("Please set your FIREBASE_KEY secret to enable Firestore integration.")
                return False
        else:
            db = firestore.client()
            print("Firebase already initialized")
            return True
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        print("Firebase features will be unavailable until credentials are provided.")
        return False

def verify_firebase_token(token: str) -> str:
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
        
        # Pass user_uid to the function
        return f(user_uid, *args, **kwargs)
    return decorated_function

def get_transactions_data(user_uid: str = None) -> List[Dict[str, Any]]:
    """Retrieve transactions from Firestore for a specific user or all users"""
    global db
    if not db:
        if not initialize_firebase():
            return []
    
    try:
        if user_uid:
            # Get user-specific transactions
            transactions_ref = db.collection('users').document(user_uid).collection('transactions')
        else:
            # Get all transactions (fallback for training)
            transactions_ref = db.collection('transactions')
        
        docs = transactions_ref.stream()
        
        transactions = []
        for doc in docs:
            transaction_data = doc.to_dict()
            transaction_data['id'] = doc.id
            transactions.append(transaction_data)
        
        return transactions
    except Exception as e:
        print(f"Error retrieving transactions: {e}")
        return []

def prepare_data_for_ml(transactions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepare transaction data for machine learning"""
    if not transactions:
        return pd.DataFrame()
    
    df = pd.DataFrame(transactions)
    
    # Convert date fields
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month_num'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    
    # Convert amount to float
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Fill missing values
    df = df.fillna(0)
    
    return df

def train_expense_model(df: pd.DataFrame):
    """Train machine learning model for expense prediction"""
    global expense_model, category_encoder
    
    if df.empty or len(df) < 5:
        print("Insufficient data for training model")
        return False
    
    try:
        # Prepare features
        feature_columns = ['day_of_year', 'month_num', 'day_of_week']
        
        # Encode categories
        if 'category' in df.columns:
            category_encoder = LabelEncoder()
            df['category_encoded'] = category_encoder.fit_transform(df['category'].astype(str))
            feature_columns.append('category_encoded')
        
        # Prepare training data
        X = df[feature_columns].fillna(0)
        y = df['amount']
        
        # Train model
        expense_model = LinearRegression()
        expense_model.fit(X, y)
        
        print("Expense prediction model trained successfully")
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def predict_next_month_expense(user_uid: str) -> float:
    """Predict next month's total expense for a specific user"""
    global expense_model, category_encoder
    
    # Get user's transaction data
    transactions = get_transactions_data(user_uid)
    if not transactions:
        return 0.0
    
    df = prepare_data_for_ml(transactions)
    if df.empty or len(df) < 3:
        return 0.0
    
    # Train model with user's data
    train_success = train_expense_model(df)
    if not train_success or not expense_model:
        return 0.0
    
    try:
        # Calculate next month's prediction
        next_month = datetime.now() + timedelta(days=30)
        total_prediction = 0.0
        
        # Get user's common categories
        if 'category' in df.columns:
            common_categories = df['category'].value_counts().head(5).index.tolist()
        else:
            common_categories = ['Groceries', 'Transportation', 'Entertainment']
        
        for category in common_categories:
            # Prepare features for prediction
            features = {
                'day_of_year': next_month.timetuple().tm_yday,
                'month_num': next_month.month,
                'day_of_week': next_month.weekday()
            }
            
            if category_encoder:
                try:
                    category_encoded = category_encoder.transform([category])[0]
                    features['category_encoded'] = category_encoded
                except:
                    features['category_encoded'] = 0
            
            # Make prediction
            X_pred = np.array([list(features.values())])
            predicted_amount = expense_model.predict(X_pred)[0]
            
            if predicted_amount > 0:
                total_prediction += predicted_amount
        
        # If no prediction was made, use average spending
        if total_prediction == 0 and 'amount' in df.columns:
            monthly_avg = df['amount'].mean() * 30  # Daily average * 30 days
            total_prediction = monthly_avg
        
        return round(total_prediction, 2)
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return 0.0

def store_prediction_to_firestore(user_uid: str, predicted_expense: float) -> bool:
    """Store user's prediction back to Firestore"""
    global db
    
    if not db or predicted_expense <= 0:
        return False
    
    try:
        # Store in user's prediction subcollection
        prediction_data = {
            'predicted_expense': predicted_expense,
            'created_at': datetime.now(),
            'month': datetime.now().strftime('%B %Y')
        }
        
        # Store in users/{uid}/prediction/next_month document
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        prediction_ref.set(prediction_data)
        
        print(f"Stored prediction {predicted_expense} for user {user_uid}")
        return True
    except Exception as e:
        print(f"Error storing prediction: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Expense Prediction API",
        "version": "1.0",
        "endpoints": [
            "GET /transactions - Get all transactions",
            "GET /predictions - Get expense predictions",
            "POST /predict - Generate new predictions",
            "GET /train - Train the ML model"
        ]
    })

@app.route('/transactions', methods=['GET'])
@require_auth
def get_transactions(user_uid):
    """Get user's transactions from Firestore"""
    try:
        if not db:
            return jsonify({
                "success": False,
                "error": "Firebase not initialized. Please configure Firebase credentials."
            }), 503
        
        transactions = get_transactions_data(user_uid)
        return jsonify({
            "success": True,
            "count": len(transactions),
            "user_uid": user_uid,
            "transactions": transactions
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/train', methods=['GET'])
@require_auth
def train_model(user_uid):
    """Train the expense prediction model for a specific user"""
    try:
        if not db:
            return jsonify({
                "success": False,
                "error": "Firebase not initialized. Please configure Firebase credentials."
            }), 503
        
        transactions = get_transactions_data(user_uid)
        if not transactions:
            return jsonify({
                "success": False,
                "error": "No transaction data found for this user"
            }), 400
        
        df = prepare_data_for_ml(transactions)
        success = train_expense_model(df)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Model trained successfully",
                "user_uid": user_uid,
                "data_points": len(df)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to train model"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predictions', methods=['GET'])
@require_auth
def get_predictions(user_uid):
    """Get user's existing predictions from Firestore"""
    global db
    
    try:
        if not db:
            return jsonify({
                "success": False,
                "error": "Firebase not initialized. Please configure Firebase credentials."
            }), 503
        
        # Get user's prediction
        prediction_ref = db.collection('users').document(user_uid).collection('prediction').document('next_month')
        doc = prediction_ref.get()
        
        if doc.exists:
            prediction_data = doc.to_dict()
            return jsonify({
                "success": True,
                "user_uid": user_uid,
                "predicted_expense": prediction_data.get('predicted_expense', 0),
                "month": prediction_data.get('month', ''),
                "created_at": prediction_data.get('created_at')
            })
        else:
            return jsonify({
                "success": True,
                "user_uid": user_uid,
                "predicted_expense": None,
                "message": "No predictions found for this user"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
@require_auth
def generate_predictions(user_uid):
    """Generate new expense prediction for user and store it"""
    try:
        if not db:
            return jsonify({
                "success": False,
                "error": "Firebase not initialized. Please configure Firebase credentials."
            }), 503
        
        # Generate prediction for next month
        predicted_expense = predict_next_month_expense(user_uid)
        
        if predicted_expense <= 0:
            return jsonify({
                "success": False,
                "error": "Not enough data to generate prediction. Please add more transactions."
            }), 400
        
        # Store prediction to Firestore
        stored = store_prediction_to_firestore(user_uid, predicted_expense)
        
        return jsonify({
            "success": True,
            "message": "Prediction generated and stored successfully",
            "user_uid": user_uid,
            "predicted_expense": predicted_expense,
            "stored_to_firestore": stored
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    firebase_status = db is not None
    model_status = expense_model is not None
    
    return jsonify({
        "status": "healthy",
        "firebase_connected": firebase_status,
        "model_trained": model_status,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting Expense Prediction API...")
    
    # Initialize Firebase on startup
    initialize_firebase()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)