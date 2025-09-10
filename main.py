from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any

app = Flask(__name__)

# Global variables for Firebase and ML model
db = None
expense_model = None
category_encoder = None

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global db
    try:
        if not firebase_admin._apps:
            # Check if we have a service account key file
            if os.path.exists('serviceAccountKey.json'):
                cred = credentials.Certificate('serviceAccountKey.json')
                firebase_admin.initialize_app(cred)
            else:
                # Use default credentials if available
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        print("Firebase initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

def get_transactions_data() -> List[Dict[str, Any]]:
    """Retrieve all transactions from Firestore"""
    global db
    if not db:
        if not initialize_firebase():
            return []
    
    try:
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

def predict_future_expenses(days_ahead: int = 30) -> List[Dict[str, Any]]:
    """Predict future expenses for the next N days"""
    global expense_model, category_encoder
    
    if not expense_model:
        return []
    
    predictions = []
    current_date = datetime.now()
    
    # Get historical data to determine common categories
    transactions = get_transactions_data()
    df = prepare_data_for_ml(transactions)
    
    if df.empty:
        return []
    
    # Get most common categories
    common_categories = df['category'].value_counts().head(5).index.tolist()
    
    try:
        for i in range(days_ahead):
            future_date = current_date + timedelta(days=i+1)
            
            for category in common_categories:
                # Prepare features for prediction
                features = {
                    'day_of_year': future_date.timetuple().tm_yday,
                    'month_num': future_date.month,
                    'day_of_week': future_date.weekday()
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
                
                # Only include meaningful predictions (positive amounts)
                if predicted_amount > 0:
                    prediction = {
                        'date': future_date.strftime('%Y-%m-%d'),
                        'day': future_date.strftime('%Y-%m-%d'),
                        'month': future_date.strftime('%B'),
                        'week': f"Week {future_date.isocalendar()[1]}",
                        'category': category,
                        'predicted_amount': round(predicted_amount, 2),
                        'description': f"Predicted {category} expense",
                        'type': 'prediction'
                    }
                    predictions.append(prediction)
    
    except Exception as e:
        print(f"Error making predictions: {e}")
    
    return predictions

def store_predictions_to_firestore(predictions: List[Dict[str, Any]]) -> bool:
    """Store predictions back to Firestore"""
    global db
    
    if not db or not predictions:
        return False
    
    try:
        # Create a predictions collection
        predictions_ref = db.collection('predictions')
        
        # Clear existing predictions
        existing_predictions = predictions_ref.stream()
        for doc in existing_predictions:
            doc.reference.delete()
        
        # Store new predictions
        for prediction in predictions:
            predictions_ref.add(prediction)
        
        print(f"Stored {len(predictions)} predictions to Firestore")
        return True
    except Exception as e:
        print(f"Error storing predictions: {e}")
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
def get_transactions():
    """Get all transactions from Firestore"""
    try:
        transactions = get_transactions_data()
        return jsonify({
            "success": True,
            "count": len(transactions),
            "transactions": transactions
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/train', methods=['GET'])
def train_model():
    """Train the expense prediction model"""
    try:
        transactions = get_transactions_data()
        if not transactions:
            return jsonify({
                "success": False,
                "error": "No transaction data found"
            }), 400
        
        df = prepare_data_for_ml(transactions)
        success = train_expense_model(df)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Model trained successfully",
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
def get_predictions():
    """Get existing predictions from Firestore"""
    global db
    
    try:
        if not db:
            initialize_firebase()
        
        predictions_ref = db.collection('predictions')
        docs = predictions_ref.stream()
        
        predictions = []
        for doc in docs:
            prediction_data = doc.to_dict()
            prediction_data['id'] = doc.id
            predictions.append(prediction_data)
        
        return jsonify({
            "success": True,
            "count": len(predictions),
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def generate_predictions():
    """Generate new expense predictions and store them"""
    try:
        # Get parameters from request
        data = request.get_json() or {}
        days_ahead = data.get('days_ahead', 30)
        
        # First, train the model with latest data
        transactions = get_transactions_data()
        if not transactions:
            return jsonify({
                "success": False,
                "error": "No transaction data found for training"
            }), 400
        
        df = prepare_data_for_ml(transactions)
        model_trained = train_expense_model(df)
        
        if not model_trained:
            return jsonify({
                "success": False,
                "error": "Failed to train prediction model"
            }), 500
        
        # Generate predictions
        predictions = predict_future_expenses(days_ahead)
        
        if not predictions:
            return jsonify({
                "success": False,
                "error": "No predictions could be generated"
            }), 400
        
        # Store predictions to Firestore
        stored = store_predictions_to_firestore(predictions)
        
        return jsonify({
            "success": True,
            "message": "Predictions generated and stored successfully",
            "prediction_count": len(predictions),
            "stored_to_firestore": stored,
            "predictions": predictions[:10]  # Return first 10 predictions
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