    # ==============================================================================
    #  Personal Finance Prediction API with Advanced ML Models
    #  Enhanced Ensemble Prediction System (v2.1 - Refactored)
    # ==============================================================================
    """
    This Flask application provides a secure API to predict a user's next-month 
    expenses based on their transaction history from Firestore. It uses a dynamic 
    ensemble of ARIMA, Random Forest, and Trend models for robust forecasting.
    """

    import os
    import json
    import warnings
    from datetime import datetime, timedelta
    from functools import wraps
    from typing import List, Dict, Any, Optional

    import firebase_admin
    import numpy as np
    import pandas as pd
    from firebase_admin import auth, credentials, firestore
    from flask import Flask, jsonify, request

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.arima.model import ARIMA

    warnings.filterwarnings("ignore")

    # ==============================================================================
    # ## --- Configuration --- ##
    # ==============================================================================

    CONFIG = {
        "ARIMA_ORDERS": [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2)],
        "ML_LOOKBACK_STEPS": 6,
        "ENSEMBLE_WEIGHTS": {
            "high_data": {  # >= 12 months
                "arima": 0.4,
                "ml": 0.4,
                "trend": 0.15,
                "avg": 0.05,
            },
            "medium_data": {"arima": 0.5, "trend": 0.3, "avg": 0.2},  # >= 6 months
            "low_data": {"trend": 0.7, "avg": 0.3},  # < 6 months
        },
    }

    # ==============================================================================
    # ## --- Initialization & Globals --- ##
    # ==============================================================================

    app = Flask(__name__)
    db: Optional[firestore.Client] = None
    model_cache: Dict[str, Any] = {}  # Simple in-memory cache for trained models

    # ==============================================================================
    # ## --- Firebase & Authentication --- ##
    # ==============================================================================


    def initialize_firebase():
        """Initialize Firebase Admin SDK using environment variable."""
        global db
        if firebase_admin._apps:
            db = firestore.client()
            print("b Firebase already initialized")
            return True

        try:
            firebase_key = os.environ.get("FIREBASE_KEY")
            if not firebase_key:
                print("b o8 FIREBASE_KEY secret not found. Using development mode.")
                return False

            service_account_info = json.loads(firebase_key)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("b Firebase initialized successfully with secret key")
            return True
        except Exception as e:
            print(f"b o8 Error initializing Firebase: {e}")
            return False


    def verify_firebase_token(token: str) -> Optional[str]:
        """Verify Firebase ID token and return user UID."""
        try:
            decoded_token = auth.verify_id_token(token)
            return decoded_token["uid"]
        except Exception as e:
            print(f"Token verification error: {e}")
            return None


    def require_auth(f):
        """Decorator to require Firebase authentication for an endpoint."""

        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return (
                    jsonify({"success": False, "error": "Authorization header required"}),
                    401,
                )

            token = auth_header.split("Bearer ")[1]
            user_uid = verify_firebase_token(token)
            if not user_uid:
                return jsonify({"success": False, "error": "Invalid token"}), 401

            return f(user_uid, *args, **kwargs)

        return decorated_function


    # ==============================================================================
    # ## --- Data Processing --- ##
    # ==============================================================================


    def get_user_transactions(uid: str) -> List[Dict[str, Any]]:
        """Fetch all transactions for a specific user or generate mock data."""
        global db
        if not db:
            print(f"p' Using mock data for user {uid}")
            mock_data = []
            base_date = datetime.now()
            for i in range(24):  # 24 months of data
                date = base_date - timedelta(days=30 * i)
                amount = np.random.randint(3000, 8000) + np.random.normal(0, 500)
                mock_data.append(
                    {
                        "amount": float(max(1000, amount)),
                        "date": date,
                        "category": np.random.choice(
                            ["Groceries", "Transport", "Entertainment", "Utilities"]
                        ),
                    }
                )
            return mock_data

        try:
            transactions_ref = (
                db.collection("users").document(uid).collection("transactions")
            )
            docs = transactions_ref.stream()

            transactions = []
            for doc in docs:
                data = doc.to_dict()
                if "amount" in data and "date" in data:
                    date = data["date"]
                    if hasattr(date, "seconds"):
                        date = datetime.fromtimestamp(date.seconds)
                    elif isinstance(date, str):
                        date = pd.to_datetime(date)

                    transactions.append(
                        {
                            "amount": float(data["amount"]),
                            "date": date,
                            "category": data.get("category", "Other"),
                        }
                    )

            print(f"p
     Retrieved {len(transactions)} transactions for user {uid}")
            return transactions
        except Exception as e:
            print(f"Error retrieving transactions: {e}")
            return []


    def prepare_monthly_data(transactions: List[Dict[str, Any]]) -> Optional[pd.Series]:
        """Aggregate transactions into a monthly time series."""
        if not transactions:
            return None

        try:
            df = pd.DataFrame(transactions)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            monthly_series = df["amount"].resample("ME").sum().sort_index()
            monthly_series = monthly_series[
                monthly_series > 0
            ]  # Remove months with zero spending

            if monthly_series.empty:
                return None

            print(f"p Prepared {len(monthly_series)} months of data")
            return monthly_series
        except Exception as e:
            print(f"Error preparing monthly data: {e}")
            return None

    class PredictionEngine:
        """Encapsulates model training, caching, and prediction logic."""

        def __init__(self, monthly_data: pd.Series, user_uid: str):
            self.monthly_data = monthly_data
            self.data_hash = pd.util.hash_pandas_object(monthly_data).sum()
            self.cache_key = f"{user_uid}_{self.data_hash}"

        def _get_or_train_model(self, model_name: str, training_func, *args):
            """Generic function to retrieve a trained model from cache or train it."""
            if self.cache_key in model_cache and model_name in model_cache[self.cache_key]:
                return model_cache[self.cache_key][model_name]

            model = training_func(*args)

            if self.cache_key not in model_cache:
                model_cache[self.cache_key] = {}
            model_cache[self.cache_key][model_name] = model
            return model

        def get_arima_prediction(self) -> Optional[float]:
            """ARIMA statistical model for time series prediction."""
            if len(self.monthly_data) < 3:
                return None

            def train():
                best_aic = float("inf")
                best_model = None
                for order in CONFIG["ARIMA_ORDERS"]:
                    try:
                        model = ARIMA(self.monthly_data, order=order).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_model = model
                    except Exception:
                        continue
                return best_model

            fitted_model = self._get_or_train_model("arima", train)
            if fitted_model:
                prediction = float(fitted_model.forecast(steps=1).iloc[0])
                print(f"p. ARIMA prediction: {prediction}")
                return prediction
            return None

        def get_ml_prediction(self) -> Optional[float]:
            """Enhanced ML prediction using Random Forest."""
            if len(self.monthly_data) < 6:
                return None

            values = self.monthly_data.values
            n_steps = min(CONFIG["ML_LOOKBACK_STEPS"], len(values) - 1)

            X, y = [], []
            for i in range(n_steps, len(values)):
                X.append(values[i - n_steps : i])
                y.append(values[i])

            if len(X) < 3:
                return None
            X, y = np.array(X), np.array(y)

            def create_features(data_slice):
                features = list(data_slice)
                features.append(float(np.mean(data_slice)))
                features.append(float(np.std(data_slice)))
                features.append(
                    data_slice[-1] - data_slice[-2] if len(data_slice) > 1 else 0
                )
                return features

            X_enhanced = np.array([create_features(xi) for xi in X])

            def train():
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_enhanced, y)
                return model

            model = self._get_or_train_model("random_forest", train)
            if model:
                last_features = create_features(values[-n_steps:])
                prediction = model.predict([last_features])[0]
                print(f"p$ ML prediction: {prediction}")
                return float(prediction)
            return None

        def get_trend_prediction(self) -> Optional[float]:
            """Simple trend-based prediction using Linear Regression."""
            if len(self.monthly_data) < 3:
                return None

            def train():
                X = np.arange(len(self.monthly_data)).reshape(-1, 1)
                y = self.monthly_data.values
                model = LinearRegression()
                model.fit(X, y)
                return model

            model = self._get_or_train_model("trend", train)
            if model:
                next_month_index = len(self.monthly_data)
                prediction = model.predict([[next_month_index]])[0]
                print(f"p
     Trend prediction: {prediction}")
                return float(prediction)
            return None

        def get_ensemble_prediction(self) -> Optional[float]:
            """Advanced ensemble prediction combining multiple models."""
            predictions = {
                "arima": self.get_arima_prediction(),
                "ml": self.get_ml_prediction(),
                "trend": self.get_trend_prediction(),
                "avg": self.monthly_data.tail(3).mean()
                if len(self.monthly_data) >= 3
                else self.monthly_data.mean(),
            }

            # Filter out failed predictions
            valid_predictions = {k: v for k, v in predictions.items() if v is not None}
            if not valid_predictions:
                return None

            # Select weights based on data availability
            num_months = len(self.monthly_data)
            if num_months >= 12:
                weights = CONFIG["ENSEMBLE_WEIGHTS"]["high_data"]
            elif num_months >= 6:
                weights = CONFIG["ENSEMBLE_WEIGHTS"]["medium_data"]
            else:
                weights = CONFIG["ENSEMBLE_WEIGHTS"]["low_data"]

            # Calculate weighted average
            total_weight = sum(
                weights[model] for model in valid_predictions if model in weights
            )
            if total_weight == 0:
                return valid_predictions.get("avg")

            ensemble_pred = (
                sum(
                    pred * weights[model]
                    for model, pred in valid_predictions.items()
                    if model in weights
                )
                / total_weight
            )

            # Apply safety bounds
            recent_avg = valid_predictions["avg"]
            max_reasonable = recent_avg * 3.0
            min_reasonable = recent_avg * 0.3
            final_prediction = max(min_reasonable, min(max_reasonable, ensemble_pred))

            active_models = [k for k in valid_predictions if k in weights]
            print(f"p/ Ensemble prediction: {final_prediction} (from {active_models})")
            return float(final_prediction)



    def store_prediction_to_firestore(user_uid: str, predicted_expense: float) -> bool:
        """Store the user's prediction to Firestore."""
        global db
        if not db or predicted_expense <= 0:
            return False

        try:
            prediction_data = {
                "predicted_expense": round(predicted_expense, 2),
                "created_at": datetime.now(),
                "month": (datetime.now() + timedelta(days=30)).strftime("%B %Y"),
            }
            prediction_ref = (
                db.collection("users")
                .document(user_uid)
                .collection("prediction")
                .document("next_month")
            )
            prediction_ref.set(prediction_data)

            print(f"p> Stored prediction {predicted_expense:.2f} for user {user_uid}")
            return True
        except Exception as e:
            print(f"Error storing prediction: {e}")
            return False




    @app.route("/", methods=["GET"])
    def index():
        """Home endpoint providing API information."""
        return jsonify(
            {
                "message": "p

        