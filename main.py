import os
import json
import warnings
from datetime import datetime, timedelta
from functools import wraps
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import firebase_admin
from firebase_admin import credentials, firestore, auth

from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings("ignore")


# ==========================================================
# CONFIGURATION
# ==========================================================

CONFIG = {
    "ARIMA_ORDERS": [
        (1,1,0),
        (0,1,1)
    ],
    "ML_LOOKBACK_STEPS": 6
}


# ==========================================================
# FLASK SETUP
# ==========================================================

app = Flask(__name__)
CORS(app)


db: Optional[firestore.Client] = None


# Model cache
model_cache: Dict[str, Any] = {}

# Prediction cache
prediction_cache: Dict[str, float] = {}

MAX_CACHE_USERS = 20



# ==========================================================
# FIREBASE INITIALIZATION
# ==========================================================

def initialize_firebase():

    global db

    try:

        if firebase_admin._apps:
            db = firestore.client()
            print("Firebase already initialized")
            return True


        firebase_key = os.environ.get("FIREBASE_KEY")


        if not firebase_key:

            print(
                "Firebase key missing. Running mock mode"
            )

            return False



        firebase_json = json.loads(firebase_key)


        cred = credentials.Certificate(firebase_json)

        firebase_admin.initialize_app(
            cred
        )


        db = firestore.client()


        print(
            "Firebase connected successfully"
        )


        return True



    except Exception as e:

        print(
            "Firebase error:",
            e
        )

        return False



initialize_firebase()



# ==========================================================
# AUTH
# ==========================================================


def verify_token(token):

    try:

        decoded = auth.verify_id_token(
            token
        )

        return decoded.get("uid")


    except Exception as e:

        print(
            "Token error:",
            e
        )

        return None




def require_auth(function):

    @wraps(function)
    def wrapper(*args, **kwargs):


        # Render testing mode
        if db is None:

            return function(
                "mock_user",
                *args,
                **kwargs
            )



        header = request.headers.get(
            "Authorization"
        )


        if not header:

            return jsonify(
                {
                    "success":False,
                    "error":"Authorization required"
                }
            ),401



        token = header.replace(
            "Bearer ",
            ""
        )


        uid = verify_token(
            token
        )


        if not uid:

            return jsonify(
                {
                    "success":False,
                    "error":"Invalid token"
                }
            ),401



        return function(
            uid,
            *args,
            **kwargs
        )


    return wrapper





# ==========================================================
# DATA FETCH
# ==========================================================


def get_user_transactions(uid):


    if db is None:


        np.random.seed(
            abs(hash(uid))
            &
            0xffffffff
        )


        data=[]

        now=datetime.now()



        for i in range(24):

            date = (
                now -
                pd.DateOffset(months=i)
            ).replace(day=1)


            amount=float(
                np.random.randint(
                    3000,
                    8000
                )
            )


            data.append(
                {
                    "amount":amount,
                    "date":date,
                    "category":"Other"
                }
            )


        return data



    try:


        docs = (
            db.collection("users")
            .document(uid)
            .collection("transactions")
            .stream()
        )


        result=[]


        for doc in docs:


            d=doc.to_dict()


            if (
                "amount" in d
                and
                "date" in d
            ):


                date=d["date"]


                if hasattr(
                    date,
                    "seconds"
                ):

                    date=datetime.fromtimestamp(
                        date.seconds
                    )


                result.append(
                    {
                        "amount":
                        float(
                            d["amount"]
                        ),

                        "date":
                        date,

                        "category":
                        d.get(
                            "category",
                            "Other"
                        )
                    }
                )


        return result



    except Exception as e:


        print(
            "Firestore fetch error:",
            e
        )


        return []





def prepare_monthly_data(
        transactions
):


    if not transactions:

        return None



    df=pd.DataFrame(
        transactions
    )


    df["date"]=pd.to_datetime(
        df["date"]
    )


    df.set_index(
        "date",
        inplace=True
    )


    monthly=(
        df["amount"]
        .resample("MS")
        .sum()
        .sort_index()
    )


    return monthly[monthly>0]





# ==========================================================
# MODELS
# ==========================================================


def arima_prediction(series):


    best=None
    best_aic=float("inf")



    for order in CONFIG["ARIMA_ORDERS"]:

        try:

            model=ARIMA(
                series,
                order=order
            ).fit()


            if model.aic < best_aic:

                best_aic=model.aic

                best=float(
                    model.forecast(
                        1
                    ).iloc[0]
                )


        except:

            continue



    return best
# ==========================================================
# PREDICTION ENGINE
# ==========================================================

class PredictionEngine:


    def __init__(self, monthly_data, uid):

        self.monthly_data = monthly_data
        self.uid = uid


        self.cache_key = (
            uid +
            "_" +
            str(
                int(
                    pd.util
                    .hash_pandas_object(monthly_data)
                    .sum()
                )
            )
        )


        if self.cache_key not in model_cache:

            model_cache[self.cache_key] = {}




    def cached(self,name):

        return model_cache[self.cache_key].get(name)




    def save_cache(self,name,value):

        if len(model_cache) > MAX_CACHE_USERS:

            first=list(model_cache.keys())[0]

            del model_cache[first]


        model_cache[self.cache_key][name]=value





    # ---------------- ARIMA ----------------

    def get_arima(self):


        old=self.cached("arima")


        if old:

            return old



        pred=arima_prediction(
            self.monthly_data
        )


        if pred:

            self.save_cache(
                "arima",
                pred
            )


        return pred





    # ---------------- RANDOM FOREST ----------------

    def get_random_forest(self):


        old=self.cached(
            "rf"
        )


        if old:

            model=old


        else:


            values=self.monthly_data.values


            if len(values)<6:

                return None



            X=[]
            y=[]



            for i in range(
                6,
                len(values)
            ):


                window=values[i-6:i]


                X.append(
                    [
                        *window,
                        window.mean(),
                        window.std(),
                        window[-1]-window[-2]
                    ]
                )


                y.append(
                    values[i]
                )



            if len(X)<3:

                return None



            model=RandomForestRegressor(
                n_estimators=30,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )


            model.fit(
                X,
                y
            )


            self.save_cache(
                "rf",
                model
            )



        last=self.monthly_data.values[-6:]


        features=[
            *last,
            last.mean(),
            last.std(),
            last[-1]-last[-2]
        ]


        try:

            return float(
                model.predict(
                    [features]
                )[0]
            )


        except:

            return None





    # ---------------- TREND ----------------


    def get_trend(self):


        old=self.cached(
            "trend"
        )


        if old:

            model=old


        else:

            X=np.arange(
                len(self.monthly_data)
            ).reshape(
                -1,
                1
            )


            y=self.monthly_data.values


            model=LinearRegression()


            model.fit(
                X,
                y
            )


            self.save_cache(
                "trend",
                model
            )



        try:

            return float(
                model.predict(
                    [
                        [
                            len(
                                self.monthly_data
                            )
                        ]
                    ]
                )[0]
            )


        except:

            return None





    # ---------------- CUSTOM MODEL ----------------


    def get_custom(self):


        if len(self.monthly_data)<3:

            return None


        return float(
            self.monthly_data
            .tail(6)
            .mean()
            *
            1.05
        )





    # ---------------- FINAL ----------------


    def predict(self):


        cache_key=self.uid+"_final"


        if cache_key in prediction_cache:

            return prediction_cache[
                cache_key
            ]



        predictions=[

            self.get_arima(),

            self.get_random_forest(),

            self.get_trend(),

            self.get_custom()

        ]



        valid=[

            x for x in predictions

            if x and x>0

        ]



        if not valid:

            return None



        result=max(valid)



        prediction_cache[
            cache_key
        ]=result



        return result





# ==========================================================
# FIRESTORE SAVE
# ==========================================================


def save_prediction(
        uid,
        value
):


    if db is None:

        return False



    try:


        db.collection(
            "users"
        ).document(
            uid
        ).collection(
            "prediction"
        ).document(
            "next_month"
        ).set(

            {

                "predicted_expense":
                round(
                    value,
                    2
                ),


                "created_at":
                datetime.now(),


                "month":
                (
                    datetime.now()
                    +
                    timedelta(days=30)
                )
                .strftime(
                    "%B %Y"
                )

            }

        )


        return True



    except Exception as e:


        print(
            "Save error:",
            e
        )


        return False





# ==========================================================
# ROUTES
# ==========================================================


@app.route("/")
def home():

    return jsonify(

        {

            "message":
            "MoneyMinder Expense Prediction API",

            "version":
            "3.0",

            "models":
            [
                "ARIMA",
                "RandomForest",
                "LinearRegression",
                "CustomAverage"
            ]

        }

    )





@app.route(
    "/transactions",
    methods=["GET"]
)
@require_auth
def transactions(uid):


    data=get_user_transactions(
        uid
    )


    return jsonify(

        {

            "success":True,

            "count":
            len(data),

            "transactions":
            data

        }

    )





@app.route(
    "/train",
    methods=["GET"]
)
@require_auth
def train(uid):


    tx=get_user_transactions(
        uid
    )


    monthly=prepare_monthly_data(
        tx
    )


    if monthly is None:

        return jsonify(

            {
                "success":False,
                "error":"No data"
            }

        ),400




    engine=PredictionEngine(
        monthly,
        uid
    )


    engine.get_arima()

    engine.get_random_forest()

    engine.get_trend()

    engine.get_custom()



    return jsonify(

        {

            "success":True,

            "message":
            "Models trained",

            "data_points":
            len(monthly)

        }

    )





@app.route(
    "/predict",
    methods=["POST"]
)
@require_auth
def predict(uid):


    tx=get_user_transactions(
        uid
    )


    monthly=prepare_monthly_data(
        tx
    )


    if monthly is None:

        return jsonify(

            {
                "success":False,
                "error":"No transaction data"
            }

        ),400




    engine=PredictionEngine(
        monthly,
        uid
    )


    result=engine.predict()



    if result is None:

        return jsonify(

            {
                "success":False,
                "error":"Prediction failed"
            }

        ),400




    saved=save_prediction(
        uid,
        float(result)
    )



    return jsonify(

        {

            "success":True,

            "predicted_expense":
            round(
                result,
                2
            ),

            "stored":
            saved

        }

    )





@app.route("/health")
def health():


    return jsonify(

        {

            "status":
            "healthy",

            "firebase":
            db is not None,

            "time":
            datetime.now().isoformat()

        }

    )





# ==========================================================
# START SERVER
# ==========================================================


if __name__=="__main__":


    port=int(
        os.environ.get(
            "PORT",
            10000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port

    )