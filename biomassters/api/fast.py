from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from taxifare.ml_logic.registry import load_model
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.params import DTYPES_RAW_OPTIMIZED
import pandas as pd

app = FastAPI()

app.state.model = load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(pickup_datetime: datetime,  # 2013-07-06 17:18:00
            pickup_longitude: float,    # -73.950655
            pickup_latitude: float,     # 40.783282
            dropoff_longitude: float,   # -73.984365
            dropoff_latitude: float,    # 40.769802
            passenger_count: int):      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    key = ''
    fare_amount = 5.32

    X_pred_raw = pd.DataFrame ([[key, fare_amount, pickup_datetime,
                               pickup_longitude, pickup_latitude,
                               dropoff_longitude, dropoff_latitude,
                               passenger_count]])
    X_pred_raw.columns = ['key', 'fare_amount', 'pickup_datetime',
                          'pickup_longitude', 'pickup_latitude',
                          'dropoff_longitude', 'dropoff_latitude',
                          'passenger_count']

    X_pred_raw = clean_data(X_pred_raw)
    X_pred = preprocess_features(X_pred_raw)
    y_pred = app.state.model.predict(X_pred)

#    if (pickup_datetime == datetime.strptime ('2013-07-06 17:18:00', '%Y-%m-%d %H:%M:%S') and
#        pickup_longitude == float(-73.950655) and
#        pickup_latitude == float(40.783282) and
#        dropoff_longitude == float(-73.950655) and
#        dropoff_latitude == float(40.783282) and
#        passenger_count ==  int(2)):
    return {'fare_amount': float(y_pred[0][0])}
#    else:
#        return {'fare_amount': -1}


@app.get("/")
def root():
    # Define a root `/` endpoint
    return {'greeting': 'Hello'}
