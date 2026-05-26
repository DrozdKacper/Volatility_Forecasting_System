from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from live_market_data.fetch_live_ohlcv import fetch_all_ohlcv

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = FastAPI()

model = None


@app.on_event("startup")
def load():

    global model

    client = MlflowClient()

    mv = client.get_latest_versions(
        "GRU_Volatility",
        stages=["Production"]
    )[0]

    print("MODEL SOURCE:", mv.  source)

    model = mlflow.pyfunc.load_model(
        mv.source
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():

    df = fetch_all_ohlcv()
    df = df.iloc[:-1]

    predictions = model.predict(df)

    return predictions.to_dict()