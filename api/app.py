import os

from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from live_market_data.fetch_live_ohlcv import fetch_all_ohlcv

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
app = FastAPI()




model = None


def get_model():
    global model
    if model is None:
        client = MlflowClient()

        mv = client.get_latest_versions(
            "GRU_Volatility",
            stages=["Production"]
        )[0]

        model_uri = f"runs:/{mv.run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

    return model

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    model = get_model()

    df = fetch_all_ohlcv()
    df = df.iloc[:-1]

    preds = model.predict(df)
    return preds.to_dict()