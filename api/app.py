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

    MODEL_NAME = "GRU_Volatility"
    STAGE = "Production"

    client = MlflowClient()

    mv = client.get_latest_versions(
        MODEL_NAME,
        stages=[STAGE]
    )[0]

    model_uri = f"runs:/{mv.run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    df = fetch_all_ohlcv()
    df = df.iloc[:-1]

    predictions = model.predict(df)

    return predictions.to_dict()