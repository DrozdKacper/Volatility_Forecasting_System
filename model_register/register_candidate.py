from mlflow import MlflowClient, MlflowException
import mlflow

from mlflow import MlflowClient, MlflowException
import mlflow

def register_latest_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    with open("artifacts/latest_run.txt", "r") as f:
        latest_run_id = f.read().strip()

    model_uri = f"runs:/{latest_run_id}/model"

    try:
        client.create_registered_model("GRU_Volatility")
    except MlflowException:
        print("Model already exists")

    model_version = client.create_model_version(
        name="GRU_Volatility",
        source=model_uri,
        run_id=latest_run_id
    )

    client.transition_model_version_stage(
        name="GRU_Volatility",
        version=model_version.version,
        stage="Staging"
    )

    print("Registered version: ", model_version.version)


if __name__ == "__main__":
    register_latest_model()