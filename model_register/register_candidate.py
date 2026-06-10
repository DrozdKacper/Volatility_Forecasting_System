from mlflow import MlflowClient, MlflowException
import mlflow

from mlflow import MlflowClient, MlflowException
import mlflow

def register_latest_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    experiment = client.get_experiment_by_name("GRU_model")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found")

    latest_run_id = runs[0].info.run_id

    model_uri = f"runs:/{latest_run_id}/model"

    try:
        client.get_registered_model("GRU_Volatility")
    except MlflowException:
        client.create_registered_model("GRU_Volatility")

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

    print("Registered version:", model_version.version)


if __name__ == "__main__":
    register_latest_model()