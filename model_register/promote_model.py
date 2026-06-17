import os

import mlflow
from mlflow.tracking import MlflowClient

def promote_champion():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    client = MlflowClient()

    staging_models = client.get_latest_versions(
        name="GRU_Volatility",
        stages=["Staging"]
    )

    if not staging_models:
        print("No staging model found")
        return

    prod_models = client.get_latest_versions(
        name="GRU_Volatility",
        stages=["Production"]
    )

    # pick best candidate by mse
    def get_mse(m):
        run = client.get_run(m.run_id)
        return run.data.metrics.get("mse_mean", float("inf"))

    candidate = min(staging_models, key=get_mse)

    candidate_run = client.get_run(candidate.run_id)
    candidate_mse = candidate_run.data.metrics.get("mse_mean")

    # first deployment
    if not prod_models:
        client.transition_model_version_stage(
            name="GRU_Volatility",
            version=candidate.version,
            stage="Production"
        )
        print("First production model deployed")
        return

    prod = max(prod_models, key=get_mse)

    prod_run = client.get_run(prod.run_id)
    prod_mse = prod_run.data.metrics.get("mse_mean")

    print("Candidate MSE:", candidate_mse)
    print("Production MSE:", prod_mse)

    if candidate_mse < prod_mse:

        client.transition_model_version_stage(
            name="GRU_Volatility",
            version=prod.version,
            stage="Archived"
        )

        client.transition_model_version_stage(
            name="GRU_Volatility",
            version=candidate.version,
            stage="Production"
        )

        print("New champion promoted!")

    else:
        client.transition_model_version_stage(
            name="GRU_Volatility",
            version=candidate.version,
            stage="Archived"
        )

        print("Candidate rejected")


if __name__ == "__main__":
    promote_champion()