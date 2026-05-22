import mlflow
from mlflow.tracking import MlflowClient

def promote_champion():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    with open("artifacts/latest_run.txt", "r") as f:
        dvc_run_id = f.read().strip()

    staging_models = client.get_latest_versions(
        name="GRU_Volatility",
        stages=["Staging"]
    )

    if len(staging_models) == 0:
        print("No staging model found")
        return

    candidate = staging_models[0]

    if dvc_run_id:
        for m in staging_models:
            if m.run_id == dvc_run_id:
                candidate = m
                break

    candidate_run = client.get_run(candidate.run_id)

    candidate_mse = candidate_run.data.metrics["mse_mean"]

    prod_models = client.get_latest_versions(
        "GRU_Volatility",
        stages=["Production"]
    )


    # first deployment
    if len(prod_models) == 0:

        print("No production model yet")

        client.transition_model_version_stage(
            name="GRU_Volatility",
            version=candidate.version,
            stage="Production"
        )

        print("First production model deployed")

    else:

        prod_model = prod_models[0]

        prod_run = client.get_run(prod_model.run_id)

        prod_mse = prod_run.data.metrics["mse_mean"]

        print("Candidate MSE:", candidate_mse)
        print("Production MSE:", prod_mse)

        # lower mse = better
        if candidate_mse < prod_mse:

            client.transition_model_version_stage(
                name="GRU_Volatility",
                version=prod_model.version,
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

            print("Candidate archived")

if __name__ == "__main__":
    promote_champion()