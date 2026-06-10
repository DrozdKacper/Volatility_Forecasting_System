from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

mv = client.get_latest_versions("GRU_Volatility")[0]

print("VERSION:", mv.version)
print("SOURCE:", mv.source)
print("RUN_ID:", mv.run_id)