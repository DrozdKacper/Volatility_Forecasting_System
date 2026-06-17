from mlflow import MlflowClient

client = MlflowClient()

for mv in client.search_model_versions("name='GRU_Volatility'"):
    print("version:", mv.version)
    print("source:", mv.source)
    print("run_id:", mv.run_id)
    print("---")



run = client.get_run(
    "b7012d1759a14ca3a0dbc94172c65623"
)

print(run.info.artifact_uri)


for f in client.list_artifacts(
    "b7012d1759a14ca3a0dbc94172c65623"
):
    print(f.path)