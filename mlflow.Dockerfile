FROM ghcr.io/mlflow/mlflow:v3.11.1
RUN pip install boto3==1.43.29
EXPOSE 5000