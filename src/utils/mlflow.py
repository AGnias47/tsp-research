import mlflow

from config import config

mlflow.set_tracking_uri(config.mlflow_uri)
