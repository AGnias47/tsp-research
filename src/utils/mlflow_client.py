"""
Utilized Google Gemini to set this up
"""

import mlflow

from config import config
from src.models.tsp import TSP
import tempfile
import numpy as np
# mlflow.set_tracking_uri(config.mlflow_uri)

def log_results(problem_name: str, algorithm: TSP):
    with mlflow.start_run() as run:
        mlflow.log_param("problem", problem_name)
        mlflow.log_param("algorithm", algorithm.algorithm_name)
        mlflow.log_params(algorithm.hyperparameters)
        mlflow.log_metric("runtime", algorithm.runtime)
        mlflow.log_metric("cost", algorithm.best_cost)
        with tempfile.NamedTemporaryFile() as fp:
            np.savetxt(fp, algorithm.best_route)
            mlflow.log_artifact(fp.name)
