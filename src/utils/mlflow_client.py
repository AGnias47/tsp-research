"""
Utilized Google Gemini to set this up
"""

from pathlib import Path
from uuid import uuid4

import mlflow
import numpy as np
import regex

from config import config
from src.models.tsp import TSP
from src.utils.git_utils import get_short_hash

ARTIFACTS_DIRECTORY = config.mlflow["artifacts_directory"]
# https://regex101.com/r/05tJx2/1
SIZE_REGEX = regex.compile(r"[A-Za-z]+(?P<problem_size>\d+)")



def log_results(problem_name: str, algorithm: TSP):
    with mlflow.start_run(run_name=f"{problem_name}-{algorithm.abbreviation}"):
        print("Logging results to MLflow")
        mlflow.log_param("problem", problem_name)
        mlflow.log_param(
            "problem_size", SIZE_REGEX.search(problem_name)["problem_size"]
        )
        mlflow.log_param("algorithm", algorithm.algorithm_name)
        mlflow.log_param("commit_hash", get_short_hash())
        mlflow.log_params(algorithm.hyperparameters)
        mlflow.log_metric("runtime", algorithm.runtime)
        mlflow.log_metric("cost", algorithm.best_cost)
        Path(ARTIFACTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        filename = f"{ARTIFACTS_DIRECTORY}/{uuid4()}.txt"
        np.savetxt(filename, algorithm.best_route, fmt="%d")
        mlflow.log_artifact(filename)
        if concorde_result := config.concorde_results.get(problem_name):
            error_rate = round((abs(concorde_result - algorithm.best_cost) / concorde_result) * 100, 3)
            mlflow.log_metric("error_rate", error_rate)
