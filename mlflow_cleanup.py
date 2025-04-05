"""
Removes deleted mlflow runs on the host backend.
Taken from https://stackoverflow.com/a/63844571/8728749
"""

import shutil

import mlflow


def get_run_dir(artifacts_uri):
    return artifacts_uri[7:-10]


def remove_run_dir(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)


exp = mlflow.tracking.MlflowClient(tracking_uri="./mlruns")
for experiment_id in [
    738455394187558780,
    228647082989280729,
    761465446413456537,
    897668194981506122,
    874586957407654680,
    136562477786329152,
    967066722789595053,
    664393851428469570,
    709098389546901036,
    882873223019460293,
    0,
]:
    runs = exp.search_runs(str(experiment_id), run_view_type=2)
    _ = [remove_run_dir(get_run_dir(run.info.artifact_uri)) for run in runs]
remove_run_dir("./mlruns/.trash/")
