import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import config
import argparse
from pprint import pprint


def log_production_model(config):
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    registered_model_name = mlflow_config["registered_model_name"]

    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(experiment_ids=1)
    lowest = runs["metrics.mae"].sort_values(ascending=True)[0]
    lowest_run_id = runs[runs["metrics.mae"] == lowest]["run_id"][0]

    client = MlflowClient()

    # Change state of models in mlflow
    for mv in client.search_model_versions(f"name = '{registered_model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model_path = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=registered_model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=registered_model_name,
                version=current_version,
                stage="Staging"
            )

    model_path = os.path.join(logged_model_path, "model.pkl")
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    saved_model_path = os.path.join(config["webapp_model_dir"], "model.joblib")
    joblib.dump(model, saved_model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", default="params.yaml")
    parsed_args = args.parse_args()
    config = config.get_config(config_path=parsed_args.config_path)
    log_production_model(config)