import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import argparse
import joblib
from config import get_config
import mlflow
from urllib.parse import urlparse


def evaluate_metrics(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = get_config(config_path)
    train_path = config["split_data"]["train_path"]
    test_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target = config["base"]["target_col"]
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = train.drop(target, axis=1), train[target]
    X_test, y_test = test.drop(target, axis=1), test[target]

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]
    artifacts_dir = mlflow_config["artifacts_dir"]
    registered_model_name = mlflow_config["registered_model_name"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as mlops_run:
        lr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state
        )
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        (rmse, mae, r2) = evaluate_metrics(y_pred, y_test)
        print(rmse, mae, r2)

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(lr, model_path)

        # params_path = config["reports"]["params"]
        # scores_path = config["reports"]["scores"]
        #
        # with open(scores_path, "w") as f:
        #     scores = {
        #         "RMSE": rmse,
        #         "MAE": mae,
        #         "R2_SCORE": r2
        #     }
        #     json.dump(scores, f, indent=4)
        #
        # with open(params_path, "w") as f:
        #     params = {
        #         "alpha": alpha,
        #         "l1_ratio": l1_ratio
        #     }
        #     json.dump(params, f, indent=4)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        """ Creates a db & ui for logging"""
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr,
                                     "model",
                                     registered_model_name=registered_model_name)
        else:
            mlflow.sklearn.log_model(lr,
                                     "model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config_path)
