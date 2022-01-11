import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from config import get_config


def split_data(config_path):
    config = get_config(config_path)
    test_path = config["split_data"]["test_path"]
    train_path = config["split_data"]["train_path"]
    test_size = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    raw_path = config["load_data"]["raw_dataset_csv"]

    df = pd.read_csv(raw_path, sep=",")
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(train_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_path, sep=",", index=False, encoding="utf-8")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", default="params.yaml")
    parsed_args = args.parse_args()
    split_data(parsed_args.config_path)