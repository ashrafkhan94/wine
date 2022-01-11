from get_data import get_data
from config import get_config
import argparse


def load_save_data(config_path):
    df = get_data(config_path)
    config = get_config(config_path)
    new_col_names = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", index=False, header=new_col_names)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path", default="params.yaml")
    parsed_args = args.parse_args()
    load_save_data(config_path=parsed_args.config_path)