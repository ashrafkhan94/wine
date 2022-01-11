import pandas as pd
from config import get_config


def get_data(config_path):
    config = get_config(config_path)
    data_path = config["data_source"]["s3_source"]
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    return df
