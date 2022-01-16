import os
import joblib
from src import config
import numpy as np


def predict(x):
    conf = config.get_config()
    model_dir = conf["webapp_model_dir"]
    file = os.path.join(model_dir, "model.joblib")
    with open(file, "rb") as f:
        model = joblib.load(f)

    prediction = model.predict([x])[0]
    return prediction


def api_response(request):
    conf = config.get_config()
    model_dir = conf["webapp_model_dir"]
    x = np.array(list(request.json.values()))
    file = os.path.join(model_dir, "model.joblib")
    with open(file, "rb") as f:
        model = joblib.load(f)
    prediction = model.predict([x])[0]
    response = {"response": prediction}
    return response
