import os
import joblib
from src import config
import numpy as np


def predict(x):
    conf = config.get_config()
    model_dir = conf["webapp_model_dir"]
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    prediction = model.predict([x])[0]
    return prediction


def api_response(request):
    conf = config.get_config()
    model_dir = conf["webapp_model_dir"]
    x = np.array(list(request.json.values()))
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    prediction = model.predict([x])[0]
    response = {"response": prediction}
    return response
