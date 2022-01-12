from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np

webapp_root = "webapp"
static_path = os.path.join(webapp_root, "static")
template_path = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_path, template_folder=template_path)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pass
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
