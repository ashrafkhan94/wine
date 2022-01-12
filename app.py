from flask import Flask, render_template, request, jsonify
import os
from prediction_service import prediction


webapp_root = "webapp"
static_path = os.path.join(webapp_root, "static")
template_path = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_path, template_folder=template_path)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        """try:"""
        if request.form:
            data = dict(request.form).values()
            data = [x[0] for x in data]
            X = list(map(float, data))
            y_pred = prediction.predict(X)
            return render_template("index.html", response=y_pred)
        elif request.json:
            y_pred = prediction.api_response(request)
            return jsonify(y_pred)
        
        """except Exception as e:
            print(e)
            error = {"error": "Something Went Wrong!!"}
            return render_template("404.html", error=error)"""
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
