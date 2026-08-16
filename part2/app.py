"""
Part 2 - Deployment (Aidan).

Small Flask API + simple HTML form around the final tuned model
(part2/final_model.pkl - see PART2_PROGRESS.md for how it was chosen).

Run it: python3 part2/app.py  (from the main project folder)
Then open http://127.0.0.1:5000 in a browser for the form, or POST JSON to
http://127.0.0.1:5000/predict for the raw API.

Example API call:
    curl -X POST http://127.0.0.1:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"DISTRICT": "Scarborough", "LIGHT": "Dark", "HOUR": 5, "SPEEDING": "Yes"}'

You don't need to provide every field - anything left out is treated as
missing and the model's own imputer (same one used during training) fills
it in the same way it does for missing values in the real dataset.
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.pkl")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
pipeline = saved["pipeline"]
model_name = saved["model_name"]

# Pull the exact list of columns the model expects straight from the fitted
# preprocessor inside the pipeline, instead of hardcoding a second copy that
# could drift out of sync with KSI.py.
preprocessor = pipeline.named_steps["preprocessor"]
ALL_COLUMNS = []
for name, transformer, cols in preprocessor.transformers_:
    if name == "remainder":
        continue
    ALL_COLUMNS.extend(cols)

app = Flask(__name__)

# A handful of the fields exploration found to matter most, for the simple
# form. The API itself (/predict) accepts any subset of the real columns,
# not just these - this is just what's shown on the page.
FORM_FIELDS = [
    ("DISTRICT", "text", "e.g. Scarborough, Toronto and East York"),
    ("LIGHT", "text", "e.g. Dark, Daylight"),
    ("RDSFCOND", "text", "e.g. Dry, Wet, Snow, Ice"),
    ("VISIBILITY", "text", "e.g. Clear, Rain"),
    ("TRAFFCTL", "text", "e.g. Traffic Signal, No Control"),
    ("HOUR", "number", "0-23"),
    ("LATITUDE", "number", "e.g. 43.7"),
    ("LONGITUDE", "number", "e.g. -79.4"),
    ("SPEEDING", "text", "Yes or leave blank"),
    ("ALCOHOL", "text", "Yes or leave blank"),
    ("AG_DRIV", "text", "Yes or leave blank"),
    ("PEDESTRIAN", "text", "Yes or leave blank"),
]

PAGE = """
<!doctype html>
<title>KSI Fatality Risk Predictor</title>
<h2>KSI Fatality Risk Predictor</h2>
<p>Model: {{ model_name }}</p>
<p>Fill in what you know, leave the rest blank - the model fills in missing
values the same way it was trained to.</p>
<form method="POST" action="/predict_form">
{% for name, type, hint in fields %}
  <label>{{ name }} ({{ hint }}): <input type="{{ type }}" name="{{ name }}"></label><br><br>
{% endfor %}
  <button type="submit">Predict</button>
</form>
{% if result %}
<h3>Result</h3>
<p>Prediction: <b>{{ result.prediction }}</b></p>
<p>Estimated probability of fatal: {{ result.probability }}</p>
{% endif %}
"""


def build_row(data: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with every column the model expects.
    Anything not provided becomes None -> the pipeline's own imputer
    (fit during training) fills it in, same as it does for real missing data.
    """
    row = {col: (data.get(col) or None) for col in ALL_COLUMNS}
    return pd.DataFrame([row])


def run_prediction(data: dict) -> dict:
    df = build_row(data)
    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][1]
    return {
        "prediction": "FATAL" if pred == 1 else "NOT FATAL",
        "probability": round(float(proba), 3),
    }


@app.route("/")
def home():
    return render_template_string(PAGE, fields=FORM_FIELDS, model_name=model_name, result=None)


@app.route("/predict_form", methods=["POST"])
def predict_form():
    data = {k: v for k, v in request.form.items()}
    result = run_prediction(data)
    return render_template_string(PAGE, fields=FORM_FIELDS, model_name=model_name, result=result)


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True) or {}
    result = run_prediction(data)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model_name, "n_features_expected": len(ALL_COLUMNS)})


if __name__ == "__main__":
    print(f"Loaded model: {model_name}")
    print(f"Expects {len(ALL_COLUMNS)} columns: {ALL_COLUMNS}")
    app.run(debug=True, port=5000)
