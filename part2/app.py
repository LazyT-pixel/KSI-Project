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
import numpy as np
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
<style>
  :root {
    --navy: #1b2a4a;
    --blue: #2c5aa0;
    --red: #c0392b;
    --green: #1e8449;
    --bg: #f4f6f9;
    --border: #d3dae3;
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--bg);
    color: #222;
    margin: 0;
    padding: 32px 16px;
  }
  .card {
    max-width: 640px;
    margin: 0 auto;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 28px 32px 32px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  }
  h2 {
    margin: 0 0 4px;
    color: var(--navy);
  }
  .model-tag {
    display: inline-block;
    font-size: 13px;
    color: var(--blue);
    background: #eaf1fb;
    border-radius: 20px;
    padding: 3px 12px;
    margin-bottom: 16px;
  }
  .hint {
    color: #666;
    font-size: 14px;
    margin-bottom: 22px;
  }
  .field {
    display: grid;
    grid-template-columns: 150px 1fr;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }
  .field label {
    font-weight: 600;
    font-size: 14px;
    color: var(--navy);
  }
  .field .sub {
    display: block;
    font-weight: 400;
    color: #888;
    font-size: 12px;
  }
  .field input {
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 14px;
  }
  .field input:focus {
    outline: none;
    border-color: var(--blue);
    box-shadow: 0 0 0 2px rgba(44,90,160,0.15);
  }
  button {
    margin-top: 12px;
    background: var(--blue);
    color: #fff;
    border: none;
    padding: 10px 22px;
    border-radius: 6px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
  }
  button:hover { background: var(--navy); }
  .result {
    margin-top: 26px;
    padding: 18px 20px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .result.fatal { background: #fdecea; border-color: #f3c6c1; }
  .result.not-fatal { background: #eafaf0; border-color: #bfe8cf; }
  .result h3 {
    margin: 0 0 8px;
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #555;
  }
  .verdict {
    font-size: 24px;
    font-weight: 700;
  }
  .verdict.fatal { color: var(--red); }
  .verdict.not-fatal { color: var(--green); }
  .prob {
    margin-top: 6px;
    color: #444;
    font-size: 14px;
  }
</style>

<div class="card">
  <h2>KSI Fatality Risk Predictor</h2>
  <span class="model-tag">Model: {{ model_name }}</span>
  <p class="hint">Fill in what you know, leave the rest blank &mdash; the model fills in
  missing values the same way it was trained to.</p>

  <form method="POST" action="/predict_form">
    {% for name, type, hint in fields %}
    <div class="field">
      <label>{{ name }}<span class="sub">{{ hint }}</span></label>
      <input type="{{ type }}" name="{{ name }}" {% if type == 'number' %}step="any"{% endif %}>
    </div>
    {% endfor %}
    <button type="submit">Predict</button>
  </form>

  {% if result %}
  {% if result.warning %}
  <div class="result" style="background:#fff8e1;border-color:#f0d99a;margin-bottom:14px;">
    <h3 style="color:#8a6d00;">Check your input</h3>
    <div style="color:#6b5400;font-size:14px;">{{ result.warning }}</div>
  </div>
  {% endif %}
  <div class="result {{ 'fatal' if result.prediction == 'FATAL' else 'not-fatal' }}">
    <h3>Prediction</h3>
    <div class="verdict {{ 'fatal' if result.prediction == 'FATAL' else 'not-fatal' }}">{{ result.prediction }}</div>
    <div class="prob">Estimated probability of fatal outcome: {{ (result.probability * 100) | round(1) }}%</div>
  </div>

  {% if result.explanation %}
  <div class="result" style="margin-top:14px;">
    <h3>Why the model decided this</h3>
    <div style="font-size:13px;color:#666;margin-bottom:10px;">
      Each factor's actual weight in the model, sorted by how much it moved this specific prediction.
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
      {% for item in result.explanation %}
      <tr style="border-top:1px solid var(--border);">
        <td style="padding:6px 4px;">{{ item.feature }}</td>
        <td style="padding:6px 4px;color:{{ 'var(--red)' if item.direction == 'toward FATAL' else 'var(--green)' }};text-align:right;white-space:nowrap;">
          {{ item.direction }} ({{ item.weight }})
        </td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}
  {% endif %}
</div>
"""


def build_row(data: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with every column the model expects.
    Anything not provided becomes None -> the pipeline's own imputer
    (fit during training) fills it in, same as it does for real missing data.
    """
    row = {col: (data.get(col) or None) for col in ALL_COLUMNS}
    return pd.DataFrame([row])


# Real bounding box of Toronto collisions in the training data (with a little
# padding). LATITUDE/LONGITUDE are standardized against this narrow range, so
# a value even a few degrees outside it becomes an extreme outlier that can
# dominate the whole linear prediction and swing the result to a meaningless
# extreme, silently. Warn instead of failing silently.
LAT_RANGE = (43.0, 44.5)
LON_RANGE = (-80.5, -78.5)


def check_coord_warning(data: dict) -> str | None:
    try:
        lat = float(data.get("LATITUDE")) if data.get("LATITUDE") else None
    except (TypeError, ValueError):
        lat = None
    try:
        lon = float(data.get("LONGITUDE")) if data.get("LONGITUDE") else None
    except (TypeError, ValueError):
        lon = None

    problems = []
    if lat is not None and not (LAT_RANGE[0] <= lat <= LAT_RANGE[1]):
        problems.append(f"LATITUDE {lat} is outside Toronto's range ({LAT_RANGE[0]}-{LAT_RANGE[1]})")
    if lon is not None and not (LON_RANGE[0] <= lon <= LON_RANGE[1]):
        problems.append(f"LONGITUDE {lon} is outside Toronto's range ({LON_RANGE[0]} to {LON_RANGE[1]}, note the minus sign)")
    if problems:
        return "; ".join(problems) + ". The prediction below is unreliable — the model was only trained on Toronto-area coordinates."
    return None


def explain_prediction(df: pd.DataFrame, top_n: int = 6) -> list[dict]:
    """Break down which features actually pushed this specific prediction
    toward FATAL vs NOT FATAL - not a separate model or an LLM guessing,
    this is the real Logistic Regression math: each input feature has a
    learned coefficient, and contribution = coefficient * (this input's
    standardized value). Sorted by magnitude, so the top entries are what
    actually decided this prediction.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []  # only meaningful for linear models

    feature_names = preprocessor.get_feature_names_out()
    x_transformed = preprocessor.transform(df)
    if hasattr(x_transformed, "toarray"):
        x_transformed = x_transformed.toarray()
    x_row = np.asarray(x_transformed)[0]
    coefs = clf.coef_[0]
    contributions = coefs * x_row

    order = np.argsort(-np.abs(contributions))
    breakdown = []
    for i in order:
        if abs(contributions[i]) < 1e-6:
            continue
        name = feature_names[i]
        if "__" in name:
            name = name.split("__", 1)[1]
        breakdown.append({
            "feature": name,
            "direction": "toward FATAL" if contributions[i] > 0 else "toward NOT FATAL",
            "weight": round(float(contributions[i]), 3),
        })
        if len(breakdown) >= top_n:
            break
    return breakdown


def run_prediction(data: dict) -> dict:
    df = build_row(data)
    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][1]
    return {
        "prediction": "FATAL" if pred == 1 else "NOT FATAL",
        "probability": round(float(proba), 3),
        "warning": check_coord_warning(data),
        "explanation": explain_prediction(df),
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
    app.run(debug=False, port=5000)
