"""REST scoring API (Flask).

Run:   python src/api.py            -> http://127.0.0.1:5000
Docs:  docs/API.md
Mirrors the request/response shape of a Watson Machine Learning online deployment so the same
client code works against either endpoint.
"""
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from config import FEATURES, MODEL_FILE, risk_category

app = Flask(__name__)
_bundle = joblib.load(MODEL_FILE)
_pipe, _thr = _bundle["pipeline"], _bundle["threshold"]


def _score(records: list[dict]) -> list[dict]:
    df = pd.DataFrame(records)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    p = _pipe.predict_proba(df[FEATURES])[:, 1]
    return [{"attrition_probability": round(float(x), 4),
             "prediction": "Yes" if x >= _thr else "No",
             "risk_category": risk_category(float(x))} for x in p]


@app.get("/health")
def health():
    return jsonify(status="ok", model=_bundle["name"], threshold=_thr)


@app.post("/predict")
def predict():
    t0 = time.perf_counter()
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(error="Body must be valid JSON"), 400
    records = body.get("records") if isinstance(body, dict) and "records" in body else body
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        return jsonify(error="Send a JSON object or {'records': [...]}"), 400
    try:
        preds = _score(records)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # unseen types etc.
        return jsonify(error=f"Scoring failed: {e}"), 422
    return jsonify(predictions=preds, model=_bundle["name"],
                   latency_ms=round((time.perf_counter() - t0) * 1000, 2)), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
