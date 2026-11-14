import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd  # noqa: E402

from api import app  # noqa: E402
from config import DATA_FILE, FEATURES  # noqa: E402

client = app.test_client()
row = pd.read_csv(DATA_FILE)[FEATURES].iloc[0].to_dict()
row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def test_health():
    assert client.get("/health").status_code == 200


def test_predict_ok():
    r = client.post("/predict", json=row)
    assert r.status_code == 200
    p = r.get_json()["predictions"][0]
    assert 0 <= p["attrition_probability"] <= 1 and p["risk_category"] in {"Low", "Medium", "High"}


def test_predict_batch():
    r = client.post("/predict", json={"records": [row, row]})
    assert len(r.get_json()["predictions"]) == 2


def test_missing_field_400():
    bad = {k: v for k, v in row.items() if k != "Age"}
    assert client.post("/predict", json=bad).status_code == 400


def test_bad_json_400():
    assert client.post("/predict", data="nope", content_type="text/plain").status_code == 400
