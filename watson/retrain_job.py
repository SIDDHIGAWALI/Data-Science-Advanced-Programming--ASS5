"""Entry point for a Watson Studio *Job*: retrain, compare, redeploy (new version) automatically.

Create the job in Watson Studio:  Project -> Assets -> New asset -> Code script/Notebook job
  - Upload this repo (or src/ + watson/) as project assets, select this script
  - Schedule: e.g. weekly, Monday 02:00  (Job -> Schedule)
  - Add job env vars WML_URL, WML_API_KEY, WML_SPACE_ID
It retrains (src/train.py), and only updates the deployment if the new test AUC >= MIN_AUC.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_AUC = float(os.getenv("MIN_AUC", "0.70"))

subprocess.check_call([sys.executable, str(ROOT / "src" / "train.py")])
m = json.loads((ROOT / "models" / "metrics.json").read_text())
auc = m["comparison"][0]["test_roc_auc"]
print(f"New best model: {m['best_model']}  test AUC={auc}")
if auc < MIN_AUC:
    sys.exit(f"AUC {auc} below gate {MIN_AUC}; deployment NOT updated")

subprocess.check_call([sys.executable, str(ROOT / "watson" / "deploy_to_watson.py")])
print("Redeployed.")
