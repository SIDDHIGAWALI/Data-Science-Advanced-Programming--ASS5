"""Call the deployed Watson ML endpoint with one sample employee.   python watson/score_watson.py"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_FILE, FEATURES  # noqa: E402
from ibm_watsonx_ai import APIClient, Credentials  # noqa: E402

client = APIClient(Credentials(url=os.environ["WML_URL"], api_key=os.environ["WML_API_KEY"]))
client.set.default_space(os.environ["WML_SPACE_ID"])
dep_id = Path(__file__).with_name("deployment_id.txt").read_text().strip()

row = pd.read_csv(DATA_FILE)[FEATURES].iloc[[0]]
payload = {"input_data": [{"fields": FEATURES, "values": row.values.tolist()}]}
print(client.deployments.score(dep_id, payload))
