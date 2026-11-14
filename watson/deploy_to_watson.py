"""Deploy the trained pipeline to IBM Watson Machine Learning (online scoring endpoint).

Prereqs:  pip install ibm-watsonx-ai
          export WML_URL=https://us-south.ml.cloud.ibm.com   (your region)
          export WML_API_KEY=<IBM Cloud API key>
          export WML_SPACE_ID=<deployment space id>
Optional: WML_SOFTWARE_SPEC (default runtime-24.1-py3.11), WML_MODEL_TYPE (default scikit-learn_1.3)
          -> pick the model type/runtime whose scikit-learn version matches `pip show scikit-learn`
             (list with client.software_specifications.list()).
Run:      python watson/deploy_to_watson.py
"""
import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MODEL_FILE  # noqa: E402

try:
    from ibm_watsonx_ai import APIClient, Credentials
except ImportError:
    sys.exit("Install the SDK first:  pip install ibm-watsonx-ai")

for var in ("WML_URL", "WML_API_KEY", "WML_SPACE_ID"):
    if not os.getenv(var):
        sys.exit(f"Set environment variable {var}")

client = APIClient(Credentials(url=os.environ["WML_URL"], api_key=os.environ["WML_API_KEY"]))
client.set.default_space(os.environ["WML_SPACE_ID"])

bundle = joblib.load(MODEL_FILE)
spec = client.software_specifications.get_id_by_name(os.getenv("WML_SOFTWARE_SPEC", "runtime-24.1-py3.11"))
M = client.repository.ModelMetaNames
model = client.repository.store_model(
    model=bundle["pipeline"],
    meta_props={
        M.NAME: "attrition-risk-model",
        M.TYPE: os.getenv("WML_MODEL_TYPE", "scikit-learn_1.3"),
        M.SOFTWARE_SPEC_ID: spec,
    },
)
model_id = client.repository.get_model_id(model)
print("Stored model:", model_id)

D = client.deployments.ConfigurationMetaNames
dep = client.deployments.create(model_id, meta_props={D.NAME: "attrition-online", D.ONLINE: {}})
dep_id = client.deployments.get_id(dep)
print("Deployment id:", dep_id)
print("Scoring URL  :", client.deployments.get_scoring_href(dep))
Path(__file__).with_name("deployment_id.txt").write_text(dep_id)
