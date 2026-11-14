# Employee Attrition Analytics Platform

End-to-end HR analytics: model comparison (classical ML vs Bayesian vs discriminant analysis vs neural
network), REST scoring API, IBM Watson ML deployment + scheduled retraining job, interactive Streamlit
dashboard, technical report and executive deck.

## Quick start
Windows: `run_all.bat`   macOS/Linux: `./run_all.sh`   (creates venv, installs, generates data, trains, tests, builds deck)

Then, in two terminals (venv active):
```
python src/api.py                  # REST API  -> http://127.0.0.1:5000
streamlit run src/dashboard.py     # dashboard -> http://localhost:8501
```
Manual steps: `pip install -r requirements.txt` → `python src/generate_data.py` → `python src/train.py`
→ `python -m pytest -q tests` → `python src/build_deck.py`.

## Layout
| Path | What |
|---|---|
| src/generate_data.py | synthetic IBM-HR-style data (drop the real IBM HR CSV at data/hr_attrition.csv to use it instead) |
| src/train.py | trains 7 models, picks best by test ROC-AUC, writes models/, reports/ |
| src/api.py | Flask REST API (docs/API.md) |
| src/dashboard.py | Streamlit dashboard (KPIs, departments, risk, demographics, model metrics, live predict) |
| watson/ | deploy_to_watson.py, score_watson.py, retrain_job.py (Watson Studio Job) |
| reports/Technical_Report.md, Executive_Presentation.pptx | written deliverables |

## IBM Watson deployment
`pip install ibm-watsonx-ai`, set `WML_URL`, `WML_API_KEY`, `WML_SPACE_ID`, then
`python watson/deploy_to_watson.py` and `python watson/score_watson.py`. Schedule
`watson/retrain_job.py` as a Watson Studio Job. These need your IBM Cloud account and were not run here.
