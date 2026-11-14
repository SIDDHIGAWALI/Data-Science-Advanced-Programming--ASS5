# Technical Report — Employee Attrition Analytics Platform

## 1. Objective
Predict which employees are likely to leave, explain the drivers, deploy the model behind a REST API (IBM Watson Machine Learning) and expose results in an HR dashboard.

## 2. Data
2,000 records, 22 predictors (demographics, pay, tenure, satisfaction, overtime, travel) and a binary Attrition target (~15% positive). The data is **synthetic**, generated to follow the IBM HR Analytics schema (`src/generate_data.py`); the real IBM HR CSV can be dropped into `data/hr_attrition.csv` and the whole pipeline re-run unchanged. Because the generator uses a logistic relationship, results below demonstrate the method, not real-world HR findings.

## 3. Method
Stratified 75/25 train/test split; one-hot encoding of categoricals and standardisation of numerics inside a single scikit-learn Pipeline (no leakage); 5-fold stratified CV on the training set; test ROC-AUC selects the deployed model.

Models compared:
- **Classical ML:** logistic regression, random forest, gradient boosting.
- **Bayesian:** Gaussian Naive Bayes — applies Bayes' theorem P(class|x) ∝ P(x|class)·P(class) assuming conditionally independent features.
- **Discriminant analysis:** LDA (shared covariance → linear boundary) and QDA (class-specific covariance → quadratic boundary, regularised with reg_param=0.6 because one-hot features are collinear). Appropriate here because the target is a two-class problem with mostly continuous/ordinal predictors; QDA's extra flexibility did not help.
- **Neural network:** MLP (32-16 hidden units, ReLU, L2 alpha=0.5).

## 4. Results
| model | cv_roc_auc | test_roc_auc | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7758 | 0.7929 | 0.74 | 0.3291 | 0.6842 | 0.4444 |
| LDA | 0.7765 | 0.7913 | 0.744 | 0.3354 | 0.6974 | 0.453 |
| QDA | 0.7578 | 0.7707 | 0.714 | 0.3064 | 0.6974 | 0.4257 |
| Random Forest | 0.7542 | 0.7684 | 0.822 | 0.403 | 0.3553 | 0.3776 |
| Gradient Boosting | 0.7502 | 0.7466 | 0.738 | 0.305 | 0.5658 | 0.3963 |
| Naive Bayes (Bayesian) | 0.7271 | 0.7432 | 0.688 | 0.2826 | 0.6842 | 0.4 |
| Neural Network (MLP) | 0.7083 | 0.6799 | 0.754 | 0.2673 | 0.3553 | 0.3051 |

Non-balanced models (LDA, QDA, GB, NB, MLP) use a decision threshold equal to the training base rate; LR and RF use class-weight balancing with threshold 0.5. ROC-AUC is the threshold-free comparison.

**Selected model: Logistic Regression** (test AUC 0.7929). Confusion matrix on 500 held-out employees [[TN, FP],[FN, TP]] = [[318, 106], [24, 52]]. Recall of 68% means most leavers are caught, at the cost of many false alarms (precision 33%) — acceptable if the follow-up action is a low-cost conversation.

**Neural network vs classical ML:** the MLP was the weakest model (AUC 0.6799). With 1,500 training rows and mostly tabular, near-linear signal, simpler models generalise better; MLP results also varied with settings, so they should be treated as indicative.

## 5. Explainability
Permutation importance on held-out data (drop in ROC-AUC when a feature is shuffled):
1. Age (0.051)
2. OverTime (0.046)
3. EnvironmentSatisfaction (0.030)
4. MonthlyIncome (0.026)
5. JobSatisfaction (0.020)
6. MaritalStatus (0.018)
7. DistanceFromHome (0.011)
8. WorkLifeBalance (0.006)

Overtime, age, satisfaction scores and income are the main drivers. SHAP values are listed as a future enhancement (`pip install shap`).

## 6. Deployment
- `src/api.py` — Flask API, `POST /predict` (single or batch JSON), `GET /health`; status codes 200/400/422. Single-record scoring latency ≈ 6.17 ms in-process (benchmark in `models/metrics.json`). See `docs/API.md`.
- `watson/deploy_to_watson.py` stores the pipeline in a Watson ML deployment space and creates an online deployment; `watson/score_watson.py` calls it with the Watson payload format.
- `watson/retrain_job.py` is the script scheduled as a **Watson Studio Job**: retrains, applies an AUC quality gate (default 0.70), then redeploys.
- **Status:** the Watson scripts were written to the ibm-watsonx-ai SDK but not executed (they need an IBM Cloud account and API key). Runtime/model-type names may need adjusting to your region.

## 7. Dashboard
Streamlit app (`src/dashboard.py`): HR KPIs (attrition and retention rate, tenure trend), attrition by department and role, risk categories (Low <30%, Medium 30–60%, High ≥60%), workforce demographics, model-performance metrics, and a live prediction form.

## 8. Limitations and next steps
Synthetic data; no fairness audit (gender/age) — required before real use; no drift monitoring yet; probability calibration not assessed. Next: SHAP, real HRIS data, calibration, bias metrics, monitoring in Watson OpenScale.
