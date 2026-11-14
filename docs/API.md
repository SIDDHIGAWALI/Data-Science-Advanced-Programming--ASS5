# REST API documentation

Local server: `python src/api.py` → `http://127.0.0.1:5000`.
The same payload shape is used for the Watson ML endpoint (see `watson/`).

## GET /health
`200 {"status":"ok","model":"Logistic Regression","threshold":0.5}`

## POST /predict
Body: one employee object, or `{"records":[ {...}, {...} ]}` for a batch.

Required fields (all): Age, DistanceFromHome, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike,
TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion, YearsWithCurrManager,
JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement, Education (numbers) and
Department, JobRole, Gender, MaritalStatus, BusinessTravel, OverTime, EducationField (strings).

### Sample request
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
 "Age":29,"DistanceFromHome":18,"MonthlyIncome":3200,"NumCompaniesWorked":4,"PercentSalaryHike":12,
 "TotalWorkingYears":7,"TrainingTimesLastYear":2,"YearsAtCompany":2,"YearsSinceLastPromotion":2,
 "YearsWithCurrManager":1,"JobSatisfaction":1,"EnvironmentSatisfaction":2,"WorkLifeBalance":2,
 "JobInvolvement":2,"Education":3,"Department":"Sales","JobRole":"Sales Executive","Gender":"Male",
 "MaritalStatus":"Single","BusinessTravel":"Travel_Frequently","OverTime":"Yes","EducationField":"Marketing"}'
```
### Sample response (200)
```json
{"model":"Logistic Regression","latency_ms":4.1,
 "predictions":[{"attrition_probability":0.86,"prediction":"Yes","risk_category":"High"}]}
```
Risk bands: Low < 0.30 ≤ Medium < 0.60 ≤ High.

## Status codes
| Code | Meaning |
|---|---|
| 200 | Scored successfully |
| 400 | Body is not JSON, is empty, or required fields are missing (message lists them) |
| 422 | Fields present but values unusable (e.g. text in a numeric field) |
| 500 | Unexpected server error |

## Watson ML equivalent
`POST {scoring_url}?version=2021-05-01` with header `Authorization: Bearer <IAM token>` and body
`{"input_data":[{"fields":[...],"values":[[...]]}]}`. 401 = missing/expired token.
