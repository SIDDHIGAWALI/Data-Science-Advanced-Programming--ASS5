"""Shared paths and feature definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_FILE = DATA_DIR / "hr_attrition.csv"
MODEL_FILE = MODEL_DIR / "best_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
SCORED_FILE = DATA_DIR / "hr_scored.csv"

TARGET = "Attrition"

NUMERIC = [
    "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
    "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "JobInvolvement", "Education",
]
CATEGORICAL = [
    "Department", "JobRole", "Gender", "MaritalStatus", "BusinessTravel",
    "OverTime", "EducationField",
]
FEATURES = NUMERIC + CATEGORICAL

# Risk bands applied to predicted attrition probability
RISK_BANDS = [(0.0, 0.30, "Low"), (0.30, 0.60, "Medium"), (0.60, 1.01, "High")]


def risk_category(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "High"
