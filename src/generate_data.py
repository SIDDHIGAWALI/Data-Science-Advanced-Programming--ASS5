"""Generate a synthetic HR dataset modelled on the public IBM HR Analytics schema.

Why synthetic?  The real IBM HR file is a Kaggle download that needs a login.  If you
have it, save it as data/hr_attrition.csv (columns must include config.FEATURES + Attrition
as Yes/No or 1/0) and skip this script - every other step works unchanged.
"""
import numpy as np
import pandas as pd

from config import DATA_FILE, DATA_DIR

ROLES = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician",
                               "Manufacturing Director", "Healthcare Representative"],
    "Human Resources": ["Human Resources", "Manager"],
}
DEPT_P = [0.30, 0.62, 0.08]
FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"]


def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dept = rng.choice(list(ROLES), n, p=DEPT_P)
    role = [rng.choice(ROLES[d]) for d in dept]
    age = np.clip(rng.normal(37, 9, n), 18, 60).round().astype(int)
    total_years = np.clip(age - 18 - rng.integers(0, 6, n), 0, 40)
    years_co = np.clip((total_years * rng.uniform(0.1, 0.8, n)).round(), 0, 35).astype(int)
    income = np.clip(2500 + total_years * 330 + rng.normal(0, 1800, n), 1100, 20000).round().astype(int)
    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n + 1),
        "Age": age,
        "Department": dept,
        "JobRole": role,
        "Gender": rng.choice(["Male", "Female"], n, p=[0.58, 0.42]),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22]),
        "BusinessTravel": rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[0.10, 0.70, 0.20]),
        "OverTime": rng.choice(["Yes", "No"], n, p=[0.28, 0.72]),
        "EducationField": rng.choice(FIELDS, n, p=[0.41, 0.31, 0.11, 0.09, 0.08]),
        "Education": rng.integers(1, 6, n),
        "DistanceFromHome": np.clip(rng.exponential(8, n).round(), 1, 29).astype(int),
        "MonthlyIncome": income,
        "NumCompaniesWorked": rng.integers(0, 10, n),
        "PercentSalaryHike": rng.integers(11, 26, n),
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": rng.integers(0, 7, n),
        "YearsAtCompany": years_co,
        "YearsSinceLastPromotion": np.minimum(rng.integers(0, 12, n), years_co),
        "YearsWithCurrManager": np.minimum(rng.integers(0, 15, n), years_co),
        "JobSatisfaction": rng.integers(1, 5, n),
        "EnvironmentSatisfaction": rng.integers(1, 5, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "JobInvolvement": rng.integers(1, 5, n),
    })
    # Latent log-odds of leaving: interpretable drivers + noise
    z = (-2.0
         + 1.3 * (df.OverTime == "Yes")
         + 0.9 * (df.MaritalStatus == "Single")
         + 0.6 * (df.BusinessTravel == "Travel_Frequently")
         + 0.5 * (df.Department == "Sales")
         + 0.04 * df.DistanceFromHome
         - 0.00022 * df.MonthlyIncome
         - 0.035 * df.Age
         - 0.04 * df.YearsAtCompany
         + 0.08 * df.YearsSinceLastPromotion
         + 0.10 * df.NumCompaniesWorked
         - 0.35 * (df.JobSatisfaction - 2.5)
         - 0.30 * (df.EnvironmentSatisfaction - 2.5)
         - 0.30 * (df.WorkLifeBalance - 2.5)
         - 0.25 * (df.JobInvolvement - 2.5)
         + 0.85
         + rng.normal(0, 0.7, n))
    p = 1 / (1 + np.exp(-z))
    df["Attrition"] = np.where(rng.random(n) < p, "Yes", "No")
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    d = generate()
    d.to_csv(DATA_FILE, index=False)
    print(f"Wrote {DATA_FILE}  rows={len(d)}  attrition_rate={(d.Attrition == 'Yes').mean():.1%}")
