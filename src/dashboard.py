"""HR Analytics dashboard (Streamlit).   Run: streamlit run src/dashboard.py"""
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from config import FEATURES, METRICS_FILE, MODEL_FILE, SCORED_FILE, risk_category

st.set_page_config(page_title="HR Attrition Analytics", layout="wide")
st.title("Employee Attrition Analytics Platform")

df = pd.read_csv(SCORED_FILE)
metrics = json.loads(METRICS_FILE.read_text())
bundle = joblib.load(MODEL_FILE)
df["Left"] = df["Attrition"].map({"Yes": 1, "No": 0}) if df["Attrition"].dtype != "int64" else df["Attrition"]

# ---- sidebar filters ----
st.sidebar.header("Filters")
depts = st.sidebar.multiselect("Department", sorted(df.Department.unique()), default=sorted(df.Department.unique()))
risk = st.sidebar.multiselect("Risk category", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
f = df[df.Department.isin(depts) & df.RiskCategory.isin(risk)]

tab_kpi, tab_dept, tab_risk, tab_demo, tab_model, tab_predict = st.tabs(
    ["HR KPIs", "Attrition by department", "Risk categories", "Demographics", "Model performance", "Predict"])

with tab_kpi:
    attr = f.Left.mean() if len(f) else 0
    c = st.columns(4)
    c[0].metric("Headcount", f"{len(f):,}")
    c[1].metric("Attrition rate", f"{attr:.1%}")
    c[2].metric("Retention rate", f"{1 - attr:.1%}")
    c[3].metric("High-risk employees", f"{(f.RiskCategory == 'High').sum():,}")
    trend = f.groupby("YearsAtCompany").Left.mean().reset_index()
    st.plotly_chart(px.line(trend, x="YearsAtCompany", y="Left", markers=True,
                            title="Attrition rate by tenure (years at company)",
                            labels={"Left": "Attrition rate"}), width="stretch")
    ot = f.groupby("OverTime").Left.mean().reset_index()
    st.plotly_chart(px.bar(ot, x="OverTime", y="Left", title="Attrition rate: overtime vs not",
                           labels={"Left": "Attrition rate"}), width="stretch")

with tab_dept:
    g = f.groupby("Department").agg(headcount=("Left", "size"), leavers=("Left", "sum"),
                                    attrition_rate=("Left", "mean")).reset_index()
    g["retention_rate"] = 1 - g.attrition_rate
    st.plotly_chart(px.bar(g, x="Department", y="attrition_rate", color="Department", text_auto=".1%",
                           title="Attrition rate by department"), width="stretch")
    st.dataframe(g.style.format({"attrition_rate": "{:.1%}", "retention_rate": "{:.1%}"}), width="stretch")
    jr = f.groupby("JobRole").Left.mean().sort_values().reset_index()
    st.plotly_chart(px.bar(jr, x="Left", y="JobRole", orientation="h", title="Attrition rate by job role",
                           labels={"Left": "Attrition rate"}), width="stretch")

with tab_risk:
    order = ["Low", "Medium", "High"]
    cnt = f.RiskCategory.value_counts().reindex(order).fillna(0).reset_index()
    cnt.columns = ["RiskCategory", "employees"]
    a, b = st.columns(2)
    a.plotly_chart(px.pie(cnt, names="RiskCategory", values="employees", hole=0.45, title="Employees by risk band",
                          color="RiskCategory", color_discrete_map={"Low": "#2e9e5b", "Medium": "#e0a21b", "High": "#c0392b"}),
                   width="stretch")
    b.plotly_chart(px.histogram(f, x="AttritionProbability", nbins=30, title="Distribution of predicted probability"),
                   width="stretch")
    st.subheader("Top 25 highest-risk employees")
    st.dataframe(f.sort_values("AttritionProbability", ascending=False)
                 [["EmployeeID", "Department", "JobRole", "OverTime", "MonthlyIncome", "AttritionProbability", "RiskCategory"]]
                 .head(25), width="stretch")

with tab_demo:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="Age", nbins=20, title="Age distribution"), width="stretch")
    b.plotly_chart(px.pie(f, names="Gender", title="Gender split"), width="stretch")
    a.plotly_chart(px.pie(f, names="MaritalStatus", title="Marital status"), width="stretch")
    b.plotly_chart(px.box(f, x="Department", y="MonthlyIncome", title="Monthly income by department"),
                   width="stretch")

with tab_model:
    st.write(f"**Deployed model:** {metrics['best_model']}  |  decision threshold {metrics['threshold']}  |  "
             f"single-record latency ≈ {metrics['latency_ms_single_record']} ms")
    comp = pd.DataFrame(metrics["comparison"])
    st.dataframe(comp, width="stretch")
    st.plotly_chart(px.bar(comp, x="model", y=["cv_roc_auc", "test_roc_auc"], barmode="group",
                           title="ROC-AUC by model"), width="stretch")
    imp = pd.DataFrame(metrics["feature_importance"])
    st.plotly_chart(px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h",
                           title="Top attrition drivers (permutation importance)"), width="stretch")

with tab_predict:
    st.write("Enter an employee profile to get a live attrition prediction from the deployed model.")
    base = df.iloc[0]
    vals = {}
    cols = st.columns(3)
    for i, col in enumerate(FEATURES):
        with cols[i % 3]:
            if df[col].dtype == "object" or str(df[col].dtype).startswith("str"):
                opts = sorted(df[col].unique())
                vals[col] = st.selectbox(col, opts, index=opts.index(base[col]))
            else:
                vals[col] = st.number_input(col, value=int(base[col]), step=1)
    if st.button("Predict"):
        p = float(bundle["pipeline"].predict_proba(pd.DataFrame([vals])[FEATURES])[0, 1])
        st.metric("Attrition probability", f"{p:.1%}", risk_category(p) + " risk")
