"""Build reports/Executive_Presentation.pptx from models/metrics.json and the scored data."""
import json

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from config import METRICS_FILE, REPORT_DIR, SCORED_FILE

NAVY = RGBColor(0x14, 0x2B, 0x4F)
m = json.loads(METRICS_FILE.read_text())
d = pd.read_csv(SCORED_FILE)
rate = d.Attrition.mean()
dept = d.groupby("Department").Attrition.mean()
ot = d.groupby("OverTime").Attrition.mean()
high = int((d.RiskCategory == "High").sum())
best = m["comparison"][0]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def slide(title, bullets=None, image=None, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.1))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12), Inches(0.8))
    p = tb.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(30); p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    width = Inches(6.3) if image else Inches(12.2)
    if bullets:
        box = s.shapes.add_textbox(Inches(0.6), Inches(1.5), width, Inches(5.5)).text_frame
        box.word_wrap = True
        for i, b in enumerate(bullets):
            para = box.paragraphs[0] if i == 0 else box.add_paragraph()
            para.text = "• " + b; para.font.size = Pt(20); para.space_after = Pt(10)
    if image:
        s.shapes.add_picture(str(REPORT_DIR / image), Inches(7.2), Inches(1.5), width=Inches(5.8))
    s.notes_slide.notes_text_frame.text = notes
    return s


slide("Employee Attrition Analytics Platform",
      ["Predicting who is likely to leave — and why — so HR can act early",
       "Siddhi Gawali  |  B.Tech AI & Data Science, Sanjivani University"])
slide("The business problem",
      [f"{rate:.0%} of employees leave in the data analysed; every exit costs recruiting, onboarding and lost productivity",
       "HR learns about resignations after the decision is made",
       "Goal: flag at-risk employees early, understand the drivers, and target retention spend"])
slide("HR analytics workflow",
      ["Data: 2,000 employee records, 22 features (demographics, pay, tenure, satisfaction, overtime)",
       "Compare 7 models: logistic regression, random forest, gradient boosting, neural network, naive Bayes, LDA, QDA",
       "Deploy best model behind a REST API (IBM Watson Machine Learning); scheduled retraining via Watson Studio Jobs",
       "Interactive dashboard for HR: KPIs, departments, risk bands, live prediction"])
slide("Model comparison",
      [f"Best: {best['model']} — ROC-AUC {best['test_roc_auc']:.2f}, recall {best['recall']:.0%}",
       "Linear/Bayesian-family models matched or beat the neural network on this small tabular dataset",
       "Neural network: more flexible but needs more data to pay off; kept as a benchmark",
       f"Single-employee scoring takes ≈ {m['latency_ms_single_record']} ms"],
      image="roc_curves.png")
slide("Key insights",
      [f"Overtime is the clearest lever: {ot['Yes']:.0%} attrition with overtime vs {ot['No']:.0%} without",
       f"Sales has the highest department attrition ({dept['Sales']:.0%}) vs R&D {dept['Research & Development']:.0%}",
       "Other top drivers: age, environment and job satisfaction, income, marital status",
       f"{high} employees ({high / len(d):.0%}) currently fall in the High-risk band"],
      image="feature_importance.png")
slide("Business recommendations",
      ["Review overtime policy and workload balance, starting with Sales",
       "Run stay-interviews for the High-risk list each month",
       "Revisit pay for below-market roles and early-tenure staff; add promotion pathways",
       "Track satisfaction scores quarterly — they are leading indicators"])
slide("Expected organisational impact",
      ["Illustrative: if 10% of flagged leavers are retained, that is a direct reduction in attrition rate",
       "Savings scale with replacement cost (commonly estimated at 50–200% of annual salary)",
       "Targeted interventions cost less than blanket retention programmes",
       "Replace illustrative numbers with HR's actual replacement-cost data before presenting externally"])
slide("Future enhancements",
      ["Train on the organisation's real HRIS data and add fairness checks (gender, age) before use",
       "Add SHAP explanations to each prediction; drift monitoring and automated alerts",
       "Survey text sentiment, manager-level features",
       "Integrate with the HR portal via the scoring endpoint"])
prs.save(REPORT_DIR / "Executive_Presentation.pptx")
print("Saved", REPORT_DIR / "Executive_Presentation.pptx")
