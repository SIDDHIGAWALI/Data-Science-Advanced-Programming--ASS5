"""Train and compare classical ML, Bayesian, discriminant and neural-network models.

Outputs
  models/best_model.joblib   best pipeline (preprocessing + estimator) by test ROC-AUC
  models/metrics.json        comparison table, feature importance, confusion matrix
  data/hr_scored.csv         every employee with predicted probability + risk category
  reports/*.png              charts used in the technical report
"""
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CATEGORICAL, DATA_FILE, FEATURES, METRICS_FILE, MODEL_DIR, MODEL_FILE,
                    NUMERIC, REPORT_DIR, SCORED_FILE, TARGET, risk_category)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET].str.lower() == "yes").astype(int)
    return df


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def candidates() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.5,
                                              early_stopping=False, max_iter=300, random_state=SEED),
        "Naive Bayes (Bayesian)": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.6),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = load()
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted, probs = [], {}, {}
    for name, est in candidates().items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", est)])
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        fit_s = time.time() - t0
        p = pipe.predict_proba(Xte)[:, 1]
        # threshold chosen on the training folds' prior (attrition base-rate-aware) -> use 0.5 for
        # balanced models, base rate for others, so F1 is comparable
        thr = 0.5 if name in ("Logistic Regression", "Random Forest") else float(ytr.mean())
        pred = (p >= thr).astype(int)
        rows.append({
            "model": name,
            "cv_roc_auc": round(float(cv_auc), 4),
            "test_roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, pred)), 4),
            "f1": round(float(f1_score(yte, pred)), 4),
            "threshold": round(thr, 3),
            "fit_seconds": round(fit_s, 2),
        })
        fitted[name], probs[name] = pipe, p
        print(f"{name:26s} cvAUC={cv_auc:.3f} testAUC={rows[-1]['test_roc_auc']:.3f} F1={rows[-1]['f1']:.3f}")

    table = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    best_name = table.iloc[0]["model"]
    best = fitted[best_name]
    best_thr = float(table.iloc[0]["threshold"])
    print(f"\nBest model: {best_name}")

    # Permutation importance on held-out data (model-agnostic explainability)
    imp = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).head(12))

    pred_best = (probs[best_name] >= best_thr).astype(int)
    cm = confusion_matrix(yte, pred_best).tolist()

    # Latency benchmark for the single-record scoring path (what the REST API does)
    one = Xte.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(200):
        best.predict_proba(one)
    latency_ms = (time.perf_counter() - t0) / 200 * 1000

    joblib.dump({"pipeline": best, "name": best_name, "threshold": best_thr, "features": FEATURES},
                MODEL_FILE)
    METRICS_FILE.write_text(json.dumps({
        "best_model": best_name,
        "threshold": best_thr,
        "comparison": table.to_dict("records"),
        "feature_importance": importance.round(4).to_dict("records"),
        "confusion_matrix": cm,
        "latency_ms_single_record": round(latency_ms, 2),
        "train_rows": len(Xtr), "test_rows": len(Xte),
    }, indent=2))

    # Score every employee (the dashboard reads this file)
    scored = df.copy()
    scored["AttritionProbability"] = best.predict_proba(df[FEATURES])[:, 1].round(4)
    scored["RiskCategory"] = scored["AttritionProbability"].map(risk_category)
    scored.to_csv(SCORED_FILE, index=False)

    # ---- charts ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(yte, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (hold-out)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(importance.feature[::-1], importance.importance[::-1], color="#1f5fa8")
    ax.set(xlabel="Drop in ROC-AUC when shuffled", title=f"Top drivers ({best_name})")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nSaved model, metrics, scored data and charts.")


if __name__ == "__main__":
    main()
