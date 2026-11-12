"""
Intelligent Credit Processing - simple base project
---------------------------------------------------
Pipeline:
  1. Load data (your CSV, or generate synthetic sample data)
  2. Preprocess + train a default-risk model
  3. Evaluate the model
  4. Convert default probability -> credit score (300-850)
  5. Decision engine: APPROVE / MANUAL REVIEW / REJECT, with reasons

Run:
  pip install pandas numpy scikit-learn joblib
  python credit_processing.py                 # train + demo
  python credit_processing.py --data my.csv   # use your own data

Your CSV needs these columns (target: 'default' = 1 if the borrower defaulted):
  age, annual_income, employment_years, loan_amount, loan_term_months,
  credit_history_years, existing_debt, num_late_payments, num_credit_lines,
  home_ownership (RENT/OWN/MORTGAGE), default
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "credit_model.joblib"

NUMERIC = ["age", "annual_income", "employment_years", "loan_amount",
           "loan_term_months", "credit_history_years", "existing_debt",
           "num_late_payments", "num_credit_lines"]
CATEGORICAL = ["home_ownership"]
TARGET = "default"


# ----------------------------------------------------------------- data ----
def generate_synthetic_data(n=5000, seed=42):
    """Create realistic-looking sample applications with a hidden risk rule."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(21, 70, n),
        "annual_income": rng.lognormal(10.8, 0.5, n).round(0),
        "employment_years": rng.integers(0, 35, n),
        "loan_amount": rng.lognormal(9.5, 0.7, n).round(0),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60], n),
        "credit_history_years": rng.integers(0, 30, n),
        "existing_debt": rng.lognormal(9.0, 1.0, n).round(0),
        "num_late_payments": rng.poisson(1.2, n),
        "num_credit_lines": rng.integers(0, 12, n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[.45, .2, .35]),
    })
    dti = (df.existing_debt + df.loan_amount) / df.annual_income
    risk = (
        1.8 * dti
        + 0.45 * df.num_late_payments
        - 0.05 * df.employment_years
        - 0.04 * df.credit_history_years
        + 0.3 * (df.home_ownership == "RENT")
        + rng.normal(0, 0.6, n)
        - 0.8
    )
    df[TARGET] = (1 / (1 + np.exp(-risk)) > rng.random(n)).astype(int)
    return df


def load_data(path=None):
    if path:
        return pd.read_csv(path)
    print("No dataset supplied -> generating synthetic data.")
    return generate_synthetic_data()


# ---------------------------------------------------------------- model ----
def build_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])


def train(df):
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_pipeline().fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    print(f"\nROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(classification_report(y_te, (proba > 0.5).astype(int),
                                target_names=["repaid", "default"]))
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return pipe


# ------------------------------------------------------------- decision ----
def probability_to_score(p_default):
    """Map default probability to a 300-850 credit score (higher = safer)."""
    return int(round(300 + (1 - p_default) * 550))


def business_rules(app):
    """Hard rules that override the model. Returns list of reject reasons."""
    reasons = []
    if app["age"] < 18:
        reasons.append("Applicant must be at least 18")
    if app["annual_income"] <= 0:
        reasons.append("No verifiable income")
    dti = (app["existing_debt"] + app["loan_amount"]) / max(app["annual_income"], 1)
    if dti > 4:
        reasons.append(f"Debt-to-income ratio too high ({dti:.1f}x income)")
    if app["num_late_payments"] > 8:
        reasons.append("Excessive late payment history")
    return reasons


def explain(app):
    """Plain-language risk factors for the decision."""
    notes = []
    dti = (app["existing_debt"] + app["loan_amount"]) / max(app["annual_income"], 1)
    if dti > 2:
        notes.append(f"High debt-to-income ({dti:.1f}x)")
    if app["num_late_payments"] >= 3:
        notes.append(f"{app['num_late_payments']} late payments on record")
    if app["credit_history_years"] < 2:
        notes.append("Short credit history")
    if app["employment_years"] < 1:
        notes.append("Limited employment history")
    return notes or ["No major risk factors"]


def process_application(app, pipe, approve_below=0.25, reject_above=0.55):
    """Full decision for one application (dict)."""
    hard_fail = business_rules(app)
    if hard_fail:
        return {"decision": "REJECT", "score": None,
                "default_probability": None, "reasons": hard_fail}

    p = float(pipe.predict_proba(pd.DataFrame([app]))[0, 1])
    if p < approve_below:
        decision = "APPROVE"
    elif p > reject_above:
        decision = "REJECT"
    else:
        decision = "MANUAL REVIEW"

    return {"decision": decision, "score": probability_to_score(p),
            "default_probability": round(p, 3), "reasons": explain(app)}


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to CSV training data")
    args = ap.parse_args()

    pipe = train(load_data(args.data))

    demo_applicants = [
        {"age": 38, "annual_income": 85000, "employment_years": 12,
         "loan_amount": 15000, "loan_term_months": 36, "credit_history_years": 14,
         "existing_debt": 6000, "num_late_payments": 0, "num_credit_lines": 5,
         "home_ownership": "MORTGAGE"},
        {"age": 24, "annual_income": 28000, "employment_years": 1,
         "loan_amount": 30000, "loan_term_months": 60, "credit_history_years": 1,
         "existing_debt": 12000, "num_late_payments": 4, "num_credit_lines": 2,
         "home_ownership": "RENT"},
        {"age": 45, "annual_income": 52000, "employment_years": 8,
         "loan_amount": 20000, "loan_term_months": 48, "credit_history_years": 9,
         "existing_debt": 15000, "num_late_payments": 2, "num_credit_lines": 4,
         "home_ownership": "RENT"},
    ]

    print("\n=== Demo applications ===")
    for i, app in enumerate(demo_applicants, 1):
        r = process_application(app, pipe)
        print(f"\nApplicant {i}: {r['decision']}")
        print(f"  Score: {r['score']} | P(default): {r['default_probability']}")
        print(f"  Notes: {'; '.join(r['reasons'])}")


if __name__ == "__main__":
    main()
