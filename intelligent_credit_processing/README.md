# Intelligent Credit Processing

A simple ML-based credit approval system. It trains a default-risk model,
converts risk to a 300-850 credit score, applies business rules, and returns
APPROVE / MANUAL REVIEW / REJECT with reasons.

## Setup
    pip install -r requirements.txt

## Run
    python credit_processing.py                  # uses synthetic data
    python credit_processing.py --data data.csv  # uses your own CSV

## CSV columns
age, annual_income, employment_years, loan_amount, loan_term_months,
credit_history_years, existing_debt, num_late_payments, num_credit_lines,
home_ownership (RENT/OWN/MORTGAGE), default (1 = defaulted, 0 = repaid)

## Files
- credit_processing.py : data, training, scoring, decision engine
- requirements.txt     : dependencies
