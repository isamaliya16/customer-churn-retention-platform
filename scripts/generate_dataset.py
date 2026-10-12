"""Generate a large, realistic customer dataset (default: 100,000 customers) for the platform.

HOW IT IS MADE (honest description - this is SYNTHETIC data):
  1. Start from the real IBM/Kaggle Telco Customer Churn file (7,043 customers).
  2. Bootstrap-resample customers (with replacement) so the joint structure of services, contracts, payment methods
     and the real churn labels is preserved, then add small random noise to tenure and monthly charges so rows are
     not exact copies.
  3. Add what the raw file lacks but a real CRM has: names, e-mails, signup dates and churn dates (churners left
     during the last 24 months, with a mild upward trend), and ~1% missing values in a few fields.

Because the rows are resampled from the 7,043 real customers, a model trained on the real file has "seen" this data.
NEVER use this file to measure model accuracy - use the held-out metrics on the Model page. It exists to exercise the
platform at realistic scale (import, scoring, dashboards, segmentation) and to demo it.

Usage:
    python scripts/generate_dataset.py                       # 100,000 rows -> data/sample/customers_100k.csv
    python scripts/generate_dataset.py --rows 250000 --out /tmp/big.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw", "telco_customer_churn.csv")
DEFAULT_OUT = os.path.join(ROOT, "data", "sample", "customers_100k.csv")

FIRST = (
    "Aarav Aditi Aisha Akash Amit Ananya Anil Anjali Arjun Aryan Asha Bhavna Chirag Deepak Diya Farhan Gauri Harsh "
    "Isha Ishaan Jatin Kabir Kavya Kiran Krish Lakshmi Manish Meera Mohan Naina Neha Nikhil Nisha Om Pooja Pranav "
    "Priya Rahul Raj Rhea Rohan Ruchi Sagar Sana Sanjay Shreya Siddharth Sneha Tanvi Uday Vikram Vivek Yash Zoya "
    "Olivia Liam Emma Noah Ava Elijah Sophia Lucas Mia Mason Amelia Logan Harper Ethan Evelyn James Abigail Henry "
    "Emily Jack Ella Owen Grace Leo Chloe Wyatt Lily Julian Nora Isaac Hannah Caleb Zoe Ryan Maya Daniel Sara "
    "Carlos Lucia Mateo Valentina Diego Camila Mohammed Fatima Omar Layla Wei Mei Hiro Yuki Kenji Sakura Ivan Anya"
).split()
LAST = (
    "Patel Shah Mehta Desai Joshi Sharma Verma Gupta Singh Kumar Iyer Nair Reddy Rao Menon Bose Das Ghosh Kapoor "
    "Malhotra Chopra Bhatt Trivedi Pandya Jadeja Solanki Parmar Thakkar Modi Amin Dave Vyas Smith Johnson Brown "
    "Williams Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Wilson Anderson Thomas Taylor Moore "
    "Jackson Martin Lee Perez Thompson White Harris Clark Lewis Robinson Walker Young Allen King Wright Scott "
    "Green Baker Adams Nelson Hill Campbell Mitchell Roberts Carter Phillips Evans Turner Torres Parker Collins "
    "Edwards Stewart Morris Murphy Cook Rogers Morgan Cooper Peterson Reed Bailey Bell Kelly Howard Ward Cox "
    "Khan Ali Hussain Ahmed Tanaka Sato Suzuki Chen Wang Liu Zhang Kim Park Nguyen Ivanov Petrov Silva Santos"
).split()

PROFILE_COLUMNS = {
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
}
CONTRACT = {"Month-to-month": "month-to-month", "One year": "one-year", "Two year": "two-year"}
# Fields where a real CRM export is often incomplete (~1% blanks): the model's imputers must cope.
SPARSE_FIELDS = ["payment_method", "tech_support", "online_security", "paperless_billing"]
MONTH_DAYS = 30.4375


def build(rows: int, as_of: date, seed: int = 2026, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = pd.read_csv(RAW)
    real["TotalCharges"] = pd.to_numeric(real["TotalCharges"], errors="coerce")
    base = real.iloc[rng.integers(0, len(real), size=rows)].reset_index(drop=True)

    churned = (base["Churn"] == "Yes").to_numpy()
    tenure = base["tenure"].to_numpy().astype(int)
    jitter = np.rint(rng.normal(0, 2.0, size=rows)).astype(int)
    tenure = np.clip(tenure + jitter, 0, 72)
    tenure = np.where(churned, np.maximum(tenure, 1), tenure)  # a churner was a customer for at least a month

    monthly = base["MonthlyCharges"].to_numpy() * np.exp(rng.normal(0, 0.04, size=rows))
    monthly = np.round(np.clip(monthly, 18.25, 125.0), 2)
    total = np.round(tenure * monthly * rng.uniform(0.97, 1.03, size=rows), 2)
    total = np.where(tenure == 0, 0.0, total)

    # Churn dates: spread over the last `months` months, with a mild upward trend (recent months slightly heavier).
    month_weights = np.linspace(0.8, 1.2, months)
    month_weights /= month_weights.sum()
    month_back = rng.choice(np.arange(months)[::-1], size=rows, p=month_weights)  # 0 = most recent month
    days_back = np.rint(month_back * MONTH_DAYS + rng.uniform(1, MONTH_DAYS, size=rows)).astype(int)
    churn_day = np.array([np.datetime64(as_of) - np.timedelta64(int(d), "D") for d in days_back])
    ref = np.where(churned, churn_day, np.datetime64(as_of))
    signup = ref - np.rint(tenure * MONTH_DAYS).astype("timedelta64[D]")

    first = rng.choice(FIRST, size=rows)
    last = rng.choice(LAST, size=rows)
    ids = np.arange(1, rows + 1)
    out = pd.DataFrame(
        {
            "external_id": [f"CUST-{i:06d}" for i in ids],
            "name": [f"{f} {l}" for f, l in zip(first, last)],
            "email": [f"{f.lower()}.{l.lower()}.{i}@example.com" for f, l, i in zip(first, last, ids)],
            "signup_date": pd.to_datetime(signup).strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract_type": base["Contract"].map(CONTRACT).to_numpy(),
            "plan": base["InternetService"].to_numpy(),
            "is_active": np.where(churned, "false", "true"),
            "churned_at": np.where(churned, pd.to_datetime(churn_day).strftime("%Y-%m-%d"), ""),
        }
    )
    for raw_name, col in PROFILE_COLUMNS.items():
        out[col] = base[raw_name].to_numpy()
    for col in SPARSE_FIELDS:
        out.loc[rng.random(rows) < 0.01, col] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--as-of", default=date.today().isoformat(), help="'today' for the data (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    df = build(args.rows, date.fromisoformat(args.as_of), args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    churn = (df["is_active"] == "false").mean()
    print(f"Wrote {len(df):,} customers to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), churn {churn:.2%}")


if __name__ == "__main__":
    main()
