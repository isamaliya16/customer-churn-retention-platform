"""Fetch (and cache) the public IBM/Kaggle Telco Customer Churn dataset.

Kaggle itself requires authenticated API access, so we pull a well-known
public GitHub mirror of the same file (same 7,043-row schema). If that's
unreachable (offline / network policy), we fall back to a synthetically
generated dataset with an identical column schema, so the rest of the
pipeline runs unmodified either way.

Run directly: `python -m app.ml.fetch_data`
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "telco_customer_churn.csv")

MIRROR_URLS = [
    "https://raw.githubusercontent.com/AlaaNabil98/CodeClause_Customer_Churn_Rate_Analysis/main/WA_Fn-UseC_-Telco-Customer-Churn.csv",  # noqa: E501
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
]

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def _generate_synthetic(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Synthetic fallback with the exact same column schema as the real
    dataset, so downstream code (pipeline.py) never has to branch on which
    source was used."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    monthly_charges = np.clip(rng.normal(64, 30, n), 18.25, 118.75)
    total_charges = np.clip(monthly_charges * tenure + rng.normal(0, 50, n), 0, None)

    # Churn probability driven by realistic-ish latent structure so the
    # trained model has real signal to learn from.
    churn_logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        - 0.03 * tenure
        + 0.01 * (monthly_charges - 64)
        + 0.5 * (internet == "Fiber optic")
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
            "MonthlyCharges": monthly_charges.round(2),
            "TotalCharges": total_charges.round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return df


def describe_source(df: pd.DataFrame) -> str:
    """'ibm_telco' for the real public dataset, 'synthetic_fallback' for the offline generator
    (recorded in model metadata so a model trained on fake data can never pass for a real one)."""
    ids = df["customerID"].astype(str)
    return "synthetic_fallback" if ids.str.startswith("SYN-").all() else "ibm_telco"


def fetch_telco_data(force: bool = False) -> pd.DataFrame:
    """Returns the Telco churn DataFrame, downloading + caching to
    `data/raw/telco_customer_churn.csv` on first call. Falls back to a
    synthetic dataset if no mirror is reachable."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_PATH) and not force:
        df = pd.read_csv(RAW_PATH)
        if EXPECTED_COLUMNS.issubset(df.columns):
            logger.info("Loaded cached dataset from %s (%d rows)", RAW_PATH, len(df))
            return df

    for url in MIRROR_URLS:
        try:
            logger.info("Attempting to download dataset from %s", url)
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(RAW_PATH, "wb") as f:
                f.write(resp.content)
            df = pd.read_csv(RAW_PATH)
            if EXPECTED_COLUMNS.issubset(df.columns):
                logger.info("Downloaded and cached dataset (%d rows)", len(df))
                return df
            logger.warning("Downloaded file from %s had unexpected schema; trying next mirror", url)
        except Exception as exc:  # noqa: BLE001 - any network/parse failure -> try next source
            logger.warning("Failed to fetch from %s: %s", url, exc)

    logger.warning("All mirrors unreachable; generating synthetic fallback dataset")
    df = _generate_synthetic()
    df.to_csv(RAW_PATH, index=False)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_telco_data(force=True)
    print(data.shape)
    print(data["Churn"].value_counts(normalize=True))
