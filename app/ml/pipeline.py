"""Feature engineering pipeline for churn prediction.

This module builds a single scikit-learn ColumnTransformer-based pipeline
that handles missing-value imputation, encoding, and scaling. The SAME
fitted object is used at train time and at inference time (persisted with
joblib), so training and serving can never drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET_COL = "Churn"
ID_COL = "customerID"

# TotalCharges is deliberately NOT a model feature: it is ~tenure x MonthlyCharges, so keeping it makes the linear
# model's coefficients collinear and its "key factor" explanations unreliable (e.g. higher charges looked protective).
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "SeniorCitizen"]

# Ordinal because there's a natural order (fewer months committed -> more risk).
ORDINAL_FEATURES = ["Contract"]
ORDINAL_CATEGORIES = [["Month-to-month", "One year", "Two year"]]

NOMINAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "PaymentMethod",
]

MISSING_FLAG_THRESHOLD = 0.40  # columns with >40% missing get a "was_missing" flag


@dataclass
class OutlierReport:
    """Reports (does not silently drop) IQR-based outliers on charge fields."""

    flagged_columns: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.flagged_columns


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Type coercion + missing-value bookkeeping applied before the sklearn
    pipeline. Mirrors real-world messiness in the raw Telco CSV (e.g.
    TotalCharges is stored as a string with blank values for new customers)."""
    df = df.copy()

    # TotalCharges arrives as an object dtype with blank strings for
    # customers with 0 tenure; coerce it (it is still used for outlier reporting, not as a model feature).
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(float)

    return df


def flag_high_missing_columns(
    df: pd.DataFrame, threshold: float = MISSING_FLAG_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Adds a boolean `<col>_was_missing` flag for any column missing more
    than `threshold` fraction of values, per the global standard of
    reporting rather than silently dropping data."""
    df = df.copy()
    flagged = []
    missing_frac = df.isna().mean()
    for col, frac in missing_frac.items():
        if frac > threshold:
            df[f"{col}_was_missing"] = df[col].isna()
            flagged.append(col)
            logger.warning("Column %r is %.1f%% missing -> flagged with %s_was_missing", col, frac * 100, col)
    return df, flagged


def detect_charge_outliers(df: pd.DataFrame, cols: list[str] | None = None) -> OutlierReport:
    """IQR-based outlier detection on charge columns. Returns a report;
    callers decide what (if anything) to do - rows are never dropped here."""
    cols = cols or ["MonthlyCharges", "TotalCharges"]
    report = OutlierReport()
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        report.flagged_columns[col] = {
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_pct": round(float(outlier_mask.mean() * 100), 2),
        }
    return report


def build_feature_pipeline() -> ColumnTransformer:
    """The single source of truth for feature transforms: median imputation
    + scaling for numerics, most-frequent imputation + ordinal encoding for
    ranked categoricals, most-frequent imputation + one-hot for nominal
    categoricals."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_names(fitted_pipeline: ColumnTransformer) -> list[str]:
    """Human-readable output feature names, used for SHAP / coefficient
    labeling in key_factors."""
    names: list[str] = list(NUMERIC_FEATURES)
    names.extend(ORDINAL_FEATURES)
    nominal_encoder: OneHotEncoder = fitted_pipeline.named_transformers_["nominal"].named_steps["encoder"]
    names.extend(nominal_encoder.get_feature_names_out(NOMINAL_FEATURES).tolist())
    return names


def prepare_training_frame(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, OutlierReport, list[str]]:
    """Data preparation: clean -> flag missingness -> detect outliers ->
    split X/y. Returns (X, y, outlier_report, missing_flagged_columns)."""
    df = clean_raw_dataframe(raw_df)
    df, flagged_missing = flag_high_missing_columns(df)
    outlier_report = detect_charge_outliers(df)

    y = (df[TARGET_COL] == "Yes").astype(int)
    feature_cols = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    X = df[feature_cols]
    return X, y, outlier_report, flagged_missing
