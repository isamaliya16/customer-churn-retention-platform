"""Unit tests for app/ml/pipeline.py transforms."""

import numpy as np
import pandas as pd

from app.ml.pipeline import (
    build_feature_pipeline,
    clean_raw_dataframe,
    detect_charge_outliers,
    flag_high_missing_columns,
    get_feature_names,
    prepare_training_frame,
)


def _sample_df(n=20):
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(n)],
            "gender": ["Male", "Female"] * (n // 2),
            "SeniorCitizen": [0, 1] * (n // 2),
            "Partner": ["Yes", "No"] * (n // 2),
            "Dependents": ["No"] * n,
            "tenure": list(range(n)),
            "PhoneService": ["Yes"] * n,
            "MultipleLines": ["No"] * n,
            "InternetService": ["DSL", "Fiber optic"] * (n // 2),
            "OnlineSecurity": ["No"] * n,
            "OnlineBackup": ["No"] * n,
            "DeviceProtection": ["No"] * n,
            "TechSupport": ["No"] * n,
            "StreamingTV": ["No"] * n,
            "StreamingMovies": ["No"] * n,
            "Contract": ["Month-to-month", "One year", "Two year"] * (n // 3) + ["Month-to-month"] * (n % 3),
            "PaperlessBilling": ["Yes"] * n,
            "PaymentMethod": ["Electronic check"] * n,
            "MonthlyCharges": np.linspace(20, 120, n),
            "TotalCharges": [str(x) if i % 5 else " " for i, x in enumerate(np.linspace(20, 3000, n))],
            "Churn": ["Yes", "No"] * (n // 2),
        }
    )


def test_clean_raw_dataframe_coerces_total_charges_to_numeric():
    df = _sample_df()
    cleaned = clean_raw_dataframe(df)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    # The blank-string rows should become NaN, not raise.
    assert cleaned["TotalCharges"].isna().sum() > 0


def test_flag_high_missing_columns_adds_flag_above_threshold():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [np.nan, np.nan, np.nan, 4, 5]})
    flagged_df, flagged_cols = flag_high_missing_columns(df, threshold=0.4)
    assert "b" in flagged_cols
    assert "b_was_missing" in flagged_df.columns
    assert "a" not in flagged_cols


def test_detect_charge_outliers_reports_without_dropping():
    df = pd.DataFrame({"MonthlyCharges": [20, 21, 22, 23, 500], "TotalCharges": [1, 2, 3, 4, 5]})
    report = detect_charge_outliers(df, cols=["MonthlyCharges"])
    result = report.to_dict()
    assert "MonthlyCharges" in result
    assert result["MonthlyCharges"]["outlier_count"] >= 1


def test_build_feature_pipeline_fits_and_transforms():
    df = clean_raw_dataframe(_sample_df())
    X, y, _, _ = prepare_training_frame(df)
    pipeline = build_feature_pipeline()
    X_t = pipeline.fit_transform(X, y)
    assert X_t.shape[0] == len(X)
    assert not np.isnan(X_t).any()  # imputation should remove all NaNs


def test_get_feature_names_matches_transformed_width():
    df = clean_raw_dataframe(_sample_df())
    X, y, _, _ = prepare_training_frame(df)
    pipeline = build_feature_pipeline()
    X_t = pipeline.fit_transform(X, y)
    names = get_feature_names(pipeline)
    assert len(names) == X_t.shape[1]


def test_prepare_training_frame_encodes_churn_as_binary():
    df = _sample_df()
    _, y, _, _ = prepare_training_frame(df)
    assert set(y.unique()).issubset({0, 1})
