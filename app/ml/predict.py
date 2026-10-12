"""Single source of truth for scoring a customer's churn risk.

`predict(features)` is used by the live API (prediction service, batch scoring, CLI seeding) - never
duplicate this logic elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import joblib
import numpy as np
import pandas as pd

from app.ml.pipeline import NOMINAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))

# Tiers apply to CALIBRATED probabilities: Low = below the ~26% base churn rate, High = more likely than not.
RISK_LOW_MAX = float(os.environ.get("RISK_LOW_MAX", 0.25))
RISK_MEDIUM_MAX = float(os.environ.get("RISK_MEDIUM_MAX", 0.50))

_lock = threading.Lock()
_artifacts_cache: dict = {}

# --- Recommended-action rules table -------------------------------------------------
# One small, explicit table. The FIRST risk-increasing factor (top factor first) that matches a rule
# for the customer's risk level decides the action. Extend by adding rows.
RECOMMENDATION_RULES: list[dict] = [
    {
        "factor": "Contract=Month-to-month",
        "risk_levels": {"High", "Medium"},
        "action": "Offer a 12-month contract discount",
    },
    {
        "factor": "MonthlyCharges (high)",
        "risk_levels": {"High", "Medium"},
        "action": "Offer a loyalty / price-lock discount",
    },
    {
        "factor": "tenure (short)",
        "risk_levels": {"High", "Medium"},
        "action": "Schedule a proactive onboarding check-in call",
    },
    {
        "factor": "InternetService=Fiber optic",
        "risk_levels": {"High"},
        "action": "Investigate service quality and offer a service credit",
    },
    {
        "factor": "PaymentMethod=Electronic check",
        "risk_levels": {"High", "Medium"},
        "action": "Encourage autopay with a small incentive",
    },
    {"factor": "TechSupport=No", "risk_levels": {"High", "Medium"}, "action": "Offer a free tech-support trial"},
    {
        "factor": "OnlineSecurity=No",
        "risk_levels": {"High", "Medium"},
        "action": "Bundle online security free for 3 months",
    },
]  # noqa: E501
DEFAULT_ACTIONS = {
    "High": "Escalate to retention team for a personalized outreach call",
    "Medium": "Add to targeted email retention campaign",
    "Low": "No action needed; continue standard engagement",
}


class ModelNotTrainedError(RuntimeError):
    """Raised when predict() is called before model artifacts exist."""


def _load_artifacts() -> dict:
    """Loads model.pkl / pipeline.pkl / metadata.json once (thread-safe). Call `reload_artifacts()` after retraining."""
    with _lock:
        if _artifacts_cache:
            return _artifacts_cache

        model_path = os.path.join(MODEL_DIR, "model.pkl")
        pipeline_path = os.path.join(MODEL_DIR, "pipeline.pkl")
        metadata_path = os.path.join(MODEL_DIR, "metadata.json")

        if not (os.path.exists(model_path) and os.path.exists(pipeline_path) and os.path.exists(metadata_path)):
            raise ModelNotTrainedError(f"No trained model found in {MODEL_DIR}. Run `python -m app.ml.train` first.")

        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        calibrator_path = os.path.join(MODEL_DIR, "calibrator.pkl")
        calibrator = joblib.load(calibrator_path) if os.path.exists(calibrator_path) else None
        _artifacts_cache.update(model=model, pipeline=pipeline, metadata=metadata, calibrator=calibrator)
        logger.info("Loaded model artifacts from %s", MODEL_DIR)
        return _artifacts_cache


def reload_artifacts() -> None:
    """Forces a fresh load on the next predict() call (e.g. after retraining)."""
    with _lock:
        _artifacts_cache.clear()


def get_metadata() -> dict:
    return _load_artifacts()["metadata"]


def risk_level_for(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "Low"
    if probability < RISK_MEDIUM_MAX:
        return "Medium"
    return "High"


def rules_table() -> list[dict]:
    """The recommendation rules in JSON-friendly form (exposed by GET /model/info)."""
    return [
        {"when_top_factor_is": r["factor"], "risk_levels": sorted(r["risk_levels"]), "action": r["action"]}
        for r in RECOMMENDATION_RULES
    ] + [
        {"when_top_factor_is": "(no rule matches)", "risk_levels": [k], "action": v} for k, v in DEFAULT_ACTIONS.items()
    ]


def _raw_features_to_frame(features: dict) -> pd.DataFrame:
    """One-row DataFrame with exactly the columns the pipeline expects; missing ones become NaN so the
    pipeline's own imputers handle them exactly as during training."""
    expected_cols = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    row = {col: features.get(col, np.nan) for col in expected_cols}
    return pd.DataFrame([row])


def _readable_label(name: str, value: float, features: dict) -> str:
    """Turn an encoded feature name into a human label.

    e.g. 'InternetService_Fiber optic' -> 'InternetService=Fiber optic'
    """
    if name == "tenure":  # x_value is the standardized value: negative = shorter than the average customer
        return "tenure (short)" if value < 0 else "tenure (long)"
    if name == "MonthlyCharges":
        return "MonthlyCharges (high)" if value > 0 else "MonthlyCharges (low)"
    if name == "SeniorCitizen":
        return f"SeniorCitizen={'Yes' if features.get(name) else 'No'}"
    if name in ORDINAL_FEATURES:
        raw = features.get(name)
        return f"{name}={raw}" if raw is not None and raw == raw else name
    for base in NOMINAL_FEATURES:
        if name.startswith(base + "_"):
            category = name[len(base) + 1 :]
            return f"{base}={category}" if value > 0.5 else f"{base}!={category}"
    return name


def _contributions(model, X_transformed: np.ndarray) -> np.ndarray:
    """Per-feature contribution for one row: SHAP for tree models, coefficient x value for linear models."""
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        try:
            import shap

            values = shap.TreeExplainer(model).shap_values(X_transformed)
            if isinstance(values, list):  # older shap: one array per class
                return np.asarray(values[1])[0]
            values = np.asarray(values)
            if values.ndim == 3:  # (samples, features, classes)
                return values[0, :, 1]
            return values[0]
        except Exception:  # pragma: no cover - explanation is best-effort, never blocks a prediction
            logger.exception("SHAP failed; falling back to importance x value")
            importances = getattr(model, "feature_importances_", np.zeros(X_transformed.shape[1]))
            return importances * X_transformed[0]
    return model.coef_[0] * X_transformed[0]  # standardized coefficient x (scaled) value


def _fold_no_internet(contrib: np.ndarray, names: list[str]) -> np.ndarray:
    """Add-on columns such as `TechSupport_No internet service` are perfectly collinear with `InternetService_No`; the
    linear model splits one effect over seven columns, which would show as seven near-identical factors. Fold them
    into `InternetService_No` so an explanation lists it once."""
    if "InternetService_No" not in names:
        return contrib
    target = names.index("InternetService_No")
    cols = [i for i, n in enumerate(names) if n.endswith("_No internet service")]
    if not cols:
        return contrib
    out = np.array(contrib, dtype=float, copy=True)
    out[..., target] += out[..., cols].sum(axis=-1)
    out[..., cols] = 0.0
    return out


def _top_key_factors(
    model, X_transformed: np.ndarray, feature_names: list[str], features: dict, top_n: int = 5
) -> list[dict]:
    contrib = _fold_no_internet(_contributions(model, X_transformed), feature_names)
    ranked = sorted(zip(feature_names, contrib, X_transformed[0]), key=lambda t: abs(t[1]), reverse=True)
    factors = []
    for name, impact, x_value in ranked:
        if abs(impact) < 1e-9:
            continue  # inactive one-hot columns contribute nothing for linear models
        factors.append(
            {
                "feature": _readable_label(name, float(x_value), features),
                "impact": round(float(impact), 4),
                "direction": "increases churn risk" if impact > 0 else "reduces churn risk",
            }
        )
        if len(factors) == top_n:
            break
    return factors


def _recommended_action(key_factors: list[dict], risk_level: str) -> str:
    """Pick the action from the top risk-increasing factors (first match wins)."""
    for factor in [f for f in key_factors if f["impact"] > 0][:2]:
        for rule in RECOMMENDATION_RULES:
            if rule["factor"] == factor["feature"] and risk_level in rule["risk_levels"]:
                return rule["action"]
    return DEFAULT_ACTIONS.get(risk_level, DEFAULT_ACTIONS["Low"])


def global_feature_importance(top_n: int = 15) -> list[dict]:
    """Model-wide feature importance (|coefficient| for linear models, impurity importance for trees)."""
    artifacts = _load_artifacts()
    model, names = artifacts["model"], artifacts["metadata"]["feature_names"]
    if hasattr(model, "coef_"):
        raw = model.coef_[0]
        items = [
            (n, abs(float(v)), "increases churn risk" if v > 0 else "reduces churn risk") for n, v in zip(names, raw)
        ]
    else:
        items = [(n, float(v), "") for n, v in zip(names, model.feature_importances_)]
    items.sort(key=lambda t: t[1], reverse=True)
    return [{"feature": n, "importance": round(v, 4), "direction": d} for n, v, d in items[:top_n]]


def predict(features: dict) -> dict:
    """Scores one customer's churn features (raw, untransformed values keyed by Telco column names).

    Returns churn_probability, risk_level, key_factors, recommended_action and model_version.
    """
    artifacts = _load_artifacts()
    model, pipeline, metadata = artifacts["model"], artifacts["pipeline"], artifacts["metadata"]

    X_transformed = pipeline.transform(_raw_features_to_frame(features))
    probability = float(model.predict_proba(X_transformed)[0, 1])
    calibrator = artifacts.get("calibrator")
    if calibrator is not None:  # raw balanced-class score -> real-world churn probability
        probability = float(calibrator.transform([probability])[0])
    risk_level = risk_level_for(probability)
    key_factors = _top_key_factors(model, X_transformed, metadata["feature_names"], features)

    return {
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "key_factors": key_factors,
        "recommended_action": _recommended_action(key_factors, risk_level),
        "model_version": metadata["model_version"],
    }


# --------------------------------------------------------------------------------------------------------------
# Vectorised scoring (many customers at once). Same transform -> model -> calibrator -> tier -> action logic as
# `predict()`; tests assert both paths return identical results, so there is still one definition of "a score".
# --------------------------------------------------------------------------------------------------------------
def _contributions_matrix(model, X: np.ndarray) -> np.ndarray:
    """(n_rows, n_features) contributions: SHAP for tree models (in chunks), coefficient x value for linear models."""
    if type(model).__name__ in ("RandomForestClassifier", "XGBClassifier"):
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            parts = []
            for start in range(0, len(X), 2000):
                values = explainer.shap_values(X[start : start + 2000])
                if isinstance(values, list):
                    values = values[1]
                values = np.asarray(values)
                parts.append(values[:, :, 1] if values.ndim == 3 else values)
            return np.vstack(parts)
        except Exception:  # pragma: no cover - explanations are best-effort and never block scoring
            logger.exception("SHAP failed in batch; falling back to importance x value")
            return getattr(model, "feature_importances_", np.zeros(X.shape[1])) * X
    return model.coef_[0] * X


def predict_frame(frame: pd.DataFrame, factors: bool = True, top_n: int = 5) -> list[dict]:
    """Score every row of a DataFrame of raw features (Telco column names; missing columns/values are imputed).

    Returns one dict per row, in order, shaped exactly like `predict()`'s result. Pass `factors=False` to skip
    the per-row explanations (faster; `key_factors` is then empty and the action falls back to the tier default).
    """
    if frame.empty:
        return []
    artifacts = _load_artifacts()
    model, pipeline, metadata = artifacts["model"], artifacts["pipeline"], artifacts["metadata"]
    expected = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    frame = frame.reindex(columns=expected)

    X = np.asarray(pipeline.transform(frame), dtype=float)
    probability = model.predict_proba(X)[:, 1]
    calibrator = artifacts.get("calibrator")
    if calibrator is not None:
        probability = np.asarray(calibrator.transform(probability), dtype=float)
    probability = np.round(probability, 4)
    levels = np.where(probability < RISK_LOW_MAX, "Low", np.where(probability < RISK_MEDIUM_MAX, "Medium", "High"))

    names = metadata["feature_names"]
    contrib = _fold_no_internet(_contributions_matrix(model, X), names) if factors else None
    senior = frame["SeniorCitizen"].to_numpy() if "SeniorCitizen" in frame else None
    contract = frame["Contract"].to_numpy()
    results = []
    for i in range(len(frame)):
        key_factors: list[dict] = []
        if contrib is not None:
            row = contrib[i]
            order = np.argsort(-np.abs(row), kind="stable")
            raw = {"Contract": contract[i], "SeniorCitizen": None if senior is None else senior[i]}
            for j in order:
                impact = float(row[j])
                if abs(impact) < 1e-9:
                    break
                key_factors.append(
                    {
                        "feature": _readable_label(names[j], float(X[i, j]), raw),
                        "impact": round(impact, 4),
                        "direction": "increases churn risk" if impact > 0 else "reduces churn risk",
                    }
                )
                if len(key_factors) == top_n:
                    break
        level = str(levels[i])
        results.append(
            {
                "churn_probability": float(probability[i]),
                "risk_level": level,
                "key_factors": key_factors,
                "recommended_action": _recommended_action(key_factors, level),
                "model_version": metadata["model_version"],
            }
        )
    return results


def features_to_frame(rows: list[dict]) -> pd.DataFrame:
    """List of raw-feature dicts -> DataFrame with the pipeline's columns (missing -> NaN)."""
    expected = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
    return pd.DataFrame([{col: row.get(col, np.nan) for col in expected} for row in rows], columns=expected)
