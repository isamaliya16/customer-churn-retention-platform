"""Train and compare churn models; persist the winner.

Run directly: `python -m app.ml.train`

Trains Logistic Regression, Random Forest, and XGBoost on the same
stratified 80/20 split. All three handle class imbalance the same way (class weights), so the comparison is
fair. Models are cross-validated and the winner is chosen by RECALL and F1 on the churn (positive) class
rather than raw accuracy - for a churn use case, missing a customer who actually leaves (a false negative) is
costlier than a false positive. ROC-AUC and PR-AUC are reported too; note that the three models are close on
ROC-AUC, so recall/F1 differences at the default 0.5 cut-off are partly a threshold effect.

Because class weighting inflates predicted probabilities, the winner's scores are then CALIBRATED (Platt
scaling fitted on out-of-fold training predictions) so that "70%" really means about 70% of such customers
churn. Held-out calibration and per-risk-tier results are written to the metadata.

Persists: model.pkl, pipeline.pkl, calibrator.pkl, metadata.json under MODEL_DIR.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from xgboost import XGBClassifier

from app.ml.calibration import ProbabilityCalibrator
from app.ml.fetch_data import describe_source, fetch_telco_data
from app.ml.pipeline import build_feature_pipeline, get_feature_names, prepare_training_frame
from app.ml.predict import RISK_LOW_MAX, RISK_MEDIUM_MAX

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))
RANDOM_STATE = 42

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

# Recall and F1 on the churn class are weighted most heavily when picking
# the winner - see module docstring for rationale.
SELECTION_WEIGHTS = {"recall": 0.4, "f1": 0.4, "roc_auc": 0.2}


def _candidate_models(scale_pos_weight: float) -> dict:
    """All three models get the SAME imbalance treatment (class weights) so the comparison is fair."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _cross_validate_model(name: str, model, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    summary = {metric: float(np.mean(results[f"test_{metric}"])) for metric in SCORING}
    logger.info("[%s] 5-fold CV: %s", name, {k: round(v, 4) for k, v in summary.items()})
    return summary


def _score_for_selection(cv_summary: dict) -> float:
    return sum(SELECTION_WEIGHTS[m] * cv_summary[m] for m in SELECTION_WEIGHTS)


def _held_out_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }


def _roc_points(model, X_test, y_test, n_points: int = 40) -> dict:
    """Down-sampled ROC curve for the dashboard chart."""
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    idx = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return {"fpr": [round(float(fpr[i]), 4) for i in idx], "tpr": [round(float(tpr[i]), 4) for i in idx]}


def _next_version(model_dir: str, algorithm: str) -> str:
    """`<algorithm>_v<N>` where N counts up from the previously saved model, so every retrain is distinguishable."""
    n = 1
    try:
        with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
            match = re.search(r"_v(\d+)$", json.load(f).get("model_version", ""))
        if match:
            n = int(match.group(1)) + 1
    except (OSError, ValueError):
        pass
    return f"{algorithm}_v{n}"


def _tier_for(p: float) -> str:
    return "Low" if p < RISK_LOW_MAX else "Medium" if p < RISK_MEDIUM_MAX else "High"


def _calibration_report(y_true, raw_proba, calibrated) -> dict:
    """Held-out evidence that probabilities are honest: Brier score before/after, a binned reliability table and
    how many customers land in each risk tier and how many of them really churned."""
    y_true = np.asarray(y_true)
    edges = [0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0001]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (calibrated >= lo) & (calibrated < hi)
        if mask.any():
            bins.append(
                {
                    "range": f"{lo:.2f}-{min(hi, 1.0):.2f}",
                    "customers": int(mask.sum()),
                    "mean_predicted": round(float(calibrated[mask].mean()), 4),
                    "actual_churn_rate": round(float(y_true[mask].mean()), 4),
                }
            )
    tiers = {}
    levels = np.array([_tier_for(p) for p in calibrated])
    for tier in ("Low", "Medium", "High"):
        mask = levels == tier
        tiers[tier] = {
            "customers": int(mask.sum()),
            "share_of_customers": round(float(mask.mean()), 4),
            "mean_predicted": round(float(calibrated[mask].mean()), 4) if mask.any() else None,
            "actual_churn_rate": round(float(y_true[mask].mean()), 4) if mask.any() else None,
            "share_of_all_churners": round(float(y_true[mask].sum() / max(y_true.sum(), 1)), 4),
        }
    return {
        "brier_raw": round(float(brier_score_loss(y_true, raw_proba)), 4),
        "brier_calibrated": round(float(brier_score_loss(y_true, calibrated)), 4),
        "mean_predicted_raw": round(float(np.mean(raw_proba)), 4),
        "mean_predicted_calibrated": round(float(np.mean(calibrated)), 4),
        "actual_churn_rate": round(float(y_true.mean()), 4),
        "bins": bins,
        "risk_tiers": tiers,
        "risk_thresholds": {"low_max": RISK_LOW_MAX, "medium_max": RISK_MEDIUM_MAX},
    }


def train_and_select_best() -> dict:
    """Full training workflow. Returns the metadata dict that also gets
    written to disk as metadata.json."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    raw_df = fetch_telco_data()
    X, y, outlier_report, flagged_missing = prepare_training_frame(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    feature_pipeline = build_feature_pipeline()
    X_train_t = feature_pipeline.fit_transform(X_train, y_train)
    X_test_t = feature_pipeline.transform(X_test)
    feature_names = get_feature_names(feature_pipeline)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = _candidate_models(scale_pos_weight)
    cv_summaries: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    held_out: dict[str, dict] = {}

    for name, model in candidates.items():
        start = time.time()
        cv_summaries[name] = _cross_validate_model(name, model, X_train_t, y_train)
        model.fit(X_train_t, y_train)
        fitted_models[name] = model
        held_out[name] = _held_out_metrics(model, X_test_t, y_test)
        logger.info(
            "[%s] held-out test metrics: %s (%.1fs)",
            name,
            {k: round(v, 4) for k, v in held_out[name].items()},
            time.time() - start,
        )

    best_name = max(cv_summaries, key=lambda n: _score_for_selection(cv_summaries[n]))
    best_model = fitted_models[best_name]
    model_version = _next_version(MODEL_DIR, best_name)

    # Calibrate the winner's probabilities on OUT-OF-FOLD training predictions (never on the held-out test set).
    oof_raw = cross_val_predict(
        clone(candidates[best_name]),
        X_train_t,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    calibrator = ProbabilityCalibrator().fit(oof_raw, y_train)
    test_raw = best_model.predict_proba(X_test_t)[:, 1]
    test_calibrated = calibrator.transform(test_raw)
    calibration = _calibration_report(y_test, test_raw, test_calibrated)
    flag_threshold = float(calibrator.transform([0.5])[0])  # calibrated probability equivalent of the 0.5 cut-off

    logger.info(
        "Selected %r as best model (optimized for recall+F1 on the churn class, "
        "weighted %s over ROC-AUC/accuracy because false negatives - missed "
        "churners - are the costliest error for a retention program)",
        best_name,
        SELECTION_WEIGHTS,
    )

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(feature_pipeline, os.path.join(MODEL_DIR, "pipeline.pkl"))
    joblib.dump(calibrator, os.path.join(MODEL_DIR, "calibrator.pkl"))

    metadata = {
        "model_version": model_version,
        "algorithm": best_name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "data_source": describe_source(raw_df),
        "feature_names": feature_names,
        "selection_criteria": {
            "weights": SELECTION_WEIGHTS,
            "rationale": (
                "Recall and F1 on the churn (positive) class are weighted "
                "over raw accuracy/ROC-AUC because missing an at-risk "
                "customer (false negative) costs more than an unnecessary "
                "retention offer (false positive)."
            ),
        },
        "cv_metrics": cv_summaries,
        "held_out_metrics": held_out,
        "confusion_matrix": {
            "labels": ["stayed", "churned"],
            "matrix": confusion_matrix(y_test, best_model.predict(X_test_t)).tolist(),
        },
        "flag_threshold": round(flag_threshold, 4),
        "calibration": calibration,
        "roc_curves": {name: _roc_points(m, X_test_t, y_test) for name, m in fitted_models.items()},
        "outlier_report": outlier_report.to_dict(),
        "high_missing_columns_flagged": flagged_missing,
        "class_balance": {"churn_rate": float(y.mean())},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts written to %s", MODEL_DIR)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    meta = train_and_select_best()
    print(
        json.dumps(
            {"model_version": meta["model_version"], "held_out_metrics": meta["held_out_metrics"][meta["algorithm"]]},
            indent=2,
        )
    )
