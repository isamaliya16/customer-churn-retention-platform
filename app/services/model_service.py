"""Exposes information about the currently deployed model."""

from __future__ import annotations

from app.ml import predict as ml
from app.utils.errors import ValidationAPIError


class ModelService:
    @staticmethod
    def info() -> dict:
        try:
            meta = ml.get_metadata()
        except ml.ModelNotTrainedError as exc:
            raise ValidationAPIError(str(exc), code="model_not_trained") from exc
        return {
            "model_version": meta["model_version"],
            "algorithm": meta["algorithm"],
            "trained_at": meta["trained_at"],
            "n_train_rows": meta["n_train_rows"],
            "n_test_rows": meta["n_test_rows"],
            "n_features": len(meta["feature_names"]),
            "churn_rate_in_training": meta.get("class_balance", {}).get("churn_rate"),
            "selection_criteria": meta.get("selection_criteria", {}),
            "cv_metrics": meta.get("cv_metrics", {}),
            "held_out_metrics": meta.get("held_out_metrics", {}),
            "confusion_matrix": meta.get("confusion_matrix"),
            "roc_curves": meta.get("roc_curves", {}),
            "outlier_report": meta.get("outlier_report", {}),
            "data_source": meta.get("data_source", "unknown"),
            "flag_threshold": meta.get("flag_threshold"),
            "calibration": meta.get("calibration"),
            "feature_importance": ml.global_feature_importance(),
            "risk_thresholds": {
                "low_max": ml.RISK_LOW_MAX,  # single source of truth: app/ml/predict.py
                "medium_max": ml.RISK_MEDIUM_MAX,
            },
            "recommendation_rules": ml.rules_table(),
        }
