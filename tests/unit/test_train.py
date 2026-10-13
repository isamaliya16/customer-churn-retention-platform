"""Runs the full training workflow on a small synthetic dataset (fast) and checks the saved artifacts."""

import json
import os

import joblib
import pytest

from app.ml import predict as pm
from app.ml import train
from app.ml.fetch_data import _generate_synthetic


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    mp = pytest.MonkeyPatch()
    mp.setattr(train, "MODEL_DIR", str(out))
    mp.setattr(train, "fetch_telco_data", lambda: _generate_synthetic(700))
    meta = train.train_and_select_best()
    yield out, meta
    mp.undo()


def test_train_persists_all_artifacts(trained):
    out, meta = trained
    for name in ("model.pkl", "pipeline.pkl", "calibrator.pkl", "metadata.json"):
        assert os.path.exists(out / name)
    on_disk = json.loads((out / "metadata.json").read_text())
    assert on_disk["model_version"] == meta["model_version"]


def test_train_compares_three_models_and_records_metrics(trained):
    _, meta = trained
    assert set(meta["cv_metrics"]) == {"logistic_regression", "random_forest", "xgboost"}
    for metrics in meta["held_out_metrics"].values():
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert meta["algorithm"] in meta["cv_metrics"]
    assert meta["selection_criteria"]["weights"] == {"recall": 0.4, "f1": 0.4, "roc_auc": 0.2}


def test_train_records_roc_curves_and_confusion_matrix(trained):
    _, meta = trained
    assert set(meta["roc_curves"]) == set(meta["cv_metrics"])
    curve = meta["roc_curves"]["logistic_regression"]
    assert curve["fpr"][0] == 0 and curve["tpr"][-1] == 1 and len(curve["fpr"]) == len(curve["tpr"])
    matrix = meta["confusion_matrix"]["matrix"]
    assert len(matrix) == 2 and sum(map(sum, matrix)) == meta["n_test_rows"]


def test_train_uses_stratified_80_20_split(trained):
    _, meta = trained
    assert meta["n_train_rows"] + meta["n_test_rows"] == 700
    assert meta["n_test_rows"] == 140


def test_selection_score_prefers_recall_and_f1_over_accuracy():
    high_recall = {"recall": 0.9, "f1": 0.6, "roc_auc": 0.8, "accuracy": 0.5}
    high_accuracy = {"recall": 0.4, "f1": 0.5, "roc_auc": 0.8, "accuracy": 0.95}
    assert train._score_for_selection(high_recall) > train._score_for_selection(high_accuracy)


def test_saved_pipeline_and_model_score_new_data(trained, monkeypatch):
    out, _ = trained
    monkeypatch.setattr(pm, "MODEL_DIR", str(out))
    pm.reload_artifacts()
    try:
        result = pm.predict({"tenure": 2, "MonthlyCharges": 90.0, "Contract": "Month-to-month"})
        assert 0 <= result["churn_probability"] <= 1
        assert joblib.load(out / "pipeline.pkl") is not None
    finally:
        pm.reload_artifacts()


def test_retraining_increments_the_model_version(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(train, "fetch_telco_data", lambda: _generate_synthetic(400))
    first = train.train_and_select_best()["model_version"]
    second = train.train_and_select_best()["model_version"]
    assert first.endswith("_v1") and second.endswith("_v2")


def test_metadata_records_data_source_and_calibration(trained):
    _, meta = trained
    assert meta["data_source"] == "synthetic_fallback"  # a model trained on fake data can never pass for a real one
    cal = meta["calibration"]
    assert cal["brier_calibrated"] <= cal["brier_raw"] + 1e-9
    assert abs(cal["mean_predicted_calibrated"] - cal["actual_churn_rate"]) < 0.08
    assert set(cal["risk_tiers"]) == {"Low", "Medium", "High"}
    assert sum(t["customers"] for t in cal["risk_tiers"].values()) == meta["n_test_rows"]
    assert 0 < meta["flag_threshold"] < 1


def test_xgboost_gets_the_same_imbalance_treatment_as_the_others():
    models = train._candidate_models(2.7)
    assert models["xgboost"].get_params()["scale_pos_weight"] == 2.7
    assert models["logistic_regression"].get_params()["class_weight"] == "balanced"
    assert models["random_forest"].get_params()["class_weight"] == "balanced"


def test_calibrator_pulls_inflated_probabilities_down():
    import numpy as np

    from app.ml.calibration import ProbabilityCalibrator

    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.9, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(int)
    inflated = np.clip(true_p * 0.5 + 0.45, 0.01, 0.99)  # mimics class-weight inflation
    calibrated = ProbabilityCalibrator().fit(inflated, y).transform(inflated)
    assert abs(calibrated.mean() - y.mean()) < 0.02 < abs(inflated.mean() - y.mean())
