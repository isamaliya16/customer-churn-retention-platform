"""Unit tests for app/ml/predict.py: tiers, rules, explanations (linear + SHAP tree paths), model loading."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from app.ml import predict as pm
from app.ml.fetch_data import _generate_synthetic
from app.ml.pipeline import build_feature_pipeline, get_feature_names, prepare_training_frame

HIGH_RISK = {
    "tenure": 1,
    "MonthlyCharges": 99.0,
    "TotalCharges": 99.0,
    "SeniorCitizen": 0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "TechSupport": "No",
    "OnlineSecurity": "No",
}


def test_risk_level_boundaries():
    for prob, level in [
        (0.0, "Low"),
        (0.249, "Low"),
        (0.25, "Medium"),
        (0.499, "Medium"),
        (0.50, "High"),
        (1.0, "High"),
    ]:
        assert pm.risk_level_for(prob) == level


def test_load_artifacts_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "MODEL_DIR", str(tmp_path / "nothing-here"))
    pm.reload_artifacts()
    with pytest.raises(pm.ModelNotTrainedError):
        pm._load_artifacts()


@pytest.mark.parametrize(
    "encoded,value,expected",
    [
        ("tenure", 0.3, "tenure (long)"),
        ("tenure", -1.2, "tenure (short)"),
        ("MonthlyCharges", 0.8, "MonthlyCharges (high)"),
        ("MonthlyCharges", -0.8, "MonthlyCharges (low)"),
        ("InternetService_Fiber optic", 1.0, "InternetService=Fiber optic"),
        ("InternetService_Fiber optic", 0.0, "InternetService!=Fiber optic"),
        ("PaymentMethod_Bank transfer (automatic)", 1.0, "PaymentMethod=Bank transfer (automatic)"),
    ],
)
def test_readable_label(encoded, value, expected):
    assert pm._readable_label(encoded, value, {}) == expected


def test_readable_label_uses_raw_contract_value():
    assert pm._readable_label("Contract", 0.0, {"Contract": "Month-to-month"}) == "Contract=Month-to-month"
    assert pm._readable_label("Contract", 0.0, {}) == "Contract"


def _factor(name, impact):
    return {"feature": name, "impact": impact, "direction": ""}


def test_recommended_action_uses_top_risk_increasing_factor():
    assert "contract" in pm._recommended_action([_factor("Contract=Month-to-month", 0.5)], "High").lower()
    assert "onboarding" in pm._recommended_action([_factor("tenure (short)", 0.4)], "Medium").lower()
    assert "discount" in pm._recommended_action([_factor("MonthlyCharges (high)", 0.2)], "High").lower()


def test_recommended_action_ignores_protective_factors_and_low_risk():
    # a factor that REDUCES risk must not trigger a retention offer
    assert pm._recommended_action([_factor("Contract=Month-to-month", -0.5)], "High") == pm.DEFAULT_ACTIONS["High"]
    # rules are keyed by risk level: Low never gets an offer
    assert pm._recommended_action([_factor("Contract=Month-to-month", 0.9)], "Low") == pm.DEFAULT_ACTIONS["Low"]


def test_rules_table_is_json_friendly_and_complete():
    table = pm.rules_table()
    assert all({"when_top_factor_is", "risk_levels", "action"} <= set(row) for row in table)
    assert len(table) == len(pm.RECOMMENDATION_RULES) + 3


def test_predict_end_to_end_linear(with_model):
    result = pm.predict(HIGH_RISK)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] == pm.risk_level_for(result["churn_probability"])
    assert result["model_version"] == "tiny_lr_v1"
    assert 1 <= len(result["key_factors"]) <= 5
    assert all({"feature", "impact", "direction"} <= set(f) for f in result["key_factors"])
    impacts = [abs(f["impact"]) for f in result["key_factors"]]
    assert impacts == sorted(impacts, reverse=True)


def test_predict_handles_missing_optional_features(with_model):
    result = pm.predict({"tenure": 12, "MonthlyCharges": 50.0, "Contract": "Two year"})
    assert result["risk_level"] in ("Low", "Medium", "High")


def test_month_to_month_short_tenure_scores_higher_than_two_year_long_tenure(with_model):
    risky = pm.predict(HIGH_RISK)["churn_probability"]
    safe = pm.predict({"tenure": 70, "MonthlyCharges": 40.0, "TotalCharges": 2800.0, "Contract": "Two year"})[
        "churn_probability"
    ]
    assert risky > safe


def test_global_feature_importance_sorted(with_model):
    items = pm.global_feature_importance(top_n=5)
    assert len(items) == 5
    assert [i["importance"] for i in items] == sorted((i["importance"] for i in items), reverse=True)


@pytest.fixture(scope="module")
def fitted_models():
    X, y, _, _ = prepare_training_frame(_generate_synthetic(500))
    pipe = build_feature_pipeline()
    Xt = pipe.fit_transform(X, y)
    models = {
        "lr": LogisticRegression(max_iter=300).fit(Xt, y),
        "rf": RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0).fit(Xt, y),
        "xgb": XGBClassifier(n_estimators=30, max_depth=3, eval_metric="logloss").fit(Xt, y),
    }
    return models, Xt, get_feature_names(pipe)


@pytest.mark.parametrize("kind", ["lr", "rf", "xgb"])
def test_contributions_and_key_factors_for_every_model_type(fitted_models, kind):
    """Covers the SHAP paths for Random Forest (3-D output) and XGBoost (2-D output) as well as linear models."""
    models, Xt, names = fitted_models
    contrib = pm._contributions(models[kind], Xt[:1])
    assert contrib.shape == (Xt.shape[1],)
    assert np.isfinite(contrib).all()
    factors = pm._top_key_factors(models[kind], Xt[:1], names, {"Contract": "Month-to-month"})
    assert factors and all(np.isfinite(f["impact"]) for f in factors)


def test_global_importance_for_tree_model(fitted_models, tiny_artifacts):
    models, _, names = fitted_models
    pm.reload_artifacts()
    pm._artifacts_cache.update(
        model=models["rf"], pipeline=tiny_artifacts["pipeline"], metadata={"feature_names": names}
    )
    try:
        items = pm.global_feature_importance(3)
        assert len(items) == 3 and all(i["importance"] >= 0 for i in items)
    finally:
        pm.reload_artifacts()


def test_shipped_model_artifacts_load_and_score():
    """The model bundled in the repository must load and produce sane output."""
    import os

    if not os.path.exists(os.path.join(pm.MODEL_DIR, "model.pkl")):
        pytest.skip("no trained model bundled")
    pm.reload_artifacts()
    result = pm.predict(HIGH_RISK)
    assert result["risk_level"] == "High"
    assert pm.get_metadata()["model_version"]
    pm.reload_artifacts()


def test_predict_applies_the_calibrator_when_present(with_model):
    class Halver:
        def transform(self, probs):
            return [p / 2 for p in probs]

    raw = pm.predict(HIGH_RISK)["churn_probability"]
    pm._artifacts_cache["calibrator"] = Halver()
    try:
        assert abs(pm.predict(HIGH_RISK)["churn_probability"] - raw / 2) < 1e-3
    finally:
        pm._artifacts_cache["calibrator"] = None


def test_total_charges_is_not_a_model_feature():
    from app.ml.pipeline import NUMERIC_FEATURES

    assert "TotalCharges" not in NUMERIC_FEATURES


def test_predict_frame_matches_single_predict(with_model):
    rows = [
        HIGH_RISK,
        {"tenure": 60, "MonthlyCharges": 40.0, "Contract": "Two year", "InternetService": "DSL", "TechSupport": "Yes"},
        {"tenure": 12, "MonthlyCharges": 20.0, "Contract": "One year", "InternetService": "No"},
    ]
    batch = pm.predict_frame(pm.features_to_frame(rows))
    for row, got in zip(rows, batch):
        want = pm.predict(row)
        assert got["churn_probability"] == pytest.approx(want["churn_probability"], abs=1e-4)
        assert got["risk_level"] == want["risk_level"]
        assert [f["feature"] for f in got["key_factors"]] == [f["feature"] for f in want["key_factors"]]
    assert pm.predict_frame(pm.features_to_frame([])) == []
    fast = pm.predict_frame(pm.features_to_frame(rows[:1]), factors=False)
    assert fast[0]["key_factors"] == [] and fast[0]["risk_level"] in ("Low", "Medium", "High")
