"""Shared pytest fixtures.

* Every HTTP request gets its own app context / DB session (as in production).
* The ML model is replaced by a tiny model trained on synthetic data (`with_model`) or mocked
  (`mocked_predict`), so the suite never depends on `python -m app.ml.train` having been run.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from app import create_app
from app.extensions import db as _db
from app.models.user import User


@pytest.fixture(scope="session")
def app():
    application = create_app("testing")
    with application.app_context():
        _db.create_all()
    yield application
    with application.app_context():
        _db.session.remove()
        _db.drop_all()


@pytest.fixture(autouse=True)
def _clean_db(app):
    """Empty every table after each test so tests stay independent."""
    yield
    with app.app_context():
        _db.session.rollback()
        for table in reversed(_db.metadata.sorted_tables):
            _db.session.execute(table.delete())
        _db.session.commit()
        _db.session.remove()
    from app.extensions import cache

    with app.app_context():
        cache.clear()


@pytest.fixture
def client(app):
    return app.test_client()


def _make_user(app, email: str, password: str, role: str, active: bool = True) -> int:
    with app.app_context():
        user = User(email=email, role=role, is_active=active)
        user.set_password(password)
        _db.session.add(user)
        _db.session.commit()
        return user.id


def _login(client, email: str, password: str) -> dict:
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.get_json()
    return {"Authorization": f"Bearer {resp.get_json()['access_token']}"}


@pytest.fixture
def admin_headers(app, client):
    _make_user(app, "admin@test.com", "adminpass1", "admin")
    return _login(client, "admin@test.com", "adminpass1")


@pytest.fixture
def analyst_headers(app, client):
    _make_user(app, "analyst@test.com", "analystpass1", "analyst")
    return _login(client, "analyst@test.com", "analystpass1")


@pytest.fixture
def make_user(app):
    return lambda email, password="password123", role="analyst", active=True: _make_user(
        app, email, password, role, active
    )


@pytest.fixture
def login(client):
    return lambda email, password="password123": _login(client, email, password)


@pytest.fixture
def sample_customer_payload():
    return {
        "external_id": "CUST-TEST-0001",
        "name": "Test Customer",
        "email": "test.customer@example.com",
        "signup_date": date(2024, 1, 1).isoformat(),
        "tenure_months": 5,
        "monthly_charges": 80.0,
        "total_charges": 400.0,
        "contract_type": "month-to-month",
        "plan": "Standard",
    }


@pytest.fixture
def make_customer(client, admin_headers):
    """Create a customer through the API; returns the JSON body."""
    counter = {"n": 0}

    def _make(**overrides):
        counter["n"] += 1
        payload = {
            "name": f"Customer {counter['n']}",
            "email": f"customer{counter['n']}@example.com",
            "monthly_charges": 70.0,
            "tenure_months": 6,
            "contract_type": "month-to-month",
        }
        payload.update(overrides)
        resp = client.post("/customers", json=payload, headers=admin_headers)
        assert resp.status_code == 201, resp.get_json()
        return resp.get_json()

    return _make


FAKE_RESULT = {
    "churn_probability": 0.82,
    "risk_level": "High",
    "key_factors": [{"feature": "Contract=Month-to-month", "impact": 0.3, "direction": "increases churn risk"}],
    "recommended_action": "Offer a 12-month contract discount",
    "model_version": "test_model_v1",
}


@pytest.fixture
def mocked_predict():
    """Replace the ML call inside the prediction service with a fixed High-risk result."""
    with patch("app.services.prediction_service.ml_predict", return_value=dict(FAKE_RESULT)) as mock, patch(
        "app.services.prediction_service.ml_predict_frame",
        side_effect=lambda frame, **kw: [dict(FAKE_RESULT) for _ in range(len(frame))],
    ):
        yield mock


@pytest.fixture(scope="session")
def tiny_artifacts():
    """A small logistic-regression model + fitted pipeline trained on synthetic Telco-shaped data."""
    from sklearn.linear_model import LogisticRegression

    from app.ml.fetch_data import _generate_synthetic
    from app.ml.pipeline import build_feature_pipeline, get_feature_names, prepare_training_frame

    X, y, _, _ = prepare_training_frame(_generate_synthetic(600))
    pipeline = build_feature_pipeline()
    Xt = pipeline.fit_transform(X, y)
    model = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xt, y)
    names = get_feature_names(pipeline)
    metadata = {
        "model_version": "tiny_lr_v1",
        "algorithm": "logistic_regression",
        "trained_at": "2026-01-01T00:00:00Z",
        "n_train_rows": 480,
        "n_test_rows": 120,
        "feature_names": names,
        "selection_criteria": {"weights": {"recall": 0.4, "f1": 0.4, "roc_auc": 0.2}, "rationale": "test"},
        "cv_metrics": {
            "logistic_regression": {"recall": 0.8, "f1": 0.6, "roc_auc": 0.84, "accuracy": 0.75, "precision": 0.5}
        },
        "held_out_metrics": {
            "logistic_regression": {"recall": 0.78, "f1": 0.61, "roc_auc": 0.84, "accuracy": 0.74, "precision": 0.5}
        },
        "confusion_matrix": {"labels": ["stayed", "churned"], "matrix": [[70, 20], [10, 20]]},
        "roc_curves": {"logistic_regression": {"fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1]}},
        "outlier_report": {},
        "class_balance": {"churn_rate": 0.27},
    }
    return {"model": model, "pipeline": pipeline, "metadata": metadata}


@pytest.fixture
def with_model(tiny_artifacts):
    """Install the tiny model as the 'loaded' model for the duration of a test."""
    from app.ml import predict as predict_module

    predict_module.reload_artifacts()
    predict_module._artifacts_cache.update(tiny_artifacts)
    yield tiny_artifacts
    predict_module.reload_artifacts()
