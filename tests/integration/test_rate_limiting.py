"""Rate limiting is disabled in the shared `testing` config (RATELIMIT_ENABLED=False) so the rest of the
suite runs fast without hitting limits mid-test. This file spins up its OWN app instance with rate
limiting turned back on to verify the limiter actually works and returns the standard envelope."""

from unittest.mock import patch

import pytest

from app import create_app
from app.config import TestingConfig
from app.extensions import db as _db
from tests.conftest import FAKE_RESULT


class RateLimitedTestingConfig(TestingConfig):
    RATELIMIT_ENABLED = True
    RATE_LIMIT_LOGIN = "3 per minute"
    RATE_LIMIT_PREDICT = "5 per minute"
    RATE_LIMIT_BATCH = "2 per minute"


@pytest.fixture
def rl_app(monkeypatch):
    monkeypatch.setitem(
        __import__("app.config", fromlist=["config_by_name"]).config_by_name, "testing", RateLimitedTestingConfig
    )
    application = create_app("testing")
    with application.app_context():
        _db.create_all()
        from app.models.user import User

        user = User(email="rl-admin@example.com", role="admin")
        user.set_password("password123")
        _db.session.add(user)
        _db.session.commit()
    yield application
    with application.app_context():
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def rl_client(rl_app):
    return rl_app.test_client()


@pytest.fixture
def rl_headers(rl_client):
    resp = rl_client.post("/auth/login", json={"email": "rl-admin@example.com", "password": "password123"})
    return {"Authorization": f"Bearer {resp.get_json()['access_token']}"}


def test_login_rate_limit_returns_standard_envelope_with_retry_after(rl_client):
    for _ in range(3):
        rl_client.post("/auth/login", json={"email": "nobody@example.com", "password": "x"})
    limited = rl_client.post("/auth/login", json={"email": "nobody@example.com", "password": "x"})
    assert limited.status_code == 429
    assert limited.get_json()["error"]["code"] == "rate_limited"
    assert "Retry-After" in limited.headers


def test_predict_rate_limit(rl_client, rl_headers):
    resp = rl_client.post(
        "/customers",
        json={"name": "N", "email": "n@example.com", "monthly_charges": 10, "contract_type": "one-year"},
        headers=rl_headers,
    )
    customer_id = resp.get_json()["id"]
    with patch("app.services.prediction_service.ml_predict", return_value=dict(FAKE_RESULT)):
        codes = [
            rl_client.post("/predict", json={"customer_id": customer_id}, headers=rl_headers).status_code
            for _ in range(8)
        ]
    assert codes.count(429) > 0
    assert codes[:5].count(200) == 5  # the first 5 (the configured limit) succeed


def test_batch_predict_rate_limit(rl_client, rl_headers):
    with patch("app.services.prediction_service.ml_predict", return_value=dict(FAKE_RESULT)):
        codes = [
            rl_client.post("/predict/batch", json={"scope": "all"}, headers=rl_headers).status_code for _ in range(4)
        ]
    assert 429 in codes
