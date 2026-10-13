"""Integration tests: /predict, /predict/batch, /predictions (model mocked)."""


def test_predict_by_customer_id_persists_and_returns_result(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    resp = client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["risk_level"] == "High" and body["customer_id"] == c["id"] and body["id"] is not None
    mocked_predict.assert_called_once()


def test_predict_by_raw_features_not_persisted(client, admin_headers, mocked_predict):
    resp = client.post(
        "/predict",
        json={"features": {"tenure": 1, "MonthlyCharges": 80.0, "Contract": "Month-to-month"}},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["customer_id"] is None and body["id"] is None


def test_predict_raw_features_reject_unknown_value(client, admin_headers):
    resp = client.post(
        "/predict",
        json={"features": {"tenure": 1, "MonthlyCharges": 80.0, "Contract": "Whenever"}},
        headers=admin_headers,
    )
    assert resp.status_code == 422


def test_predict_rejects_both_or_neither(client, admin_headers, make_customer):
    c = make_customer()
    both = client.post(
        "/predict",
        json={"customer_id": c["id"], "features": {"tenure": 1, "MonthlyCharges": 1, "Contract": "One year"}},
        headers=admin_headers,
    )
    assert both.status_code == 422
    neither = client.post("/predict", json={}, headers=admin_headers)
    assert neither.status_code == 422


def test_predict_nonexistent_customer_returns_404(client, admin_headers, mocked_predict):
    resp = client.post("/predict", json={"customer_id": 999999}, headers=admin_headers)
    assert resp.status_code == 404


def test_predict_without_a_trained_model_returns_400(client, admin_headers, make_customer, monkeypatch):
    from app.ml import predict as pm

    monkeypatch.setattr(pm, "MODEL_DIR", "/tmp/definitely-no-model-here")
    pm.reload_artifacts()
    c = make_customer()
    try:
        resp = client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
        assert resp.status_code == 400 and resp.get_json()["error"]["code"] == "model_not_trained"
    finally:
        pm.reload_artifacts()


def test_predict_uses_the_real_model_end_to_end(client, admin_headers, make_customer, with_model):
    c = make_customer(monthly_charges=95.0, tenure_months=1, contract_type="month-to-month")
    resp = client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert 0 <= body["churn_probability"] <= 1 and body["model_version"] == "tiny_lr_v1"
    assert body["key_factors"] and {"feature", "impact", "direction"} <= set(body["key_factors"][0])


def test_predict_batch_explicit_ids(client, admin_headers, make_customer, mocked_predict):
    ids = [make_customer()["id"] for _ in range(3)]
    resp = client.post("/predict/batch", json={"customer_ids": ids}, headers=admin_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["requested"] == 3 and body["scored"] == 3 and body["by_risk"]["High"] == 3


def test_predict_batch_scope_and_limit(client, admin_headers, make_customer, mocked_predict):
    for _ in range(5):
        make_customer()
    resp = client.post("/predict/batch", json={"scope": "active", "limit": 2}, headers=admin_headers)
    assert resp.status_code == 200 and resp.get_json()["scored"] == 2


def test_predict_batch_requires_exactly_one_of_ids_or_scope(client, admin_headers):
    assert client.post("/predict/batch", json={}, headers=admin_headers).status_code == 422
    assert (
        client.post("/predict/batch", json={"customer_ids": [1], "scope": "all"}, headers=admin_headers).status_code
        == 422
    )


# NOTE: rate limiting itself is covered in tests/integration/test_rate_limiting.py (needs its own app instance,
# since RATELIMIT_ENABLED is off in the shared testing config used by every other test here for speed).


def test_list_predictions_pagination_and_filter(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    for _ in range(3):
        client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    body = client.get(f"/predictions?customer_id={c['id']}&page=1&per_page=2", headers=admin_headers).get_json()
    assert len(body["items"]) == 2 and body["total"] == 3
    assert body["items"][0]["customer_name"] == c["name"]


def test_list_predictions_date_range_filter(client, admin_headers, make_customer, mocked_predict):
    import datetime

    c = make_customer()
    client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    today = datetime.date.today().isoformat()
    future = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    past = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    assert (
        client.get(f"/predictions?date_from={today}&date_to={future}", headers=admin_headers).get_json()["total"] == 1
    )
    assert client.get(f"/predictions?date_from={future}", headers=admin_headers).get_json()["total"] == 0
    assert client.get(f"/predictions?date_to={past}", headers=admin_headers).get_json()["total"] == 0


def test_get_prediction_by_id_and_404(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    created = client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers).get_json()
    resp = client.get(f"/predictions/{created['id']}", headers=admin_headers)
    assert resp.status_code == 200 and resp.get_json()["id"] == created["id"]
    assert client.get("/predictions/999999", headers=admin_headers).status_code == 404


def test_cannot_score_an_already_churned_customer(client, admin_headers, make_customer, with_model):
    customer = make_customer()
    client.put(f"/customers/{customer['id']}", json={"is_active": False}, headers=admin_headers)
    resp = client.post("/predict", json={"customer_id": customer["id"]}, headers=admin_headers)
    assert resp.status_code == 400 and resp.get_json()["error"]["code"] == "customer_inactive"
    assert client.get("/predictions", headers=admin_headers).get_json()["total"] == 0
