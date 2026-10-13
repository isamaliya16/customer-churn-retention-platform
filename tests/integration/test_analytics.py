"""Integration tests: /analytics/*, /model/info, /export/*, /health, /docs, /dashboard."""


def test_health_liveness(client):
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.get_json()["status"] == "ok"


def test_health_readiness(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200 and resp.get_json()["checks"]["database"] == "ok"


def test_analytics_overview_requires_auth(client):
    assert client.get("/analytics/overview").status_code == 401


def test_analytics_overview_empty_state(client, admin_headers):
    body = client.get("/analytics/overview", headers=admin_headers).get_json()
    assert body["total_customers"] == 0 and body["churn_rate"] == 0.0


def test_analytics_overview_reflects_created_customer(client, admin_headers, make_customer):
    make_customer()
    body = client.get("/analytics/overview", headers=admin_headers).get_json()
    assert body["total_customers"] == 1 and body["active_customers"] == 1


def test_analytics_overview_custom_period(client, admin_headers, make_customer):
    make_customer()
    resp = client.get("/analytics/overview?days=7", headers=admin_headers)
    assert resp.status_code == 200 and resp.get_json()["period_days"] == 7
    assert client.get("/analytics/overview?days=0", headers=admin_headers).status_code == 422


def test_analytics_segments_by_contract_type(client, admin_headers, make_customer):
    make_customer(contract_type="month-to-month")
    body = client.get("/analytics/segments", headers=admin_headers).get_json()
    assert any(s["segment"] == "month-to-month" for s in body["by_contract_type"])


def test_analytics_risk_distribution_and_revenue_at_risk(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    dist = client.get("/analytics/risk-distribution", headers=admin_headers).get_json()
    assert dist["counts"]["High"] == 1
    rev = client.get("/analytics/revenue-at-risk", headers=admin_headers).get_json()
    assert rev["total"] == c["monthly_charges"]


def test_risk_distribution_counts_active_customers_only(client, admin_headers, make_customer, mocked_predict):
    scored, churned_later = make_customer(), make_customer()
    make_customer()  # a third customer that is never scored
    client.post("/predict", json={"customer_id": scored["id"]}, headers=admin_headers)
    client.post("/predict", json={"customer_id": churned_later["id"]}, headers=admin_headers)
    client.put(f"/customers/{churned_later['id']}", json={"is_active": False}, headers=admin_headers)
    dist = client.get("/analytics/risk-distribution", headers=admin_headers).get_json()
    assert dist["counts"]["High"] == 1 and dist["unscored"] == 1 and dist["total_customers"] == 2


def test_analytics_high_risk_list_and_sort(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    body = client.get("/analytics/high-risk", headers=admin_headers).get_json()
    assert body["total"] == 1 and body["items"][0]["customer_id"] == c["id"]


def test_analytics_trends_endpoint_shape(client, admin_headers):
    resp = client.get("/analytics/trends?days=30", headers=admin_headers)
    assert resp.status_code == 200 and "weekly" in resp.get_json()


def test_write_endpoints_invalidate_analytics_cache(client, admin_headers, make_customer, mocked_predict):
    first = client.get("/analytics/overview", headers=admin_headers).get_json()
    assert first["total_customers"] == 0
    make_customer()
    second = client.get("/analytics/overview", headers=admin_headers).get_json()
    assert second["total_customers"] == 1  # would be stale (0) if the cache wasn't invalidated


def test_model_info_without_trained_model(client, admin_headers, monkeypatch):
    from app.ml import predict as pm

    monkeypatch.setattr(pm, "MODEL_DIR", "/tmp/definitely-no-model-here")
    pm.reload_artifacts()
    try:
        resp = client.get("/model/info", headers=admin_headers)
        assert resp.status_code == 400 and resp.get_json()["error"]["code"] == "model_not_trained"
    finally:
        pm.reload_artifacts()


def test_model_info_with_trained_model(client, admin_headers, with_model):
    body = client.get("/model/info", headers=admin_headers).get_json()
    assert body["model_version"] == "tiny_lr_v1"
    assert body["held_out_metrics"]["logistic_regression"]["recall"] == 0.78
    assert body["confusion_matrix"]["matrix"] == [[70, 20], [10, 20]]
    assert len(body["feature_importance"]) > 0
    assert body["risk_thresholds"] == {"low_max": 0.25, "medium_max": 0.50}
    assert any(r["action"] for r in body["recommendation_rules"])


def test_export_customers_and_predictions_csv(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)
    cust_csv = client.get("/export/customers.csv", headers=admin_headers)
    assert cust_csv.status_code == 200 and cust_csv.mimetype == "text/csv"
    assert "attachment" in cust_csv.headers["Content-Disposition"]
    assert c["email"] in cust_csv.get_data(as_text=True)
    pred_csv = client.get("/export/predictions.csv", headers=admin_headers)
    assert pred_csv.status_code == 200 and "High" in pred_csv.get_data(as_text=True)
    high_csv = client.get("/export/high-risk.csv", headers=admin_headers)
    assert high_csv.status_code == 200 and c["email"] in high_csv.get_data(as_text=True)


def test_swagger_docs_and_openapi_available(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    body = spec.get_json()
    assert body["openapi"].startswith("3.")
    assert "/predict" in body["paths"] and "/customers" in body["paths"]
    assert body["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"


def test_dashboard_is_served_at_root_and_dashboard_path(client):
    root = client.get("/")
    assert root.status_code in (301, 302, 308)
    dash = client.get("/dashboard")
    assert dash.status_code == 200 and b"Retention" in dash.data


def test_strict_csp_on_api_relaxed_csp_on_docs_and_dashboard(client):
    api_csp = client.get("/health").headers.get("Content-Security-Policy", "")
    assert "unsafe-inline" not in api_csp
    dash_csp = client.get("/dashboard").headers.get("Content-Security-Policy", "")
    assert "cdn.jsdelivr.net" in dash_csp
    docs_csp = client.get("/docs").headers.get("Content-Security-Policy", "")
    assert "cdn.jsdelivr.net" in docs_csp


def test_security_headers_present(client):
    headers = client.get("/health").headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "SAMEORIGIN"
