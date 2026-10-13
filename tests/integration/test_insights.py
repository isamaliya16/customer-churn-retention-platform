"""Campaign planner, segment explorer, outcomes and bulk actions."""

from __future__ import annotations


def _score(client, headers, make_customer, n=3, **overrides):
    ids = [make_customer(**overrides)["id"] for _ in range(n)]
    client.post("/predict/batch", json={"customer_ids": ids}, headers=headers)
    return ids


def test_campaign_math_and_budget(client, admin_headers, make_customer, mocked_predict):
    _score(client, admin_headers, make_customer, n=4)  # p=0.82, $70/month each
    body = client.get(
        "/analytics/campaign?offer_cost=20&success_rate=0.5&horizon_months=12", headers=admin_headers
    ).get_json()
    assert body["targeted_customers"] == 4 and body["total_cost"] == 80.0
    # revenue saved = 4 x 0.82 x 0.5 x 70 x 12 = 1377.6
    assert body["expected_revenue_saved"] == 1377.6 and body["net_value"] == 1297.6
    assert body["roi"] == round(1297.6 / 80, 2) and len(body["gain_curve"]) == 20
    capped = client.get("/analytics/campaign?offer_cost=20&budget=45", headers=admin_headers).get_json()
    assert capped["targeted_customers"] == 2  # floor(45 / 20)
    pricey = client.get("/analytics/campaign?offer_cost=5000", headers=admin_headers).get_json()
    assert pricey["targeted_customers"] == 0  # never contact when the expected value is negative


def test_campaign_empty_and_validation(client, admin_headers):
    assert client.get("/analytics/campaign", headers=admin_headers).get_json()["targeted_customers"] == 0
    assert client.get("/analytics/campaign?success_rate=2", headers=admin_headers).status_code == 422


def test_explorer_dimensions(client, admin_headers, make_customer, mocked_predict):
    _score(client, admin_headers, make_customer, n=2, contract_type="one-year")
    make_customer(contract_type="two-year", senior_citizen=1)
    body = client.get("/analytics/explorer?dimension=contract_type", headers=admin_headers).get_json()
    by = {s["segment"]: s for s in body["segments"]}
    assert by["one-year"]["customers"] == 2 and by["one-year"]["high_risk"] == 2
    assert by["one-year"]["revenue_at_risk"] == 140.0 and by["two-year"]["avg_churn_probability"] is None
    senior = client.get("/analytics/explorer?dimension=senior_citizen", headers=admin_headers).get_json()
    assert "Senior" in {s["segment"] for s in senior["segments"]}
    assert client.get("/analytics/explorer?dimension=bogus", headers=admin_headers).status_code == 422
    assert client.get("/analytics/explorer?dimension=tenure_bucket", headers=admin_headers).status_code == 200


def test_bulk_actions_are_idempotent_and_close_with_outcome(client, admin_headers, make_customer, mocked_predict):
    ids = _score(client, admin_headers, make_customer, n=3)
    churned = make_customer()["id"]
    client.put(f"/customers/{churned}", json={"is_active": False}, headers=admin_headers)
    payload = {"customer_ids": [*ids, churned, 999999], "offer_cost": 15, "due_date": "2030-01-31"}
    first = client.post("/actions/bulk", json=payload, headers=admin_headers).get_json()
    assert first == {"requested": 5, "created": 3, "skipped": 2}
    again = client.post("/actions/bulk", json=payload, headers=admin_headers).get_json()
    assert again["created"] == 0  # open actions already exist

    listing = client.get("/actions", headers=admin_headers).get_json()["items"]
    assert listing[0]["action"] == "Offer a 12-month contract discount" and listing[0]["due_date"] == "2030-01-31"
    resp = client.put(f"/actions/{listing[0]['id']}", json={"outcome": "retained"}, headers=admin_headers)
    assert resp.get_json()["status"] == "done" and resp.get_json()["completed_at"]
    assert (
        client.put(f"/actions/{listing[1]['id']}", json={"outcome": "maybe"}, headers=admin_headers).status_code == 422
    )

    out = client.get("/analytics/outcomes", headers=admin_headers).get_json()
    assert out["retained"] == 1 and out["lost"] == 0 and out["success_rate"] == 1.0
    assert out["monthly_revenue_retained"] == 70.0 and out["total_offer_cost"] == 45.0
    assert out["by_action"][0]["count"] == 3


def test_outcomes_empty(client, admin_headers):
    body = client.get("/analytics/outcomes", headers=admin_headers).get_json()
    assert body["total_actions"] == 0 and body["success_rate"] is None


def test_new_endpoints_require_auth(client):
    for url in ("/analytics/campaign", "/analytics/explorer", "/analytics/outcomes"):
        assert client.get(url).status_code == 401
    assert client.post("/actions/bulk", json={"customer_ids": [1]}).status_code == 401
