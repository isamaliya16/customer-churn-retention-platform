"""Integration tests: /customers via the Flask test client."""


def test_create_customer_minimal_payload(client, admin_headers):
    resp = client.post(
        "/customers",
        json={"name": "Min", "email": "min@example.com", "monthly_charges": 40.0, "contract_type": "one-year"},
        headers=admin_headers,
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["external_id"].startswith("CUST-") and body["is_active"] is True and body["latest_risk_level"] is None


def test_create_customer_full_payload(client, admin_headers, sample_customer_payload):
    resp = client.post("/customers", json=sample_customer_payload, headers=admin_headers)
    assert resp.status_code == 201 and resp.get_json()["email"] == sample_customer_payload["email"]


def test_create_customer_missing_required_field_returns_422(client, admin_headers):
    resp = client.post("/customers", json={"name": "Incomplete"}, headers=admin_headers)
    assert resp.status_code == 422 and "json" in resp.get_json()["error"]["details"]


def test_create_customer_invalid_contract_type_returns_422(client, admin_headers, sample_customer_payload):
    resp = client.post(
        "/customers", json=dict(sample_customer_payload, contract_type="lifetime"), headers=admin_headers
    )
    assert resp.status_code == 422


def test_create_customer_invalid_profile_value_returns_422(client, admin_headers, sample_customer_payload):
    resp = client.post("/customers", json=dict(sample_customer_payload, gender="Other"), headers=admin_headers)
    assert resp.status_code == 422


def test_create_duplicate_email_returns_409(client, admin_headers, sample_customer_payload):
    client.post("/customers", json=sample_customer_payload, headers=admin_headers)
    dup = dict(sample_customer_payload, external_id="DIFFERENT-ID")
    resp = client.post("/customers", json=dup, headers=admin_headers)
    assert resp.status_code == 409


def test_analyst_can_create_but_not_delete(client, analyst_headers, make_customer):
    c = make_customer()
    resp = client.delete(f"/customers/{c['id']}", headers=analyst_headers)
    assert resp.status_code == 403 and resp.get_json()["error"]["code"] == "forbidden"


def test_get_customer_by_id_and_404(client, admin_headers, make_customer):
    c = make_customer()
    assert client.get(f"/customers/{c['id']}", headers=admin_headers).get_json()["id"] == c["id"]
    resp = client.get("/customers/999999", headers=admin_headers)
    assert resp.status_code == 404 and resp.get_json()["error"]["code"] == "not_found"


def test_update_customer_partial(client, admin_headers, make_customer):
    c = make_customer()
    resp = client.put(f"/customers/{c['id']}", json={"monthly_charges": 150.0}, headers=admin_headers)
    assert resp.status_code == 200 and resp.get_json()["monthly_charges"] == 150.0


def test_update_customer_deactivation_records_churn_date(client, admin_headers, make_customer):
    c = make_customer()
    resp = client.put(f"/customers/{c['id']}", json={"is_active": False}, headers=admin_headers)
    body = resp.get_json()
    assert body["is_active"] is False and body["churned_at"] is not None


def test_update_customer_email_conflict(client, admin_headers, make_customer):
    a = make_customer()
    b = make_customer()
    resp = client.put(f"/customers/{b['id']}", json={"email": a["email"]}, headers=admin_headers)
    assert resp.status_code == 409


def test_admin_can_delete_customer(client, admin_headers, make_customer):
    c = make_customer()
    assert client.delete(f"/customers/{c['id']}", headers=admin_headers).status_code == 204
    assert client.get(f"/customers/{c['id']}", headers=admin_headers).status_code == 404


def test_list_customers_pagination(client, admin_headers, make_customer):
    for _ in range(5):
        make_customer()
    body = client.get("/customers?page=1&per_page=2", headers=admin_headers).get_json()
    assert len(body["items"]) == 2 and body["total"] == 5 and body["total_pages"] == 3


def test_list_customers_filter_by_contract_type(client, admin_headers, make_customer):
    make_customer(contract_type="month-to-month")
    make_customer(contract_type="two-year")
    body = client.get("/customers?contract_type=two-year", headers=admin_headers).get_json()
    assert body["total"] == 1 and body["items"][0]["contract_type"] == "two-year"


def test_list_customers_search_by_name_email_external_id(client, admin_headers, make_customer):
    target = make_customer(name="Zara Unique", email="findme@example.com")
    for text in ["Zara", "findme", target["external_id"]]:
        body = client.get(f"/customers?q={text}", headers=admin_headers).get_json()
        assert body["total"] == 1 and body["items"][0]["id"] == target["id"]


def test_list_customers_sort(client, admin_headers, make_customer):
    make_customer(monthly_charges=10.0)
    make_customer(monthly_charges=90.0)
    body = client.get("/customers?sort_by=monthly_charges&order=desc", headers=admin_headers).get_json()
    assert [c["monthly_charges"] for c in body["items"]] == [90.0, 10.0]


def test_list_customers_filter_by_risk_level(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    make_customer()  # never scored
    client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers)  # mocked -> High
    body = client.get("/customers?risk_level=High", headers=admin_headers).get_json()
    assert body["total"] == 1 and body["items"][0]["id"] == c["id"]
    assert client.get("/customers?risk_level=Low", headers=admin_headers).get_json()["total"] == 0


def test_transactions_crud(client, admin_headers, make_customer):
    c = make_customer()
    assert client.get(f"/customers/{c['id']}/transactions", headers=admin_headers).get_json() == []
    resp = client.post(
        f"/customers/{c['id']}/transactions", json={"amount": 49.99, "type": "charge"}, headers=admin_headers
    )
    assert resp.status_code == 201 and resp.get_json()["amount"] == 49.99
    listed = client.get(f"/customers/{c['id']}/transactions", headers=admin_headers).get_json()
    assert len(listed) == 1
    bad = client.post(f"/customers/{c['id']}/transactions", json={"amount": -5}, headers=admin_headers)
    assert bad.status_code == 422
    assert client.get("/customers/999999/transactions", headers=admin_headers).status_code == 404
