"""Integration tests: /actions."""


def test_create_action_from_a_prediction(client, admin_headers, make_customer, mocked_predict):
    c = make_customer()
    prediction = client.post("/predict", json={"customer_id": c["id"]}, headers=admin_headers).get_json()
    resp = client.post(
        "/actions",
        json={"customer_id": c["id"], "prediction_id": prediction["id"], "action": prediction["recommended_action"]},
        headers=admin_headers,
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["status"] == "open" and body["customer_name"] == c["name"]


def test_action_for_unknown_customer_is_404(client, admin_headers):
    resp = client.post("/actions", json={"customer_id": 424242, "action": "Call"}, headers=admin_headers)
    assert resp.status_code == 404


def test_action_prediction_must_belong_to_customer(client, admin_headers, make_customer, mocked_predict):
    a = make_customer()
    b = make_customer()
    prediction = client.post("/predict", json={"customer_id": a["id"]}, headers=admin_headers).get_json()
    resp = client.post(
        "/actions",
        json={"customer_id": b["id"], "prediction_id": prediction["id"], "action": "x"},
        headers=admin_headers,
    )
    assert resp.status_code == 400 and resp.get_json()["error"]["code"] == "validation_error"


def test_action_status_lifecycle_and_completed_at(client, admin_headers, make_customer):
    c = make_customer()
    action = client.post("/actions", json={"customer_id": c["id"], "action": "Call"}, headers=admin_headers).get_json()
    done = client.put(f"/actions/{action['id']}", json={"status": "done"}, headers=admin_headers).get_json()
    assert done["status"] == "done" and done["completed_at"] is not None
    reopened = client.put(f"/actions/{action['id']}", json={"status": "open"}, headers=admin_headers).get_json()
    assert reopened["completed_at"] is None


def test_list_and_filter_actions(client, admin_headers, make_customer):
    c = make_customer()
    client.post("/actions", json={"customer_id": c["id"], "action": "A"}, headers=admin_headers)
    body = client.get("/actions?status=open", headers=admin_headers).get_json()
    assert body["total"] == 1
    assert client.get("/actions?status=done", headers=admin_headers).get_json()["total"] == 0


def test_analyst_cannot_delete_action_admin_can(client, admin_headers, analyst_headers, make_customer):
    c = make_customer()
    action = client.post("/actions", json={"customer_id": c["id"], "action": "A"}, headers=admin_headers).get_json()
    assert client.delete(f"/actions/{action['id']}", headers=analyst_headers).status_code == 403
    assert client.delete(f"/actions/{action['id']}", headers=admin_headers).status_code == 204
    assert client.get(f"/actions/{action['id']}", headers=admin_headers).status_code == 404
