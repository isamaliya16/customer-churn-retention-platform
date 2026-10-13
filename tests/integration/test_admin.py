"""Integration tests: /admin/* (users, logs, activity)."""


def test_non_admin_cannot_access_admin_endpoints(client, analyst_headers):
    for path in ["/admin/users", "/admin/logs", "/admin/activity"]:
        resp = client.get(path, headers=analyst_headers)
        assert resp.status_code == 403, path


def test_admin_can_list_and_create_users(client, admin_headers):
    resp = client.post(
        "/admin/users",
        json={"email": "new@example.com", "password": "password123", "role": "analyst"},
        headers=admin_headers,
    )
    assert resp.status_code == 201
    listed = client.get("/admin/users", headers=admin_headers).get_json()
    assert listed["total"] == 2  # the admin fixture itself + the new user


def test_admin_cannot_demote_or_deactivate_self(client, admin_headers):
    me = client.get("/auth/me", headers=admin_headers).get_json()
    demote = client.put(f"/admin/users/{me['id']}", json={"role": "analyst"}, headers=admin_headers)
    assert demote.status_code == 400 and demote.get_json()["error"]["code"] == "validation_error"
    deactivate = client.put(f"/admin/users/{me['id']}", json={"is_active": False}, headers=admin_headers)
    assert deactivate.status_code == 400


def test_admin_cannot_remove_last_admin(client, admin_headers):
    me = client.get("/auth/me", headers=admin_headers).get_json()
    second = client.post(
        "/admin/users",
        json={"email": "second@example.com", "password": "password123", "role": "admin"},
        headers=admin_headers,
    ).get_json()
    client.put(
        f"/admin/users/{second['id']}", json={"role": "analyst"}, headers=admin_headers
    )  # ok, first admin still exists
    resp = client.put(f"/admin/users/{me['id']}", json={"is_active": False}, headers=admin_headers)
    assert resp.status_code == 400  # blocked: this is now the LAST active admin


def test_admin_can_reset_a_users_password(client, admin_headers):
    user = client.post(
        "/admin/users",
        json={"email": "reset@example.com", "password": "password123", "role": "analyst"},
        headers=admin_headers,
    ).get_json()
    client.put(f"/admin/users/{user['id']}", json={"password": "brandnewpass1"}, headers=admin_headers)
    assert (
        client.post("/auth/login", json={"email": "reset@example.com", "password": "brandnewpass1"}).status_code == 200
    )


def test_admin_logs_and_activity(client, admin_headers, make_customer):
    make_customer()
    logs = client.get("/admin/logs", headers=admin_headers).get_json()
    assert logs["total"] > 0
    filtered = client.get("/admin/logs?method=POST&status_class=2", headers=admin_headers).get_json()
    assert all(item["method"] == "POST" for item in filtered["items"])
    activity = client.get("/admin/activity?days=7", headers=admin_headers).get_json()
    assert activity["total_requests"] > 0 and "top_endpoints" in activity
