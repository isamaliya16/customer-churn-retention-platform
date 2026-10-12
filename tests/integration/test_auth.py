"""Integration tests: /auth/* via the Flask test client."""

import pytest

NEW_USER = {"email": "a@example.com", "password": "password123"}


@pytest.fixture
def public_signup(app):
    """Temporarily enable self-service sign-up (it is OFF by default)."""
    app.config["ALLOW_PUBLIC_REGISTRATION"] = True
    yield
    app.config["ALLOW_PUBLIC_REGISTRATION"] = False


def test_public_register_is_disabled_by_default(client):
    resp = client.post("/auth/register", json=NEW_USER)
    assert resp.status_code == 403 and resp.get_json()["error"]["code"] == "forbidden"


def test_stranger_cannot_register_then_read_customers(client):
    """Regression: an unauthenticated visitor must never obtain a token that can read customer data."""
    assert client.post("/auth/register", json=NEW_USER).status_code == 403
    assert client.post("/auth/login", json=NEW_USER).status_code == 401


def test_analyst_cannot_register_users(client, analyst_headers):
    resp = client.post("/auth/register", json=NEW_USER, headers=analyst_headers)
    assert resp.status_code == 403


def test_admin_can_register_analyst_and_returns_tokens(client, admin_headers):
    resp = client.post("/auth/register", json={**NEW_USER, "role": "analyst"}, headers=admin_headers)
    assert resp.status_code == 201
    body = resp.get_json()
    assert "access_token" in body and "refresh_token" in body


def test_admin_can_create_admin(client, admin_headers):
    resp = client.post(
        "/auth/register",
        json={"email": "second-admin@example.com", "password": "password123", "role": "admin"},
        headers=admin_headers,
    )
    assert resp.status_code == 201


def test_deactivated_admin_token_cannot_register(client, admin_headers, make_user, login):
    make_user("boss@example.com", "password123", "admin")
    boss = login("boss@example.com", "password123")
    users = client.get("/admin/users", headers=admin_headers).get_json()["items"]
    boss_id = next(u["id"] for u in users if u["email"] == "boss@example.com")
    client.put(f"/admin/users/{boss_id}", json={"is_active": False}, headers=admin_headers)
    assert client.post("/auth/register", json=NEW_USER, headers=boss).status_code == 403


def test_public_signup_flag_allows_analyst_but_never_admin(client, public_signup):
    assert client.post("/auth/register", json=NEW_USER).status_code == 201
    sneaky = {"email": "sneaky@example.com", "password": "password123", "role": "admin"}
    resp = client.post("/auth/register", json=sneaky)
    assert resp.status_code == 403 and resp.get_json()["error"]["code"] == "forbidden"


def test_register_duplicate_email_returns_409(client, admin_headers):
    client.post("/auth/register", json=NEW_USER, headers=admin_headers)
    resp = client.post("/auth/register", json=NEW_USER, headers=admin_headers)
    assert resp.status_code == 409 and resp.get_json()["error"]["code"] == "conflict"


def test_register_missing_fields_and_short_password_return_422(client, admin_headers):
    resp = client.post("/auth/register", json={"email": "bad@example.com"}, headers=admin_headers)
    assert resp.status_code == 422 and resp.get_json()["error"]["code"] == "validation_error"
    resp = client.post("/auth/register", json={"email": "bad2@example.com", "password": "short"}, headers=admin_headers)
    assert resp.status_code == 422


def test_login_success_and_wrong_password(client, make_user):
    make_user("login@example.com", "password123")
    ok = client.post("/auth/login", json={"email": "login@example.com", "password": "password123"})
    assert ok.status_code == 200 and "access_token" in ok.get_json()
    bad = client.post("/auth/login", json={"email": "login@example.com", "password": "notright"})
    assert bad.status_code == 401 and bad.get_json()["error"]["code"] == "unauthorized"


def test_login_deactivated_user_rejected(client, make_user):
    make_user("inactive@example.com", "password123", active=False)
    resp = client.post("/auth/login", json={"email": "inactive@example.com", "password": "password123"})
    assert resp.status_code == 401


def test_refresh_requires_a_refresh_token_not_an_access_token(client, make_user):
    make_user("refresh@example.com", "password123")
    login = client.post("/auth/login", json={"email": "refresh@example.com", "password": "password123"})
    tokens = login.get_json()
    assert (
        client.post("/auth/refresh", headers={"Authorization": f"Bearer {tokens['access_token']}"}).status_code == 401
    )
    refreshed = client.post("/auth/refresh", headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert refreshed.status_code == 200 and "access_token" in refreshed.get_json()


def test_protected_endpoint_requires_token(client):
    resp = client.get("/customers")
    assert resp.status_code == 401 and resp.get_json()["error"]["code"] == "unauthorized"


def test_protected_endpoint_with_malformed_token_returns_401(client):
    resp = client.get("/customers", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_me_endpoint(client, admin_headers):
    resp = client.get("/auth/me", headers=admin_headers)
    assert (
        resp.status_code == 200 and resp.get_json()["email"] == "admin@test.com" and resp.get_json()["role"] == "admin"
    )


def test_change_password_flow(client, admin_headers):
    resp = client.post(
        "/auth/change-password",
        json={"current_password": "adminpass1", "new_password": "newpassword1"},
        headers=admin_headers,
    )
    assert resp.status_code == 204
    assert client.post("/auth/login", json={"email": "admin@test.com", "password": "adminpass1"}).status_code == 401
    assert client.post("/auth/login", json={"email": "admin@test.com", "password": "newpassword1"}).status_code == 200


def test_change_password_wrong_current_returns_401(client, admin_headers):
    resp = client.post(
        "/auth/change-password",
        json={"current_password": "wrong", "new_password": "newpassword1"},
        headers=admin_headers,
    )
    assert resp.status_code == 401


def test_deactivated_user_loses_access_immediately_even_with_valid_token(client, admin_headers, make_user, login):
    make_user("victim@example.com", "password123", "analyst")
    victim_headers = login("victim@example.com", "password123")
    assert client.get("/customers", headers=victim_headers).status_code == 200
    users = client.get("/admin/users", headers=admin_headers).get_json()["items"]
    victim_id = next(u["id"] for u in users if u["email"] == "victim@example.com")
    client.put(f"/admin/users/{victim_id}", json={"is_active": False}, headers=admin_headers)
    resp = client.get("/customers", headers=victim_headers)  # same still-valid JWT, now rejected
    assert resp.status_code == 401


def _tokens(client, make_user, email="out@example.com"):
    make_user(email, "password123")
    return client.post("/auth/login", json={"email": email, "password": "password123"}).get_json()


def test_logout_revokes_the_access_token(client, make_user):
    tokens = _tokens(client, make_user)
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert client.get("/customers", headers=headers).status_code == 200
    assert client.post("/auth/logout", json={}, headers=headers).status_code == 204
    resp = client.get("/customers", headers=headers)
    assert resp.status_code == 401 and "revoked" in resp.get_json()["error"]["message"].lower()


def test_logout_can_also_revoke_the_refresh_token(client, make_user):
    tokens = _tokens(client, make_user, "out2@example.com")
    access = {"Authorization": f"Bearer {tokens['access_token']}"}
    refresh = {"Authorization": f"Bearer {tokens['refresh_token']}"}
    assert (
        client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]}, headers=access).status_code == 204
    )
    assert client.post("/auth/refresh", headers=refresh).status_code == 401


def test_logout_twice_is_harmless_and_requires_a_token(client, make_user):
    tokens = _tokens(client, make_user, "out3@example.com")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert client.post("/auth/logout", json={}, headers=headers).status_code == 204
    assert client.post("/auth/logout", json={}, headers=headers).status_code == 401  # already revoked
    assert client.post("/auth/logout", json={}).status_code == 401


def test_logout_rejects_garbage_refresh_token(client, make_user):
    tokens = _tokens(client, make_user, "out4@example.com")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    resp = client.post("/auth/logout", json={"refresh_token": "garbage"}, headers=headers)
    assert resp.status_code == 400


def test_production_refuses_placeholder_secrets(monkeypatch):
    from app import create_app

    monkeypatch.setenv("SECRET_KEY", "change-me")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        from app import _assert_production_secrets

        app = create_app("testing")
        app.config["SECRET_KEY"] = "change-me"
        _assert_production_secrets(app)


def test_production_accepts_long_random_secrets():
    from app import _assert_production_secrets, create_app

    app = create_app("testing")
    app.config["SECRET_KEY"] = "k" * 40
    app.config["JWT_SECRET_KEY"] = "z9" * 24
    _assert_production_secrets(app)  # no exception
