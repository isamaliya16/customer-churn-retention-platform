"""Application factory."""

from __future__ import annotations

import logging
import os
import time
import uuid

from flask import Flask, g, redirect, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import config_by_name
from app.extensions import cache, cors, db, jwt, limiter, migrate, talisman
from app.utils.errors import APIError, error_envelope
from app.utils.logging import configure_logging

# Pages that legitimately need inline scripts / CDN assets get a looser CSP.
RELAXED_CSP = {
    "/docs": (
        "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://cdn.jsdelivr.net; "
        "connect-src 'self'"
    ),
    "/dashboard": (
        "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'"
    ),
}


def create_app(config_name: str | None = None) -> Flask:
    config_name = config_name or os.environ.get("FLASK_ENV") or "development"
    if config_name not in config_by_name:
        config_name = "development"
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    if app.config.get("TRUST_PROXY"):
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[method-assign]

    configure_logging()
    # Registered BEFORE Talisman so it runs AFTER Talisman (after_request hooks run in reverse order).
    _register_csp_override(app)
    _init_extensions(app, config_name)
    _register_jwt_callbacks()
    _register_blueprints(app)
    _register_dashboard(app)
    _register_error_handlers(app)
    _register_request_hooks(app)
    _register_cli(app)

    if config_name == "production":
        _assert_production_secrets(app)

    return app


WEAK_SECRET_PREFIXES = ("dev-", "change-me", "changeme", "secret", "password")


def _assert_production_secrets(app: Flask) -> None:
    """Refuse to boot in production with placeholder / short signing secrets (anyone could forge a token)."""
    for key in ("SECRET_KEY", "JWT_SECRET_KEY"):
        value = str(app.config.get(key) or "")
        if len(value) < 32 or value.lower().startswith(WEAK_SECRET_PREFIXES):
            raise RuntimeError(
                f"{key} is missing, a placeholder, or shorter than 32 characters. "
                'Set a long random value (e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`).'
            )


def _init_extensions(app: Flask, config_name: str) -> None:
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cache.init_app(app)
    cors.init_app(app, origins=app.config.get("CORS_ORIGINS") or [])
    limiter.init_app(app)

    # Talisman stays on even under TESTING (it only forces HTTPS/HSTS in production, so the
    # plain-http Flask test client is unaffected) so security headers are always verifiable.
    talisman.init_app(
        app,
        content_security_policy={"default-src": "'self'"},
        force_https=config_name == "production",
        strict_transport_security=config_name == "production",
    )


def _register_jwt_callbacks() -> None:
    """Make every Flask-JWT-Extended error use the standard error envelope."""

    @jwt.unauthorized_loader
    def _missing_token(reason: str):
        return error_envelope(code="unauthorized", message=reason), 401

    @jwt.invalid_token_loader
    def _invalid_token(reason: str):
        return error_envelope(code="unauthorized", message=f"Invalid token: {reason}"), 401

    @jwt.expired_token_loader
    def _expired_token(jwt_header, jwt_payload):
        return error_envelope(code="unauthorized", message="Token has expired."), 401

    @jwt.revoked_token_loader
    def _revoked_token(jwt_header, jwt_payload):
        return error_envelope(code="unauthorized", message="Token has been revoked."), 401

    @jwt.token_in_blocklist_loader
    def _is_revoked(jwt_header, jwt_payload) -> bool:
        from app.services.auth_service import AuthService

        return AuthService.is_revoked(jwt_payload["jti"])


def _register_blueprints(app: Flask) -> None:
    from app.api import register_api

    register_api(app)


def _register_dashboard(app: Flask) -> None:
    """Serve the static dashboard from the API itself (same origin => no CORS setup needed)."""
    dashboard_dir = os.path.abspath(os.path.join(app.root_path, "..", "dashboard"))

    @app.route("/")
    @limiter.exempt
    def index():
        return redirect("/dashboard")

    @app.route("/dashboard")
    @limiter.exempt
    def dashboard():
        return send_from_directory(dashboard_dir, "index.html")


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(err: APIError):
        return err.to_envelope(), err.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(err: HTTPException):
        # flask-smorest/webargs put field-level validation errors in `err.data["messages"]`.
        details = None
        data = getattr(err, "data", None)
        if isinstance(data, dict) and "messages" in data:
            details = data["messages"]
        code = (err.name or "http_error").lower().replace(" ", "_")
        message = err.description or str(err)
        status = err.code or 500
        if status == 422:
            code, message = "validation_error", "The request failed validation."
        elif status == 429:
            code, message = "rate_limited", "Too many requests. Please slow down."
        return error_envelope(code=code, message=message, details=details), status

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        app.logger.exception("Unhandled exception")
        db.session.rollback()
        return error_envelope(code="internal_error", message="An unexpected error occurred."), 500


def _register_csp_override(app: Flask) -> None:
    @app.after_request
    def _relax_csp(response):
        for prefix, policy in RELAXED_CSP.items():
            if request.path == prefix or request.path.startswith(prefix + "/"):
                response.headers["Content-Security-Policy"] = policy
        return response


def _register_request_hooks(app: Flask) -> None:
    """Request id + timer; write an APILog row per request (audit trail)."""

    @app.before_request
    def _start_timer():
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()

    @app.after_request
    def _finalize(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        if not request.path.startswith("/health"):
            logging.getLogger("app.request").info(
                "request",
                extra={
                    "extra_fields": {
                        "request_id": getattr(g, "request_id", ""),
                        "method": request.method,
                        "path": request.path,
                        "status": response.status_code,
                        "latency_ms": int((time.time() - getattr(g, "start_time", time.time())) * 1000),
                    }
                },
            )
        try:
            _write_api_log(response)
        except Exception:  # never let audit logging break a response
            db.session.rollback()
            app.logger.exception("Failed to write APILog entry")
        return response

    def _write_api_log(response):
        from flask_jwt_extended import get_jwt, verify_jwt_in_request

        from app.models.api_log import APILog

        if request.path.startswith(("/health", "/dashboard", "/docs", "/openapi")) or request.path == "/":
            return

        latency_ms = int((time.time() - getattr(g, "start_time", time.time())) * 1000)
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            sub = (get_jwt() or {}).get("sub")
            user_id = int(sub) if sub is not None else None
        except Exception:
            user_id = None

        db.session.add(
            APILog(
                user_id=user_id,
                endpoint=request.path[:255],
                method=request.method,
                status_code=response.status_code,
                latency_ms=latency_ms,
                ip_address=(request.headers.get("X-Forwarded-For", request.remote_addr) or "")[:64],
            )
        )
        db.session.commit()


def _register_cli(app: Flask) -> None:
    from app.cli import register_cli

    register_cli(app)
