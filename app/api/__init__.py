"""Registers the flask-smorest Api (OpenAPI 3 + Swagger UI at /docs) and every domain blueprint."""

from flask import Flask
from flask_smorest import Api


def register_api(app: Flask) -> None:
    app.config.setdefault(
        "API_SPEC_OPTIONS",
        {
            "components": {
                "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}}
            },
            "security": [{"bearerAuth": []}],
            "info": {
                "description": (
                    "Churn prediction and retention platform. Click **Authorize**, paste the `access_token` from "
                    "`POST /auth/login`, then try any endpoint. Every error uses the envelope "
                    '`{"error": {"code", "message", "details"}}`.'
                )
            },
        },
    )
    api = Api(app)

    from app.api.actions import blp as actions_blp
    from app.api.admin import blp as admin_blp
    from app.api.analytics import blp as analytics_blp
    from app.api.auth import blp as auth_blp
    from app.api.customers import blp as customers_blp
    from app.api.exports import blp as exports_blp
    from app.api.health import blp as health_blp
    from app.api.imports import blp as imports_blp
    from app.api.model import blp as model_blp
    from app.api.predictions import blp as predictions_blp

    for blueprint in (
        health_blp,
        auth_blp,
        customers_blp,
        predictions_blp,
        analytics_blp,
        actions_blp,
        model_blp,
        exports_blp,
        admin_blp,
        imports_blp,
    ):
        api.register_blueprint(blueprint)
