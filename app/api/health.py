"""GET /health (liveness) and /health/ready (DB + cache check)."""

from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint
from sqlalchemy import text

from app.extensions import cache, db

blp = Blueprint("health", __name__, url_prefix="/health", description="Liveness/readiness probes")


@blp.route("")
class Health(MethodView):
    def get(self):
        """Liveness probe - process is up and can respond."""
        return {"status": "ok"}, 200


@blp.route("/ready")
class HealthReady(MethodView):
    def get(self):
        """Readiness probe - checks database and Redis connectivity."""
        checks = {"database": "ok", "redis": "ok"}
        status_code = 200

        try:
            db.session.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            checks["database"] = f"error: {exc}"
            status_code = 503

        if current_app.config.get("REDIS_URL"):
            try:
                cache.get("__readiness_probe__")
            except Exception as exc:  # noqa: BLE001
                checks["redis"] = f"error: {exc}"
                status_code = 503
        else:
            checks["redis"] = "not_configured (using in-process cache)"

        return {"status": "ok" if status_code == 200 else "unavailable", "checks": checks}, status_code
