"""Admin business logic: user management and API-activity statistics."""

from __future__ import annotations

from datetime import timedelta

from app.models.user import User
from app.repositories.api_log_repository import APILogRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.utils.errors import NotFoundAPIError, ValidationAPIError
from app.utils.time import utcnow


class AdminService:
    def __init__(self):
        self.users = UserRepository()
        self.logs = APILogRepository(self.users.session)

    # --- users -----------------------------------------------------------------------
    def list_users(self, page: int, per_page: int):
        return self.users.list_users(page, per_page)

    def create_user(self, email: str, password: str, role: str) -> User:
        return AuthService(self.users).register(email, password, role)

    def update_user(self, user_id: int, data: dict, acting_user: User) -> User:
        user = self.users.get_by_id(user_id)
        if not user:
            raise NotFoundAPIError(f"User {user_id} not found.")

        demoting = "role" in data and data["role"] != "admin" and user.role == "admin"
        deactivating = data.get("is_active") is False and user.is_active
        if user.id == acting_user.id and (demoting or deactivating):
            raise ValidationAPIError("You cannot demote or deactivate your own account.")
        if (
            user.role == "admin"
            and user.is_active
            and (demoting or deactivating)
            and self.users.count_active_admins() <= 1
        ):
            raise ValidationAPIError("At least one active admin must remain.")

        if "role" in data:
            user.role = data["role"]
        if "is_active" in data:
            user.is_active = data["is_active"]
        if data.get("password"):
            user.set_password(data["password"])
        self.users.commit()
        return user

    # --- activity ---------------------------------------------------------------------
    def list_logs(self, page: int, per_page: int, **filters):
        return self.logs.list_filtered(page, per_page, **filters)

    def activity_stats(self, days: int) -> dict:
        since = utcnow() - timedelta(days=days)
        raw = self.logs.stats_since(since)
        latencies = sorted(raw["latencies"])
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0
        per_day: dict[str, int] = {}
        for created in raw["created"]:
            key = created.date().isoformat()
            per_day[key] = per_day.get(key, 0) + 1
        total = raw["total"]
        return {
            "days": days,
            "total_requests": total,
            "error_rate": round(raw["error_count"] / total, 4) if total else 0.0,
            "avg_latency_ms": round(raw["avg_latency"], 1),
            "p95_latency_ms": p95,
            "requests_per_day": [{"date": d, "count": c} for d, c in sorted(per_day.items())],
            "top_endpoints": [
                {"endpoint": e, "count": int(n), "avg_latency_ms": round(float(a or 0), 1)} for e, n, a in raw["top"]
            ],
            "status_codes": [{"status": int(s), "count": int(c)} for s, c in sorted(raw["statuses"])],
        }
