"""Data access for APILog (audit trail)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import case, func, select

from app.models.api_log import APILog
from app.models.user import User
from app.repositories.base_repository import BaseRepository


class APILogRepository(BaseRepository[APILog]):
    model = APILog

    def list_filtered(
        self,
        page: int,
        per_page: int,
        method: str | None = None,
        status_class: int | None = None,
        endpoint: str | None = None,
        user_id: int | None = None,
    ):
        stmt = select(APILog, User.email).outerjoin(User, User.id == APILog.user_id)
        if method:
            stmt = stmt.where(APILog.method == method)
        if status_class:
            stmt = stmt.where(APILog.status_code >= status_class * 100, APILog.status_code < (status_class + 1) * 100)
        if endpoint:
            stmt = stmt.where(APILog.endpoint.icontains(endpoint, autoescape=True))
        if user_id:
            stmt = stmt.where(APILog.user_id == user_id)
        total = self.count(stmt)
        rows = self.session.execute(
            stmt.order_by(APILog.created_at.desc(), APILog.id.desc()).offset((page - 1) * per_page).limit(per_page)
        ).all()
        items = []
        for log, email in rows:
            log.user_email = email
            items.append(log)
        return items, total

    def stats_since(self, since: datetime) -> dict:
        errors = func.sum(case((APILog.status_code >= 400, 1), else_=0))
        total, avg_latency, error_count = self.session.execute(
            select(func.count(), func.avg(APILog.latency_ms), errors).where(APILog.created_at >= since)
        ).one()
        latencies = [
            row[0]
            for row in self.session.execute(
                select(APILog.latency_ms)
                .where(APILog.created_at >= since)
                .order_by(APILog.created_at.desc())
                .limit(20000)
            )
        ]
        top = self.session.execute(
            select(APILog.endpoint, func.count().label("n"), func.avg(APILog.latency_ms))
            .where(APILog.created_at >= since)
            .group_by(APILog.endpoint)
            .order_by(func.count().desc())
            .limit(8)
        ).all()
        statuses = self.session.execute(
            select(APILog.status_code, func.count()).where(APILog.created_at >= since).group_by(APILog.status_code)
        ).all()
        created = [row[0] for row in self.session.execute(select(APILog.created_at).where(APILog.created_at >= since))]
        return {
            "total": int(total or 0),
            "avg_latency": float(avg_latency or 0.0),
            "error_count": int(error_count or 0),
            "latencies": latencies,
            "top": top,
            "statuses": statuses,
            "created": created,
        }
