"""APILog model - one row per HTTP request, written by the after_request hook
in app/__init__.py. This is the audit trail, distinct from the
structured stdout ops logging in app/utils/logging.py."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db
from app.utils.time import utcnow


class APILog(db.Model):
    __tablename__ = "api_logs"
    __table_args__ = (
        Index("ix_api_logs_user_id", "user_id"),
        Index("ix_api_logs_endpoint", "endpoint"),
        Index("ix_api_logs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(nullable=False)
    latency_ms: Mapped[int] = mapped_column(nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<APILog {self.method} {self.endpoint} {self.status_code}>"
