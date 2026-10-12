"""ImportJob - one bulk CSV import (progress is stored in the DB so any worker can report it)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db
from app.utils.time import utcnow

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class ImportJob(db.Model):
    __tablename__ = "import_jobs"
    __table_args__ = (Index("ix_import_jobs_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )  # pending/running/completed/failed
    phase: Mapped[str] = mapped_column(String(24), nullable=False, default="queued")  # queued/importing/scoring/done
    mode: Mapped[str] = mapped_column(String(8), nullable=False, default="skip")  # skip | update
    score_after: Mapped[bool] = mapped_column(nullable=False, default=True)
    total_rows: Mapped[int] = mapped_column(nullable=False, default=0)
    processed_rows: Mapped[int] = mapped_column(nullable=False, default=0)
    created_count: Mapped[int] = mapped_column(nullable=False, default=0)
    updated_count: Mapped[int] = mapped_column(nullable=False, default=0)
    skipped_count: Mapped[int] = mapped_column(nullable=False, default=0)
    rejected_count: Mapped[int] = mapped_column(nullable=False, default=0)
    scored_count: Mapped[int] = mapped_column(nullable=False, default=0)
    errors: Mapped[list] = mapped_column(JSON_TYPE, nullable=False, default=list)  # first N row-level problems
    warnings: Mapped[list] = mapped_column(JSON_TYPE, nullable=False, default=list)
    message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
