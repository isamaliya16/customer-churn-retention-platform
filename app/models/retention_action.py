"""RetentionAction - a follow-up task raised for an at-risk customer."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db
from app.utils.time import utcnow

if TYPE_CHECKING:
    from app.models.customer import Customer


class RetentionAction(db.Model):
    __tablename__ = "retention_actions"
    __table_args__ = (
        Index("ix_retention_actions_customer_id", "customer_id"),
        Index("ix_retention_actions_prediction_id", "prediction_id"),
        Index("ix_retention_actions_status", "status"),
        Index("ix_retention_actions_created_by", "created_by"),
        Index("ix_retention_actions_outcome", "outcome"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    due_date: Mapped[date | None] = mapped_column(nullable=True)
    offer_cost: Mapped[float | None] = mapped_column(nullable=True)  # what the incentive / call actually cost
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)  # retained | lost (closes the loop)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="actions")
