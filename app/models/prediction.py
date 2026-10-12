"""Prediction model - stores each churn prediction made by the model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db
from app.utils.time import utcnow

if TYPE_CHECKING:
    from app.models.customer import Customer


class Prediction(db.Model):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("churn_probability >= 0 AND churn_probability <= 1", name="ck_prediction_probability_range"),
        Index("ix_predictions_customer_id", "customer_id"),
        Index("ix_predictions_customer_id_id", "customer_id", "id"),
        Index("ix_predictions_risk_level", "risk_level"),
        Index("ix_predictions_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    churn_probability: Mapped[float] = mapped_column(nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)  # Low/Medium/High
    key_factors: Mapped[list] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False, default=list)
    recommended_action: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="predictions")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Prediction id={self.id} customer_id={self.customer_id} risk={self.risk_level}>"
