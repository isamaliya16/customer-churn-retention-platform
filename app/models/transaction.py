"""Transaction model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.extensions import db
from app.utils.time import utcnow

if TYPE_CHECKING:
    from app.models.customer import Customer


class Transaction(db.Model):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_transaction_amount_nonneg"),
        Index("ix_transactions_customer_id", "customer_id"),
        Index("ix_transactions_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    amount: Mapped[float] = mapped_column(nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)  # charge/refund/credit
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="transactions")

    @validates("amount")
    def validate_amount(self, key: str, value: float) -> float:
        if value is None or value < 0:
            raise ValueError(f"amount must be non-negative, got {value!r}")
        return value

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Transaction id={self.id} customer_id={self.customer_id} amount={self.amount}>"
