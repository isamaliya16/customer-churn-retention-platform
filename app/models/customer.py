"""Customer model."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.extensions import db
from app.utils.time import utcnow

if TYPE_CHECKING:
    from app.models.prediction import Prediction
    from app.models.retention_action import RetentionAction
    from app.models.transaction import Transaction

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Customer(db.Model):
    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint("monthly_charges >= 0", name="ck_customer_monthly_charges_nonneg"),
        CheckConstraint("total_charges >= 0", name="ck_customer_total_charges_nonneg"),
        CheckConstraint("tenure_months >= 0", name="ck_customer_tenure_nonneg"),
        Index("ix_customers_contract_type", "contract_type"),
        Index("ix_customers_is_active", "is_active"),
        Index("ix_customers_plan", "plan"),
        Index("ix_customers_churned_at", "churned_at"),
        Index("ix_customers_signup_date", "signup_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    signup_date: Mapped[date] = mapped_column(nullable=False)
    tenure_months: Mapped[int] = mapped_column(nullable=False, default=0)
    monthly_charges: Mapped[float] = mapped_column(nullable=False, default=0.0)
    total_charges: Mapped[float] = mapped_column(nullable=False, default=0.0)
    contract_type: Mapped[str] = mapped_column(String(32), nullable=False)  # month-to-month/one-year/two-year
    plan: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    churned_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Optional service profile (IBM Telco vocabulary). NULL = unknown -> imputed by the model pipeline.
    gender: Mapped[str | None] = mapped_column(String(16))
    senior_citizen: Mapped[int | None] = mapped_column()
    partner: Mapped[str | None] = mapped_column(String(8))
    dependents: Mapped[str | None] = mapped_column(String(8))
    phone_service: Mapped[str | None] = mapped_column(String(8))
    multiple_lines: Mapped[str | None] = mapped_column(String(32))
    internet_service: Mapped[str | None] = mapped_column(String(32))
    online_security: Mapped[str | None] = mapped_column(String(32))
    online_backup: Mapped[str | None] = mapped_column(String(32))
    device_protection: Mapped[str | None] = mapped_column(String(32))
    tech_support: Mapped[str | None] = mapped_column(String(32))
    streaming_tv: Mapped[str | None] = mapped_column(String(32))
    streaming_movies: Mapped[str | None] = mapped_column(String(32))
    paperless_billing: Mapped[str | None] = mapped_column(String(8))
    payment_method: Mapped[str | None] = mapped_column(String(40))

    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow, nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="customer", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="customer", cascade="all, delete-orphan")
    actions: Mapped[list["RetentionAction"]] = relationship(back_populates="customer", cascade="all, delete-orphan")

    @validates("email")
    def validate_email(self, key: str, value: str) -> str:
        cleaned = (value or "").strip().lower()
        if not cleaned or not EMAIL_RE.match(cleaned):
            raise ValueError(f"Invalid email address: {value!r}")
        return cleaned

    @validates("monthly_charges", "total_charges")
    def validate_nonneg_charges(self, key: str, value: float) -> float:
        if value is None or value < 0:
            raise ValueError(f"{key} must be non-negative, got {value!r}")
        return value

    @validates("is_active")
    def track_churn_date(self, key: str, value: bool) -> bool:
        """Keep `churned_at` consistent with `is_active` automatically."""
        if value is False and self.churned_at is None:
            self.churned_at = utcnow()
        elif value is True and self.churned_at is not None:
            self.churned_at = None
        return value

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Customer id={self.id} email={self.email}>"
