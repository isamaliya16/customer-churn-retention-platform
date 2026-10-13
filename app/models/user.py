"""User model - required by auth/RBAC."""

from __future__ import annotations

import re
from datetime import datetime

import bcrypt
from sqlalchemy import CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.extensions import db
from app.utils.time import utcnow

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
VALID_ROLES = ("admin", "analyst")


class User(db.Model):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin', 'analyst')", name="ck_user_role_valid"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="analyst")
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, nullable=False)

    @validates("email")
    def validate_email(self, key: str, value: str) -> str:
        cleaned = (value or "").strip().lower()
        if not cleaned or not EMAIL_RE.match(cleaned):
            raise ValueError(f"Invalid email address: {value!r}")
        return cleaned

    @validates("role")
    def validate_role(self, key: str, value: str) -> str:
        if value not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {value!r}")
        return value

    def set_password(self, raw_password: str) -> None:
        self.password_hash = bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def check_password(self, raw_password: str) -> bool:
        return bcrypt.checkpw(raw_password.encode("utf-8"), self.password_hash.encode("utf-8"))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} email={self.email} role={self.role}>"
