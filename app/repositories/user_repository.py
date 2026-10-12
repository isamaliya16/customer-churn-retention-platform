"""Data access for User."""

from __future__ import annotations

from sqlalchemy import func, select

from app.models.user import User
from app.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        return self.session.scalar(select(User).where(User.email == email.lower().strip()))

    def list_users(self, page: int, per_page: int):
        return self.paginate(select(User).order_by(User.id), page, per_page)

    def count_active_admins(self) -> int:
        return (
            self.session.scalar(
                select(func.count()).select_from(User).where(User.role == "admin", User.is_active.is_(True))
            )
            or 0
        )
