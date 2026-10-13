"""Generic repository base class.

Repositories are the ONLY layer that builds ORM queries for writes/reads used by services; services
call repositories, routes call services. This keeps business logic out of both route handlers and
raw DB access code.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.extensions import db

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, session: Session | None = None):
        self.session = session or db.session

    def get_by_id(self, id_: int) -> ModelT | None:
        return self.session.get(self.model, id_)

    def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        self.session.commit()
        return instance

    def commit(self) -> None:
        self.session.commit()

    def delete(self, instance: ModelT) -> None:
        self.session.delete(instance)
        self.session.commit()

    def count(self, stmt) -> int:
        return self.session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0

    def paginate(self, stmt, page: int, per_page: int):
        """Returns (items, total_count) for a `select(Model)` statement."""
        total = self.count(stmt)
        items = self.session.scalars(stmt.offset((page - 1) * per_page).limit(per_page)).all()
        return items, total
