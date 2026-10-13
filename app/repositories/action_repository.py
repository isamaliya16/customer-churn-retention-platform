"""Data access for RetentionAction."""

from __future__ import annotations

from sqlalchemy import select

from app.models.customer import Customer
from app.models.retention_action import RetentionAction
from app.repositories.base_repository import BaseRepository


class ActionRepository(BaseRepository[RetentionAction]):
    model = RetentionAction

    def list_filtered(self, page: int, per_page: int, status: str | None = None, customer_id: int | None = None):
        stmt = select(RetentionAction, Customer.name).join(Customer, Customer.id == RetentionAction.customer_id)
        if status:
            stmt = stmt.where(RetentionAction.status == status)
        if customer_id:
            stmt = stmt.where(RetentionAction.customer_id == customer_id)
        total = self.count(stmt)
        rows = self.session.execute(
            stmt.order_by(RetentionAction.created_at.desc(), RetentionAction.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        ).all()
        items = []
        for action, name in rows:
            action.customer_name = name
            items.append(action)
        return items, total

    def with_customer_name(self, action: RetentionAction) -> RetentionAction:
        action.customer_name = self.session.scalar(select(Customer.name).where(Customer.id == action.customer_id))
        return action
