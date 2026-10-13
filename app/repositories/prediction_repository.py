"""Data access for Prediction."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from sqlalchemy import func, insert, select

from app.models.customer import Customer
from app.models.prediction import Prediction
from app.repositories.base_repository import BaseRepository


def latest_prediction_ids(customer_ids: list[int] | None = None):
    """Scalar subquery: newest prediction id per customer (portable: no DISTINCT ON)."""
    stmt = select(func.max(Prediction.id)).group_by(Prediction.customer_id)
    if customer_ids is not None:
        stmt = stmt.where(Prediction.customer_id.in_(customer_ids))
    return stmt.scalar_subquery()


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    def add_many(self, predictions: list[Prediction]) -> None:
        self.session.add_all(predictions)
        self.session.commit()

    def bulk_insert(self, rows: list[dict]) -> None:
        """Fast Core-style insert of plain dicts (no ORM objects) - used for 100k-scale scoring."""
        if rows:
            self.session.execute(insert(Prediction), rows)
            self.session.commit()

    def list_filtered(
        self,
        page: int,
        per_page: int,
        customer_id: int | None = None,
        risk_level: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ):
        """Returns (predictions, total); each prediction gets a transient `customer_name`."""
        stmt = select(Prediction, Customer.name).join(Customer, Customer.id == Prediction.customer_id)
        if customer_id:
            stmt = stmt.where(Prediction.customer_id == customer_id)
        if risk_level:
            stmt = stmt.where(Prediction.risk_level == risk_level)
        if date_from:
            stmt = stmt.where(Prediction.created_at >= datetime.combine(date_from, time.min))
        if date_to:  # inclusive of the whole end day
            stmt = stmt.where(Prediction.created_at < datetime.combine(date_to + timedelta(days=1), time.min))
        total = self.count(stmt)
        rows = self.session.execute(
            stmt.order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        ).all()
        items = []
        for prediction, name in rows:
            prediction.customer_name = name
            items.append(prediction)
        return items, total

    def get_with_customer_name(self, prediction_id: int) -> Prediction | None:
        row = self.session.execute(
            select(Prediction, Customer.name)
            .join(Customer, Customer.id == Prediction.customer_id)
            .where(Prediction.id == prediction_id)
        ).first()
        if not row:
            return None
        row[0].customer_name = row[1]
        return row[0]

    def latest_summary_for(self, customer_ids: list[int]) -> dict[int, tuple[str, float]]:
        """{customer_id: (risk_level, probability)} for the latest prediction of each given customer."""
        if not customer_ids:
            return {}
        rows = self.session.execute(
            select(Prediction.customer_id, Prediction.risk_level, Prediction.churn_probability).where(
                Prediction.id.in_(latest_prediction_ids(customer_ids))
            )
        ).all()
        return {cid: (risk, prob) for cid, risk, prob in rows}
