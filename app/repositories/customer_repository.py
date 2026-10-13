"""Data access for Customer."""

from __future__ import annotations

from sqlalchemy import exists, func, or_, select

from app.constants import CONTRACT_TO_RAW, PROFILE_TO_FEATURE
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.transaction import Transaction
from app.repositories.base_repository import BaseRepository
from app.repositories.prediction_repository import latest_prediction_ids

SORTABLE = {
    "id": Customer.id,
    "name": Customer.name,
    "email": Customer.email,
    "tenure_months": Customer.tenure_months,
    "monthly_charges": Customer.monthly_charges,
    "total_charges": Customer.total_charges,
    "created_at": Customer.created_at,
    "signup_date": Customer.signup_date,
}


class CustomerRepository(BaseRepository[Customer]):
    model = Customer

    def get_by_email(self, email: str) -> Customer | None:
        return self.session.scalar(select(Customer).where(Customer.email == email.lower().strip()))

    def get_by_external_id(self, external_id: str) -> Customer | None:
        return self.session.scalar(select(Customer).where(Customer.external_id == external_id))

    def list_filtered(
        self,
        page: int,
        per_page: int,
        q: str | None = None,
        contract_type: str | None = None,
        is_active: bool | None = None,
        risk_level: str | None = None,
        sort_by: str = "id",
        order: str = "asc",
    ):
        stmt = select(Customer)
        if q:
            stmt = stmt.where(
                or_(
                    Customer.name.icontains(q, autoescape=True),
                    Customer.email.icontains(q, autoescape=True),
                    Customer.external_id.icontains(q, autoescape=True),
                )
            )
        if contract_type:
            stmt = stmt.where(Customer.contract_type == contract_type)
        if is_active is not None:
            stmt = stmt.where(Customer.is_active == is_active)
        if risk_level:
            stmt = stmt.join(Prediction, Prediction.customer_id == Customer.id).where(
                Prediction.id.in_(latest_prediction_ids()), Prediction.risk_level == risk_level
            )

        column = SORTABLE.get(sort_by, Customer.id)
        stmt = stmt.order_by(column.desc() if order == "desc" else column.asc(), Customer.id.asc())
        return self.paginate(stmt, page, per_page)

    def update(self, customer: Customer, data: dict) -> Customer:
        for key, value in data.items():
            setattr(customer, key, value)
        self.session.commit()
        return customer

    def recent_transactions(self, customer_id: int, limit: int = 50) -> list[Transaction]:
        return list(
            self.session.scalars(
                select(Transaction)
                .where(Transaction.customer_id == customer_id)
                .order_by(Transaction.created_at.desc(), Transaction.id.desc())
                .limit(limit)
            )
        )

    def iter_all(self, active_only: bool = False, only_ids: list[int] | None = None, limit: int | None = None):
        stmt = select(Customer).order_by(Customer.id)
        if active_only:
            stmt = stmt.where(Customer.is_active.is_(True))
        if only_ids is not None:
            stmt = stmt.where(Customer.id.in_(only_ids))
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def unscored(self, limit: int) -> list[Customer]:
        return list(self.session.scalars(self._unscored_stmt(select(Customer)).limit(limit)))

    @staticmethod
    def _unscored_stmt(stmt):
        has_prediction = exists().where(Prediction.customer_id == Customer.id)
        return stmt.where(~has_prediction, Customer.is_active.is_(True)).order_by(Customer.id)

    def unscored_ids(self, limit: int) -> list[int]:
        return list(self.session.scalars(self._unscored_stmt(select(Customer.id)).limit(limit)))

    def active_ids(self, only_ids: list[int] | None = None, limit: int | None = None, active_only: bool = True):
        stmt = select(Customer.id).order_by(Customer.id)
        if active_only or only_ids is not None:
            stmt = stmt.where(Customer.is_active.is_(True))
        if only_ids is not None:
            stmt = stmt.where(Customer.id.in_(only_ids))
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def feature_rows(self, ids: list[int]) -> list[tuple[int, dict]]:
        """[(customer_id, raw model features)] for the given ids, reading only the needed columns."""
        columns = [getattr(Customer, c) for c in PROFILE_TO_FEATURE]
        rows = self.session.execute(
            select(Customer.id, Customer.tenure_months, Customer.monthly_charges, Customer.contract_type, *columns)
            .where(Customer.id.in_(ids))
            .order_by(Customer.id)
        ).all()
        out = []
        for row in rows:
            features = {
                "tenure": row.tenure_months,
                "MonthlyCharges": row.monthly_charges,
                "Contract": CONTRACT_TO_RAW.get(row.contract_type, "Month-to-month"),
            }
            for column, feature in PROFILE_TO_FEATURE.items():
                value = getattr(row, column)
                if value is not None:
                    features[feature] = value
            out.append((row.id, features))
        return out

    def export_rows(self, limit: int):
        """Customers + latest risk in ONE query (no giant IN-lists), for CSV export at 100k+ scale."""
        latest = (
            select(Prediction.customer_id.label("cid"), func.max(Prediction.id).label("pid"))
            .group_by(Prediction.customer_id)
            .subquery()
        )
        return self.session.execute(
            select(Customer, Prediction.risk_level, Prediction.churn_probability)
            .outerjoin(latest, latest.c.cid == Customer.id)
            .outerjoin(Prediction, Prediction.id == latest.c.pid)
            .order_by(Customer.id)
            .limit(limit)
        ).all()
