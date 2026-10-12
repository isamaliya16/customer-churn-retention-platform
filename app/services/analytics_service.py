"""Business logic for the /analytics/* endpoints.

Definitions (see README):
* churn rate (period)  = customers that churned during the period / customers active at period start
* retention rate       = 1 - churn rate
* revenue at risk      = sum(monthly_charges) of ACTIVE customers whose LATEST prediction is High
* estimated CLV        = avg monthly charge x expected lifetime (1 / monthly churn rate, capped at 60 months)

Caching is applied at the route layer; everything here is portable across SQLite and PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import and_, case, func, or_, select

from app.extensions import db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.retention_action import RetentionAction
from app.repositories.prediction_repository import latest_prediction_ids
from app.utils.time import utcnow

TENURE_BUCKETS = [
    (0, 12, "0-12 months"),
    (12, 24, "12-24 months"),
    (24, 48, "24-48 months"),
    (48, 10_000, "48+ months"),
]
MAX_LIFETIME_MONTHS = 60


def _rate(numerator, denominator) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


class AnalyticsService:
    def __init__(self, session=None):
        self.session = session or db.session

    # --- shared SQL fragments --------------------------------------------------------
    @staticmethod
    def _churn_aggregates(start: datetime, end: datetime):
        """(active_at_start, churned_in_period) as SUM(CASE ...) expressions."""
        started = Customer.signup_date <= start.date()
        active_at_start = case(
            (and_(started, or_(Customer.churned_at.is_(None), Customer.churned_at >= start)), 1), else_=0
        )
        churned = case((and_(started, Customer.churned_at >= start, Customer.churned_at <= end), 1), else_=0)
        return func.coalesce(func.sum(active_at_start), 0), func.coalesce(func.sum(churned), 0)

    def _period_churn(self, start: datetime, end: datetime) -> tuple[int, int]:
        active, churned = self.session.execute(select(*self._churn_aggregates(start, end))).one()
        return int(active), int(churned)

    def _latest_high_subquery(self):
        return (
            select(Customer.id, Customer.monthly_charges, Customer.contract_type, Customer.plan)
            .join(Prediction, Prediction.customer_id == Customer.id)
            .where(
                Prediction.id.in_(latest_prediction_ids()),
                Prediction.risk_level == "High",
                Customer.is_active.is_(True),
            )
            .subquery()
        )

    # --- /analytics/overview ---------------------------------------------------------
    def overview(self, days: int = 30) -> dict:
        end = utcnow()
        start = end - timedelta(days=days)

        total = self.session.scalar(select(func.count()).select_from(Customer)) or 0
        active = (
            self.session.scalar(select(func.count()).select_from(Customer).where(Customer.is_active.is_(True))) or 0
        )
        active_at_start, churned_in_period = self._period_churn(start, end)
        churn_rate = _rate(churned_in_period, active_at_start)

        high = self._latest_high_subquery()
        revenue_at_risk = self.session.scalar(select(func.coalesce(func.sum(high.c.monthly_charges), 0.0))) or 0.0
        high_risk_count = self.session.scalar(select(func.count()).select_from(high)) or 0
        total_mrr = (
            self.session.scalar(
                select(func.coalesce(func.sum(Customer.monthly_charges), 0.0)).where(Customer.is_active.is_(True))
            )
            or 0.0
        )

        avg_monthly = float(self.session.scalar(select(func.avg(Customer.monthly_charges))) or 0.0)
        monthly_churn = 1 - (1 - churn_rate) ** (30 / days) if churn_rate < 1 else 1.0
        lifetime = min(1 / monthly_churn, MAX_LIFETIME_MONTHS) if monthly_churn > 0 else MAX_LIFETIME_MONTHS

        action_counts = {
            status: count
            for status, count in self.session.execute(
                select(RetentionAction.status, func.count()).group_by(RetentionAction.status)
            ).all()
        }
        return {
            "period_days": days,
            "total_customers": total,
            "active_customers": active,
            "churned_customers": total - active,
            "active_at_period_start": active_at_start,
            "churned_in_period": churned_in_period,
            "churn_rate": churn_rate,
            "retention_rate": round(1 - churn_rate, 4),
            "lifetime_churn_rate": _rate(total - active, total),
            "revenue_at_risk": round(float(revenue_at_risk), 2),
            "monthly_recurring_revenue": round(float(total_mrr), 2),
            "revenue_at_risk_share": _rate(revenue_at_risk, total_mrr),
            "high_risk_count": high_risk_count,
            "estimated_clv": round(avg_monthly * lifetime, 2),
            "prediction_volume": self._prediction_volume(days),
            "open_actions": action_counts.get("open", 0) + action_counts.get("in_progress", 0),
            "completed_actions": action_counts.get("done", 0),
        }

    # --- /analytics/segments ---------------------------------------------------------
    def segments(self, days: int = 90) -> dict:
        end = utcnow()
        start = end - timedelta(days=days)
        return {
            "period_days": days,
            "by_contract_type": self._segment_by(Customer.contract_type, start, end),
            "by_plan": self._segment_by(Customer.plan, start, end),
            "by_tenure_bucket": self._segment_by_tenure(start, end),
        }

    def _segment_row(self, segment, total, lifetime_churned, active_start, churned) -> dict:
        return {
            "segment": segment,
            "total_customers": int(total or 0),
            "lifetime_churned": int(lifetime_churned or 0),
            "active_at_period_start": int(active_start or 0),
            "churned_in_period": int(churned or 0),
            "churn_rate": _rate(churned, active_start),
            "lifetime_churn_rate": _rate(lifetime_churned, total),
        }

    def _segment_by(self, column, start, end) -> list[dict]:
        active_start, churned = self._churn_aggregates(start, end)
        lifetime = func.sum(case((Customer.is_active.is_(False), 1), else_=0))
        rows = self.session.execute(
            select(column, func.count(), lifetime, active_start, churned).group_by(column).order_by(func.count().desc())
        ).all()
        return [self._segment_row(*row) for row in rows]

    def _segment_by_tenure(self, start, end) -> list[dict]:
        active_start, churned = self._churn_aggregates(start, end)
        lifetime = func.sum(case((Customer.is_active.is_(False), 1), else_=0))
        result = []
        for low, high, label in TENURE_BUCKETS:
            row = self.session.execute(
                select(func.count(), lifetime, active_start, churned).where(
                    Customer.tenure_months >= low, Customer.tenure_months < high
                )
            ).one()
            result.append(self._segment_row(label, *row))
        return result

    # --- /analytics/risk-distribution + /analytics/revenue-at-risk -----------------------
    def risk_distribution(self) -> dict:
        """Latest risk level of every ACTIVE customer (churned customers are no longer actionable)."""
        latest = select(func.max(Prediction.id).label("pid")).group_by(Prediction.customer_id).subquery()
        rows = self.session.execute(
            select(Prediction.risk_level, func.count())
            .select_from(latest)
            .join(Prediction, Prediction.id == latest.c.pid)
            .join(Customer, Customer.id == Prediction.customer_id)
            .where(Customer.is_active.is_(True))
            .group_by(Prediction.risk_level)
        ).all()
        counts = {"Low": 0, "Medium": 0, "High": 0}
        counts.update({level: int(n) for level, n in rows})
        total = self.session.scalar(select(func.count()).select_from(Customer).where(Customer.is_active.is_(True))) or 0
        scored = sum(counts.values())
        return {"counts": counts, "unscored": max(total - scored, 0), "total_customers": total}

    def revenue_at_risk(self) -> dict:
        high = self._latest_high_subquery()

        def grouped(column):
            rows = self.session.execute(
                select(column, func.count(), func.sum(high.c.monthly_charges))
                .group_by(column)
                .order_by(func.sum(high.c.monthly_charges).desc())
            ).all()
            return [
                {"segment": seg, "customers": int(n), "revenue_at_risk": round(float(rev or 0), 2)}
                for seg, n, rev in rows
            ]

        total = self.session.scalar(select(func.coalesce(func.sum(high.c.monthly_charges), 0.0))) or 0.0
        return {
            "total": round(float(total), 2),
            "by_contract_type": grouped(high.c.contract_type),
            "by_plan": grouped(high.c.plan),
        }

    # --- /analytics/trends -----------------------------------------------------------
    def trends(self, days: int = 90) -> dict:
        """Weekly churn rate (churned / active at week start) + prediction volume, oldest first."""
        end = utcnow()
        first = end - timedelta(days=days)
        week_start = (first - timedelta(days=first.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

        predictions = self.session.execute(
            select(Prediction.created_at, Prediction.churn_probability, Prediction.risk_level).where(
                Prediction.created_at >= week_start
            )
        ).all()

        weeks = []
        cursor = week_start
        while cursor <= end:
            nxt = cursor + timedelta(days=7)
            active_start, churned = self._period_churn(cursor, min(nxt, end))
            in_week = [p for p in predictions if cursor <= p[0] < nxt]
            weeks.append(
                {
                    "week": cursor.date().isoformat(),
                    "active_at_week_start": active_start,
                    "churned": churned,
                    "churn_rate": _rate(churned, active_start),
                    "prediction_volume": len(in_week),
                    "avg_churn_probability": (
                        round(sum(float(p[1]) for p in in_week) / len(in_week), 4) if in_week else 0.0
                    ),
                    "high_risk_count": sum(1 for p in in_week if p[2] == "High"),
                }
            )
            cursor = nxt
        return {"period_days": days, "weekly": weeks}

    # --- /analytics/high-risk --------------------------------------------------------
    def high_risk_customers(self, page: int, per_page: int, sort_desc: bool = True, contract_type: str | None = None):
        order = Prediction.churn_probability.desc() if sort_desc else Prediction.churn_probability.asc()
        stmt = (
            select(Customer, Prediction.id, Prediction.churn_probability, Prediction.recommended_action)
            .join(Prediction, Prediction.customer_id == Customer.id)
            .where(
                Prediction.id.in_(latest_prediction_ids()),
                Prediction.risk_level == "High",
                Customer.is_active.is_(True),
            )
            .order_by(order, Customer.id)
        )
        if contract_type:
            stmt = stmt.where(Customer.contract_type == contract_type)
        total = self.session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery())) or 0
        rows = self.session.execute(stmt.offset((page - 1) * per_page).limit(per_page)).all()
        items = [
            {
                "customer_id": c.id,
                "prediction_id": pid,
                "name": c.name,
                "email": c.email,
                "contract_type": c.contract_type,
                "tenure_months": c.tenure_months,
                "monthly_charges": c.monthly_charges,
                "churn_probability": round(float(prob), 4),
                "recommended_action": action,
                "is_active": c.is_active,
            }
            for c, pid, prob, action in rows
        ]
        return items, total

    def _prediction_volume(self, days: int) -> int:
        since = utcnow() - timedelta(days=days)
        return (
            self.session.scalar(select(func.count()).select_from(Prediction).where(Prediction.created_at >= since)) or 0
        )
