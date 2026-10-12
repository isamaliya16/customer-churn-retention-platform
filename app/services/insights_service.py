"""Decision-support analytics: campaign planner (ROI / gain curve), segment explorer, retention outcomes.

Everything runs on the LATEST prediction of each ACTIVE customer and is computed in SQL or one numpy pass,
so it stays fast at 100k+ customers (results are cached at the route layer).
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import case, func, select

from app.extensions import db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.retention_action import RetentionAction

EXPLORER_DIMENSIONS = {
    "contract_type": Customer.contract_type,
    "plan": Customer.plan,
    "internet_service": Customer.internet_service,
    "payment_method": Customer.payment_method,
    "gender": Customer.gender,
    "senior_citizen": Customer.senior_citizen,
    "partner": Customer.partner,
    "dependents": Customer.dependents,
    "paperless_billing": Customer.paperless_billing,
    "tech_support": Customer.tech_support,
    "online_security": Customer.online_security,
    "streaming_tv": Customer.streaming_tv,
    "tenure_bucket": case(
        (Customer.tenure_months < 12, "0-12 months"),
        (Customer.tenure_months < 24, "12-24 months"),
        (Customer.tenure_months < 48, "24-48 months"),
        else_="48+ months",
    ),
    "charge_band": case(
        (Customer.monthly_charges < 35, "Under $35"),
        (Customer.monthly_charges < 65, "$35-65"),
        (Customer.monthly_charges < 90, "$65-90"),
        else_="$90+",
    ),
}
GAIN_POINTS = 20


def _latest():
    return (
        select(Prediction.customer_id.label("cid"), func.max(Prediction.id).label("pid"))
        .group_by(Prediction.customer_id)
        .subquery()
    )


class InsightsService:
    def __init__(self, session=None):
        self.session = session or db.session

    # ------------------------------------------------------------------ campaign planner
    def campaign(
        self,
        offer_cost: float = 20.0,
        success_rate: float = 0.30,
        horizon_months: int = 12,
        budget: float | None = None,
        min_probability: float = 0.0,
        preview: int = 25,
    ) -> dict:
        """Who to contact, given what an offer costs and how often it works.

        expected value of contacting a customer = p(churn) x success_rate x monthly_charges x horizon - offer_cost
        We contact customers in descending expected value while it is positive and the budget allows."""
        latest = _latest()
        rows = self.session.execute(
            select(
                Customer.id,
                Customer.name,
                Customer.contract_type,
                Customer.monthly_charges,
                Prediction.churn_probability,
                Prediction.risk_level,
                Prediction.recommended_action,
            )
            .select_from(latest)
            .join(Prediction, Prediction.id == latest.c.pid)
            .join(Customer, Customer.id == latest.c.cid)
            .where(Customer.is_active.is_(True))
        ).all()
        scored = len(rows)
        if not scored:
            return self._empty_campaign(offer_cost, success_rate, horizon_months, budget)

        prob = np.array([r.churn_probability for r in rows], dtype=float)
        monthly = np.array([r.monthly_charges for r in rows], dtype=float)
        value = prob * success_rate * monthly * horizon_months - offer_cost
        eligible = (value > 0) & (prob >= min_probability)
        order = np.argsort(-value, kind="stable")
        order = order[eligible[order]]
        if budget is not None:
            order = order[: int(budget // offer_cost) if offer_cost > 0 else len(order)]

        n = len(order)
        cost = float(n * offer_cost)
        expected_churners = float(prob[order].sum())
        saves = expected_churners * success_rate
        revenue_saved = float((prob[order] * success_rate * monthly[order] * horizon_months).sum())
        net = revenue_saved - cost

        # Gain curve: if we contact customers in descending risk, what share of all expected churn do we reach?
        by_risk = np.argsort(-prob, kind="stable")
        cum = np.cumsum(prob[by_risk])
        total_expected = float(cum[-1]) or 1.0
        gain = []
        for i in range(1, GAIN_POINTS + 1):
            k = max(int(round(scored * i / GAIN_POINTS)), 1)
            gain.append(
                {
                    "contacted_share": round(i / GAIN_POINTS, 3),
                    "churn_captured_share": round(float(cum[k - 1]) / total_expected, 4),
                    "lift": round((float(cum[k - 1]) / total_expected) / (i / GAIN_POINTS), 2),
                }
            )

        top = [
            {
                "customer_id": int(rows[i].id),
                "name": rows[i].name,
                "contract_type": rows[i].contract_type,
                "monthly_charges": float(rows[i].monthly_charges),
                "churn_probability": float(rows[i].churn_probability),
                "risk_level": rows[i].risk_level,
                "expected_value": round(float(value[i]), 2),
                "recommended_action": rows[i].recommended_action,
            }
            for i in order[:preview]
        ]
        return {
            "assumptions": {
                "offer_cost": offer_cost,
                "success_rate": success_rate,
                "horizon_months": horizon_months,
                "budget": budget,
                "min_probability": min_probability,
            },
            "scored_active_customers": scored,
            "targeted_customers": int(n),
            "targeted_share": round(n / scored, 4),
            "total_cost": round(cost, 2),
            "expected_churners_in_target": round(expected_churners, 1),
            "expected_customers_saved": round(saves, 1),
            "expected_revenue_saved": round(revenue_saved, 2),
            "net_value": round(net, 2),
            "roi": round(net / cost, 2) if cost else 0.0,
            "avg_probability_targeted": round(float(prob[order].mean()), 4) if n else 0.0,
            "customer_ids": [int(rows[i].id) for i in order[:5000]],
            "top_customers": top,
            "gain_curve": gain,
        }

    @staticmethod
    def _empty_campaign(offer_cost, success_rate, horizon_months, budget) -> dict:
        return {
            "assumptions": {
                "offer_cost": offer_cost,
                "success_rate": success_rate,
                "horizon_months": horizon_months,
                "budget": budget,
                "min_probability": 0.0,
            },
            "scored_active_customers": 0,
            "targeted_customers": 0,
            "targeted_share": 0.0,
            "total_cost": 0.0,
            "expected_churners_in_target": 0.0,
            "expected_customers_saved": 0.0,
            "expected_revenue_saved": 0.0,
            "net_value": 0.0,
            "roi": 0.0,
            "avg_probability_targeted": 0.0,
            "customer_ids": [],
            "top_customers": [],
            "gain_curve": [],
        }

    # ------------------------------------------------------------------ segment explorer
    def explorer(self, dimension: str) -> dict:
        column = EXPLORER_DIMENSIONS[dimension]
        latest = _latest()
        active = Customer.is_active.is_(True)
        high = Prediction.risk_level == "High"
        segment = func.coalesce(column.cast(db.String) if dimension == "senior_citizen" else column, "Unknown")
        rows = self.session.execute(
            select(
                segment.label("segment"),
                func.count(Customer.id),
                func.sum(case((active, 1), else_=0)),
                func.sum(case((active, 0), else_=1)),
                func.avg(case((active, Prediction.churn_probability))),
                func.sum(case((active & high, 1), else_=0)),
                func.sum(case((active, Customer.monthly_charges), else_=0.0)),
                func.sum(case((active & high, Customer.monthly_charges), else_=0.0)),
            )
            .select_from(Customer)
            .outerjoin(latest, latest.c.cid == Customer.id)
            .outerjoin(Prediction, Prediction.id == latest.c.pid)
            .group_by(segment)
        ).all()
        out = []
        for seg, total, act, churned, avg_p, high_n, mrr, rar in rows:
            if seg == "1" and dimension == "senior_citizen":
                seg = "Senior"
            elif seg == "0" and dimension == "senior_citizen":
                seg = "Non-senior"
            out.append(
                {
                    "segment": str(seg),
                    "customers": int(total),
                    "active": int(act or 0),
                    "churned": int(churned or 0),
                    "lifetime_churn_rate": round((churned or 0) / total, 4) if total else 0.0,
                    "avg_churn_probability": round(float(avg_p), 4) if avg_p is not None else None,
                    "high_risk": int(high_n or 0),
                    "high_risk_share": round((high_n or 0) / act, 4) if act else 0.0,
                    "monthly_revenue": round(float(mrr or 0), 2),
                    "revenue_at_risk": round(float(rar or 0), 2),
                }
            )
        out.sort(key=lambda r: r["revenue_at_risk"], reverse=True)
        return {"dimension": dimension, "segments": out}

    # ------------------------------------------------------------------ retention outcomes
    def outcomes(self) -> dict:
        retained = RetentionAction.outcome == "retained"
        lost = RetentionAction.outcome == "lost"
        base = (
            func.count(RetentionAction.id),
            func.sum(case((RetentionAction.status.in_(("done", "dismissed")), 1), else_=0)),
            func.sum(case((retained, 1), else_=0)),
            func.sum(case((lost, 1), else_=0)),
            func.sum(case((retained, Customer.monthly_charges), else_=0.0)),
            func.coalesce(func.sum(RetentionAction.offer_cost), 0.0),
        )
        total, closed, won, lost_n, mrr_saved, cost = self.session.execute(
            select(*base).select_from(RetentionAction).join(Customer, Customer.id == RetentionAction.customer_id)
        ).one()
        by_action = self.session.execute(
            select(
                RetentionAction.action,
                func.count(),
                func.sum(case((retained, 1), else_=0)),
                func.sum(case((lost, 1), else_=0)),
            )
            .group_by(RetentionAction.action)
            .order_by(func.count().desc())
            .limit(8)
        ).all()
        decided = int(won or 0) + int(lost_n or 0)
        return {
            "total_actions": int(total or 0),
            "closed_actions": int(closed or 0),
            "retained": int(won or 0),
            "lost": int(lost_n or 0),
            "success_rate": round(int(won or 0) / decided, 4) if decided else None,
            "monthly_revenue_retained": round(float(mrr_saved or 0), 2),
            "total_offer_cost": round(float(cost or 0), 2),
            "by_action": [
                {
                    "action": a,
                    "count": int(n),
                    "retained": int(w or 0),
                    "lost": int(lo or 0),
                    "success_rate": (
                        round(int(w or 0) / (int(w or 0) + int(lo or 0)), 4) if (w or 0) + (lo or 0) else None
                    ),
                }
                for a, n, w, lo in by_action
            ],
        }
