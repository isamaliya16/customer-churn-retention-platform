"""Analytics maths verified against a small, hand-computed data set."""

from datetime import timedelta

import pytest

from app.extensions import db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.retention_action import RetentionAction
from app.services.analytics_service import AnalyticsService
from app.utils.time import utcnow


def add_customer(
    i,
    *,
    contract="month-to-month",
    charges=100.0,
    tenure=12,
    active=True,
    churned_days_ago=None,
    signup_days_ago=400,
    plan="Fiber optic",
):
    now = utcnow()
    c = Customer(
        external_id=f"E{i}",
        name=f"C{i}",
        email=f"c{i}@example.com",
        signup_date=(now - timedelta(days=signup_days_ago)).date(),
        tenure_months=tenure,
        monthly_charges=charges,
        total_charges=charges * tenure,
        contract_type=contract,
        plan=plan,
    )
    db.session.add(c)
    if not active:
        c.is_active = False
        c.churned_at = now - timedelta(days=churned_days_ago if churned_days_ago is not None else 5)
    db.session.flush()
    return c


def add_prediction(customer, prob, risk, days_ago=0):
    p = Prediction(
        customer_id=customer.id,
        model_version="v",
        churn_probability=prob,
        risk_level=risk,
        key_factors=[],
        recommended_action="act",
        created_at=utcnow() - timedelta(days=days_ago),
    )
    db.session.add(p)
    db.session.flush()
    return p


@pytest.fixture
def dataset(app):
    """
    10 customers, period = last 30 days:
      * 6 active all along, 1 churned 10 days ago, 1 churned 20 days ago      -> active at start = 8, churned = 2
      * 1 churned 100 days ago (was NOT active at period start)               -> excluded from both
      * 1 signed up 5 days ago (NOT active at period start)                   -> excluded from both
    """
    with app.app_context():
        for i in range(6):
            add_customer(i, contract="two-year" if i < 2 else "month-to-month", charges=50.0 + i)
        add_customer(6, active=False, churned_days_ago=10)
        add_customer(7, active=False, churned_days_ago=20)
        add_customer(8, active=False, churned_days_ago=100)
        add_customer(9, signup_days_ago=5)
        db.session.commit()
        yield


def test_period_churn_rate_matches_definition(dataset, app):
    with app.app_context():
        o = AnalyticsService().overview(30)
        assert (o["active_at_period_start"], o["churned_in_period"]) == (8, 2)
        assert o["churn_rate"] == 0.25 and o["retention_rate"] == 0.75
        assert (o["total_customers"], o["active_customers"], o["churned_customers"]) == (10, 7, 3)
        assert o["lifetime_churn_rate"] == 0.3


def test_longer_window_includes_older_churn(dataset, app):
    with app.app_context():
        o = AnalyticsService().overview(180)
        assert o["churned_in_period"] == 3 and o["active_at_period_start"] == 9  # the day-100 churner now counts


def test_revenue_at_risk_uses_latest_prediction_of_active_customers_only(app):
    with app.app_context():
        a = add_customer(1, charges=100.0)
        b = add_customer(2, charges=40.0)
        gone = add_customer(3, charges=999.0, active=False)
        add_prediction(a, 0.9, "High", days_ago=5)
        add_prediction(a, 0.9, "High", days_ago=1)  # scored twice: counted once
        add_prediction(b, 0.8, "High", days_ago=9)
        add_prediction(b, 0.1, "Low", days_ago=1)  # latest is Low -> no longer at risk
        add_prediction(gone, 0.95, "High")  # already churned -> not "at risk"
        db.session.commit()
        o = AnalyticsService().overview(30)
        assert o["revenue_at_risk"] == 100.0 and o["high_risk_count"] == 1
        assert o["revenue_at_risk_share"] == round(100.0 / 140.0, 4)
        breakdown = AnalyticsService().revenue_at_risk()
        assert breakdown["total"] == 100.0 and breakdown["by_contract_type"][0]["segment"] == "month-to-month"


def test_risk_distribution_counts_latest_and_unscored(app):
    with app.app_context():
        a, b, c, _ = (add_customer(i) for i in range(4))
        add_prediction(a, 0.1, "Low")
        add_prediction(b, 0.5, "Medium")
        add_prediction(b, 0.9, "High")
        add_prediction(c, 0.9, "High")
        db.session.commit()
        r = AnalyticsService().risk_distribution()
        assert r["counts"] == {"Low": 1, "Medium": 0, "High": 2} and r["unscored"] == 1 and r["total_customers"] == 4


def test_segments_split_by_contract_and_tenure(dataset, app):
    with app.app_context():
        seg = AnalyticsService().segments(30)
        by = {r["segment"]: r for r in seg["by_contract_type"]}
        assert by["two-year"]["total_customers"] == 2 and by["two-year"]["churned_in_period"] == 0
        assert by["month-to-month"]["churned_in_period"] == 2
        tenure = {r["segment"]: r for r in seg["by_tenure_bucket"]}
        assert tenure["12-24 months"]["total_customers"] == 10  # all fixtures have tenure 12
        assert sum(r["total_customers"] for r in seg["by_tenure_bucket"]) == 10


def test_trends_weeks_are_continuous_and_churn_bucketed(dataset, app):
    with app.app_context():
        weekly = AnalyticsService().trends(30)["weekly"]
        starts = [w["week"] for w in weekly]
        assert starts == sorted(starts) and len(set(starts)) == len(starts)
        assert sum(w["churned"] for w in weekly) >= 2  # both recent churners fall inside some week
        assert all(0 <= w["churn_rate"] <= 1 for w in weekly)


def test_trends_prediction_volume(app):
    with app.app_context():
        c = add_customer(1)
        for d in (0, 1, 2):
            add_prediction(c, 0.9, "High", days_ago=d)
        db.session.commit()
        total = sum(w["prediction_volume"] for w in AnalyticsService().trends(30)["weekly"])
        assert total == 3
        assert AnalyticsService().overview(30)["prediction_volume"] == 3


def test_high_risk_list_sorted_paginated_and_filtered(app):
    with app.app_context():
        specs = [(1, 0.72, "month-to-month"), (2, 0.95, "month-to-month"), (3, 0.85, "one-year")]
        for i, prob, contract in specs:
            add_prediction(add_customer(i, contract=contract), prob, "High")
        add_prediction(add_customer(4), 0.4, "Medium")
        db.session.commit()
        svc = AnalyticsService()
        items, total = svc.high_risk_customers(1, 10)
        assert total == 3 and [i["churn_probability"] for i in items] == [0.95, 0.85, 0.72]
        asc, _ = svc.high_risk_customers(1, 10, sort_desc=False)
        assert asc[0]["churn_probability"] == 0.72
        page2, _ = svc.high_risk_customers(2, 2)
        assert len(page2) == 1
        only, n = svc.high_risk_customers(1, 10, contract_type="one-year")
        assert n == 1 and only[0]["contract_type"] == "one-year"


def test_clv_and_open_actions(app):
    with app.app_context():
        c = add_customer(1, charges=100.0)
        for status in ("open", "in_progress", "done", "dismissed"):
            db.session.add(RetentionAction(customer_id=c.id, action="x", status=status))
        db.session.commit()
        o = AnalyticsService().overview(30)
        assert o["open_actions"] == 2 and o["completed_actions"] == 1
        assert o["estimated_clv"] == 6000.0  # no churn -> capped 60-month lifetime x 100
