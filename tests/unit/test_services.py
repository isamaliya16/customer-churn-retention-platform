"""Service-layer unit tests (real DB, called directly rather than through HTTP)."""

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from app.extensions import db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.services.action_service import ActionService
from app.services.admin_service import AdminService
from app.services.auth_service import AuthService
from app.services.customer_service import CustomerService
from app.services.prediction_service import PredictionService
from app.utils.errors import ConflictAPIError, NotFoundAPIError, UnauthorizedAPIError, ValidationAPIError
from tests.conftest import FAKE_RESULT

BASE = dict(name="Svc Test", email="svc@example.com", monthly_charges=50.0, contract_type="one-year")


# ---------------------------------------------------------------- auth
def test_auth_register_login_and_duplicate(app):
    with app.app_context():
        service = AuthService()
        user = service.register("svc-test@example.com", "password123", "analyst")
        assert user.id and user.check_password("password123")
        with pytest.raises(ConflictAPIError):
            service.register("svc-test@example.com", "password123", "analyst")
        assert {"access_token", "refresh_token"} <= set(service.login("svc-test@example.com", "password123"))


def test_auth_login_rejects_bad_password_unknown_and_inactive(app):
    with app.app_context():
        service = AuthService()
        user = service.register("a@example.com", "correctpass1")
        for email, pw in [("a@example.com", "wrong"), ("nobody@example.com", "x")]:
            with pytest.raises(UnauthorizedAPIError):
                service.login(email, pw)
        user.is_active = False
        db.session.commit()
        with pytest.raises(UnauthorizedAPIError):
            service.login("a@example.com", "correctpass1")
        with pytest.raises(UnauthorizedAPIError):
            service.refresh(user.id)


def test_auth_change_password_rules(app):
    with app.app_context():
        service = AuthService()
        user = service.register("cp@example.com", "oldpassword1")
        with pytest.raises(UnauthorizedAPIError):
            service.change_password(user, "wrong", "newpassword1")
        with pytest.raises(ValidationAPIError):
            service.change_password(user, "oldpassword1", "oldpassword1")
        service.change_password(user, "oldpassword1", "newpassword1")
        assert user.check_password("newpassword1")


# ---------------------------------------------------------------- customers
def test_customer_create_fills_defaults(app):
    with app.app_context():
        c = CustomerService().create({**BASE, "tenure_months": 10})
        assert c.external_id.startswith("CUST-") and c.plan == "Standard"
        assert c.total_charges == 500.0  # monthly x tenure when omitted
        assert c.signup_date <= date.today() - timedelta(days=290)


def test_customer_create_plan_defaults_to_internet_service(app):
    with app.app_context():
        assert CustomerService().create({**BASE, "internet_service": "DSL"}).plan == "DSL"


def test_customer_conflicts_and_not_found(app):
    with app.app_context():
        service = CustomerService()
        service.create({**BASE, "external_id": "E1"})
        with pytest.raises(ConflictAPIError):
            service.create({**BASE, "external_id": "E2"})  # same email
        with pytest.raises(ConflictAPIError):
            service.create({**BASE, "email": "other@example.com", "external_id": "E1"})
        with pytest.raises(NotFoundAPIError):
            service.get_or_404(999999)
        with pytest.raises(NotFoundAPIError):
            service.delete(999999)


def test_customer_update_conflict_and_deactivation(app):
    with app.app_context():
        service = CustomerService()
        a = service.create({**BASE, "external_id": "A"})
        b = service.create({**BASE, "email": "b@example.com", "external_id": "B"})
        with pytest.raises(ConflictAPIError):
            service.update(b.id, {"email": a.email})
        service.update(a.id, {"is_active": False})
        assert a.churned_at is not None
        service.update(a.id, {"is_active": True})
        assert a.churned_at is None


def test_to_features_omits_unknown_profile_fields(app):
    with app.app_context():
        c = CustomerService().create(
            {**BASE, "tenure_months": 4, "internet_service": "Fiber optic", "senior_citizen": 1}
        )
        f = CustomerService.to_features(c)
        assert f["Contract"] == "One year" and f["tenure"] == 4 and f["InternetService"] == "Fiber optic"
        assert f["SeniorCitizen"] == 1
        assert "gender" not in f and "TechSupport" not in f  # unknown -> left to the imputer


def test_list_customers_attaches_latest_risk_without_n_plus_one(app):
    with app.app_context():
        service = CustomerService()
        ids = [service.create({**BASE, "email": f"c{i}@example.com"}).id for i in range(5)]
        for cid in ids:
            db.session.add(
                Prediction(
                    customer_id=cid,
                    model_version="v",
                    churn_probability=0.2,
                    risk_level="Low",
                    key_factors=[],
                    recommended_action="x",
                )
            )
            db.session.add(
                Prediction(
                    customer_id=cid,
                    model_version="v",
                    churn_probability=0.9,
                    risk_level="High",
                    key_factors=[],
                    recommended_action="x",
                )
            )
        db.session.commit()

        statements = []
        from sqlalchemy import event

        def count(conn, cursor, statement, *a):
            statements.append(statement)

        event.listen(db.engine, "before_cursor_execute", count)
        try:
            items, total = service.list_customers(1, 10)
        finally:
            event.remove(db.engine, "before_cursor_execute", count)
        assert total == 5 and all(c.latest_risk_level == "High" for c in items)  # the NEWEST prediction wins
        assert len(statements) <= 3  # count + page + latest-risk lookup, regardless of page size


def test_transactions_roundtrip_and_validation(app):
    with app.app_context():
        service = CustomerService()
        c = service.create(BASE)
        service.add_transaction(c.id, {"amount": 12.5, "type": "charge", "status": "completed"})
        assert len(service.list_transactions(c.id)) == 1
        with pytest.raises(ValidationAPIError):
            service.add_transaction(c.id, {"amount": -1, "type": "charge", "status": "completed"})
        with pytest.raises(NotFoundAPIError):
            service.list_transactions(404404)


# ---------------------------------------------------------------- predictions
def test_predict_batch_scopes_and_counts(app):
    with app.app_context():
        customers = CustomerService()
        ids = [customers.create({**BASE, "email": f"p{i}@example.com"}).id for i in range(4)]
        customers.update(ids[0], {"is_active": False})
        with patch(
            "app.services.prediction_service.ml_predict_frame",
            side_effect=lambda frame, **kw: [dict(FAKE_RESULT) for _ in range(len(frame))],
        ):
            service = PredictionService()
            assert service.predict_batch(scope="active")["scored"] == 3
            assert service.predict_batch(scope="unscored")["scored"] == 0  # churned customers are never scored
            churned_only = service.predict_batch(customer_ids=[ids[0]])
            assert (churned_only["requested"], churned_only["scored"], churned_only["skipped"]) == (1, 0, 1)
            explicit = service.predict_batch(customer_ids=[ids[1], 999999])
            assert (explicit["requested"], explicit["scored"], explicit["skipped"]) == (2, 1, 1)
            assert explicit["by_risk"]["High"] == 1 and explicit["model_version"] == "test_model_v1"


def test_predict_raw_features_is_not_persisted(app):
    with app.app_context(), patch("app.services.prediction_service.ml_predict", return_value=dict(FAKE_RESULT)):
        result = PredictionService().predict(raw_features={"tenure": 1})
        assert result["id"] is None and Prediction.query.count() == 0


def test_predict_unknown_customer_is_404(app):
    with app.app_context(), pytest.raises(NotFoundAPIError):
        PredictionService().predict(customer_id=12345)


# ---------------------------------------------------------------- actions
def test_action_lifecycle(app):
    with app.app_context():
        customer = CustomerService().create(BASE)
        service = ActionService()
        action = service.create({"customer_id": customer.id, "action": "Call them"}, None)
        assert action.status == "open" and action.customer_name == "Svc Test"
        done = service.update(action.id, {"status": "done"})
        assert done.completed_at is not None
        reopened = service.update(action.id, {"status": "in_progress"})
        assert reopened.completed_at is None
        with pytest.raises(NotFoundAPIError):
            service.create({"customer_id": 9999, "action": "x"}, None)
        other = CustomerService().create({**BASE, "email": "o@example.com"})
        pred = Prediction(
            customer_id=other.id,
            model_version="v",
            churn_probability=0.5,
            risk_level="Medium",
            key_factors=[],
            recommended_action="x",
        )
        db.session.add(pred)
        db.session.commit()
        with pytest.raises(ValidationAPIError):
            service.create({"customer_id": customer.id, "prediction_id": pred.id, "action": "x"}, None)


# ---------------------------------------------------------------- admin
def test_admin_cannot_demote_self_or_last_admin(app):
    with app.app_context():
        auth = AuthService()
        first = auth.register("first@example.com", "password123", "admin")
        second = auth.register("second@example.com", "password123", "admin")
        service = AdminService()
        with pytest.raises(ValidationAPIError):  # self
            service.update_user(first.id, {"role": "analyst"}, first)
        with pytest.raises(ValidationAPIError):
            service.update_user(first.id, {"is_active": False}, first)
        service.update_user(second.id, {"role": "analyst"}, first)  # fine: another admin remains
        with pytest.raises(ValidationAPIError):  # first is now the last admin
            service.update_user(first.id, {"is_active": False}, second)
        with pytest.raises(NotFoundAPIError):
            service.update_user(4242, {"role": "admin"}, first)


def test_admin_password_reset(app):
    with app.app_context():
        auth = AuthService()
        admin = auth.register("adm@example.com", "password123", "admin")
        user = auth.register("usr@example.com", "password123")
        AdminService().update_user(user.id, {"password": "brandnewpass1"}, admin)
        assert user.check_password("brandnewpass1")


def test_activity_stats_on_empty_log(app):
    with app.app_context():
        stats = AdminService().activity_stats(7)
        assert stats["total_requests"] == 0 and stats["error_rate"] == 0.0 and stats["p95_latency_ms"] == 0


def test_customer_model_count_helper(app):
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count()).select_from(Customer)) == 0
