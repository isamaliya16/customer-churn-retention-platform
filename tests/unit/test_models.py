"""Model-level validation and behaviour."""

from datetime import date

import pytest

from app.extensions import db
from app.models.customer import Customer
from app.models.transaction import Transaction
from app.models.user import User


def _customer(**kw):
    base = dict(
        external_id="X1",
        name="N",
        email="n@example.com",
        signup_date=date(2024, 1, 1),
        monthly_charges=10.0,
        total_charges=10.0,
        contract_type="one-year",
        plan="P",
    )
    base.update(kw)
    return Customer(**base)


@pytest.mark.parametrize("bad", ["not-an-email", "", "a@b", "@x.com"])
def test_customer_rejects_invalid_email(bad):
    with pytest.raises(ValueError):
        _customer(email=bad)


def test_customer_normalises_email_case():
    assert _customer(email="  Foo@Example.COM ").email == "foo@example.com"


def test_customer_rejects_negative_charges():
    with pytest.raises(ValueError):
        _customer(monthly_charges=-1)
    with pytest.raises(ValueError):
        _customer(total_charges=-0.01)


def test_transaction_rejects_negative_amount():
    with pytest.raises(ValueError):
        Transaction(customer_id=1, amount=-5, type="charge")


def test_churned_at_follows_is_active():
    customer = _customer()
    assert customer.churned_at is None
    customer.is_active = False
    assert customer.churned_at is not None
    customer.is_active = True
    assert customer.churned_at is None


def test_user_password_hashing_and_role_validation():
    user = User(email="u@example.com", role="analyst")
    user.set_password("s3cretpass")
    assert user.password_hash != "s3cretpass"
    assert user.check_password("s3cretpass") and not user.check_password("nope")
    with pytest.raises(ValueError):
        User(email="u2@example.com", role="superuser")
    with pytest.raises(ValueError):
        User(email="bad", role="admin")


def test_indexes_exist_on_fks_and_hot_columns(app):
    with app.app_context():
        inspector = db.inspect(db.engine)

        def indexed(table):
            return {c for idx in inspector.get_indexes(table) for c in idx["column_names"]}

        assert {"customer_id", "risk_level", "created_at"} <= indexed("predictions")
        assert {"customer_id"} <= indexed("transactions")
        assert {"contract_type", "is_active", "email", "external_id", "churned_at"} <= indexed("customers")
        assert {"customer_id", "status", "prediction_id"} <= indexed("retention_actions")
        assert {"user_id", "endpoint", "created_at"} <= indexed("api_logs")
