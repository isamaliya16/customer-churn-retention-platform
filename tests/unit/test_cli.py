"""Flask CLI commands."""

from unittest.mock import patch

from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.retention_action import RetentionAction
from app.models.transaction import Transaction
from app.models.user import User
from tests.conftest import FAKE_RESULT


def test_init_db_runs(app):
    result = app.test_cli_runner().invoke(args=["init-db"])
    assert result.exit_code == 0 and "ready" in result.output


def test_create_admin_creates_then_resets(app):
    runner = app.test_cli_runner()
    first = runner.invoke(args=["create-admin", "--email", "cli-admin@example.com", "--password", "Passw0rd!x"])
    assert first.exit_code == 0 and "Created admin" in first.output
    second = runner.invoke(args=["create-admin", "--email", "cli-admin@example.com", "--password", "NewPassw0rd!"])
    assert second.exit_code == 0 and "Updated" in second.output
    with app.app_context():
        user = User.query.filter_by(email="cli-admin@example.com").one()
        assert user.role == "admin" and user.check_password("NewPassw0rd!")


def test_seed_demo_creates_full_demo_data_and_is_idempotent(app):
    with patch("app.ml.predict.predict", return_value={**FAKE_RESULT, "risk_level": "High"}):
        result = app.test_cli_runner().invoke(args=["seed-demo", "--count", "40"])
        assert result.exit_code == 0, result.output
        again = app.test_cli_runner().invoke(args=["seed-demo", "--count", "40"])
        assert again.exit_code == 0
    with app.app_context():
        active = Customer.query.filter_by(is_active=True).count()
        assert Customer.query.count() == 40 and Prediction.query.count() == active  # churned customers not scored
        assert Transaction.query.count() >= 40
        churned = Customer.query.filter_by(is_active=False).all()
        assert churned and all(c.churned_at is not None for c in churned)  # churn dates recorded
        assert Customer.query.filter(Customer.internet_service.isnot(None)).count() == 40  # profile stored
        assert RetentionAction.query.count() >= 1


def test_seed_demo_without_model_still_seeds_customers(app):
    from app.ml.predict import ModelNotTrainedError

    with patch("app.ml.predict.predict", side_effect=ModelNotTrainedError("none")):
        result = app.test_cli_runner().invoke(args=["seed-demo", "--count", "5"])
    assert result.exit_code == 0 and "without predictions" in result.output
    with app.app_context():
        assert Customer.query.count() == 5 and Prediction.query.count() == 0


def test_train_command(app):
    with patch("app.ml.train.train_and_select_best", return_value={"model_version": "x_v1"}):
        result = app.test_cli_runner().invoke(args=["train"])
    assert result.exit_code == 0 and "x_v1" in result.output


def test_import_csv_command_loads_and_scores(app, tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(
        "external_id,name,email,tenure_months,monthly_charges,contract_type\n"
        "X1,Ann,ann@example.com,4,60,month-to-month\n"
        "X2,Bob,bob@example.com,30,45,two-year\n"
    )
    with patch(
        "app.services.prediction_service.ml_predict_frame",
        side_effect=lambda frame, **kw: [dict(FAKE_RESULT) for _ in range(len(frame))],
    ):
        result = app.test_cli_runner().invoke(args=["import-csv", str(path)])
    assert result.exit_code == 0, result.output
    assert "completed" in result.output and "scored 2" in result.output
    with app.app_context():
        assert Customer.query.count() == 2 and Prediction.query.count() == 2


def test_seed_actions_command(app, tmp_path):
    with app.app_context():
        from app.extensions import db

        c = Customer(
            external_id="S1", name="S", email="s@example.com", signup_date=__import__("datetime").date(2024, 1, 1),
            monthly_charges=50, contract_type="month-to-month", plan="DSL",
        )  # fmt: skip
        db.session.add(c)
        db.session.flush()
        db.session.add(
            Prediction(
                customer_id=c.id, model_version="t", churn_probability=0.9, risk_level="High",
                key_factors=[], recommended_action="Call",
            )
        )  # fmt: skip
        db.session.commit()
    result = app.test_cli_runner().invoke(args=["seed-actions", "--count", "5"])
    assert result.exit_code == 0 and "Created 1 demo actions" in result.output
    with app.app_context():
        assert RetentionAction.query.count() == 1
