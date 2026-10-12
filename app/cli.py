"""Flask CLI commands: init-db, create-admin, seed-demo, train.

Usage (from the project folder):
    python -m flask --app wsgi init-db
    python -m flask --app wsgi create-admin --email admin@example.com --password Admin@12345
    python -m flask --app wsgi seed-demo --count 300
    python -m flask --app wsgi train
"""

from __future__ import annotations

import os
import random
from datetime import timedelta

import click
import pandas as pd
from flask import Flask

from app.constants import PROFILE_TO_FEATURE, RAW_TO_CONTRACT
from app.extensions import db
from app.utils.time import utcnow


def register_cli(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db() -> None:
        """Create all tables (use `flask db upgrade` instead when running migrations on PostgreSQL)."""
        import app.models  # noqa: F401  (register models on the metadata)

        db.create_all()
        click.echo("Database tables ready.")

    @app.cli.command("create-admin")
    @click.option("--email", required=True)
    @click.option("--password", required=True)
    def create_admin(email: str, password: str) -> None:
        """Create (or reset) an admin user."""
        from app.models.user import User
        from app.repositories.user_repository import UserRepository

        repo = UserRepository()
        user = repo.get_by_email(email)
        if user is None:
            user = User(email=email, role="admin")
            user.set_password(password)
            repo.add(user)
            click.echo(f"Created admin {email}")
        else:
            user.role = "admin"
            user.is_active = True
            user.set_password(password)
            db.session.commit()
            click.echo(f"Updated existing user {email} -> admin with new password")

    @app.cli.command("train")
    def train() -> None:
        """Train / compare the models and save the best one (same as `python -m app.ml.train`)."""
        from app.ml.predict import reload_artifacts
        from app.ml.train import train_and_select_best

        meta = train_and_select_best()
        reload_artifacts()
        click.echo(f"Trained {meta['model_version']}")

    @app.cli.command("seed-demo")
    @click.option("--count", default=300, show_default=True, help="How many demo customers to load.")
    def seed_demo(count: int) -> None:
        """Load demo customers (profile, billing history, predictions, actions) from the Telco dataset.

        DEMO DATA ONLY: churn dates are randomly spread over the last 120 days, so period churn rates and trend
        charts are illustrative. Customers come from the same dataset the model trained on, so results on this
        data are in-sample - use the held-out metrics in /model/info to judge the model."""
        from app.ml.fetch_data import fetch_telco_data
        from app.ml.predict import ModelNotTrainedError, predict
        from app.models.customer import Customer
        from app.models.prediction import Prediction
        from app.models.retention_action import RetentionAction
        from app.models.transaction import Transaction
        from app.services.customer_service import CustomerService

        rng = random.Random(42)
        now = utcnow()
        df = fetch_telco_data().sample(n=min(count, 7043), random_state=42)
        existing = {row[0] for row in db.session.execute(db.select(Customer.external_id)).all()}
        model_ready = True
        created = scored = actions = 0

        for _, row in df.iterrows():
            ext_id = str(row["customerID"])
            if ext_id in existing:
                continue
            tenure = int(row["tenure"])
            total = pd.to_numeric(row["TotalCharges"], errors="coerce")
            churned = row["Churn"] == "Yes"
            profile = {
                col: (int(row[feat]) if col == "senior_citizen" else str(row[feat]))
                for col, feat in PROFILE_TO_FEATURE.items()
            }
            customer = Customer(
                external_id=ext_id,
                name=f"Customer {ext_id}",
                email=f"{ext_id.lower()}@example.com",
                signup_date=(now - timedelta(days=30 * tenure)).date(),
                tenure_months=tenure,
                monthly_charges=float(row["MonthlyCharges"]),
                total_charges=max(float(total) if pd.notna(total) else 0.0, 0.0),
                contract_type=RAW_TO_CONTRACT.get(row["Contract"], "month-to-month"),
                plan=str(row["InternetService"]),
                **profile,
            )
            db.session.add(customer)
            if churned:  # spread churn dates over the last 120 days so period churn rates are meaningful
                customer.is_active = False
                customer.churned_at = now - timedelta(days=rng.randint(1, 120), hours=rng.randint(0, 23))
                customer.signup_date = min(customer.signup_date, customer.churned_at.date() - timedelta(days=30))
            db.session.flush()
            created += 1

            for months_ago in range(min(3, max(tenure, 1))):  # last few monthly invoices
                db.session.add(
                    Transaction(
                        customer_id=customer.id,
                        amount=customer.monthly_charges,
                        type="charge",
                        status="completed",
                        created_at=now - timedelta(days=30 * months_ago + rng.randint(0, 3)),
                    )
                )

            if (
                not model_ready or not customer.is_active
            ):  # churned customers are never scored (nothing left to prevent)
                continue
            try:
                result = predict(CustomerService.to_features(customer))
            except ModelNotTrainedError:
                model_ready = False
                click.echo("No trained model found - customers seeded without predictions (run `flask train`).")
                continue
            prediction = Prediction(
                customer_id=customer.id,
                model_version=result["model_version"],
                churn_probability=result["churn_probability"],
                risk_level=result["risk_level"],
                key_factors=result["key_factors"],
                recommended_action=result["recommended_action"],
                created_at=now - timedelta(days=rng.randint(0, 60), hours=rng.randint(0, 23)),
            )
            db.session.add(prediction)
            db.session.flush()
            scored += 1

            if result["risk_level"] == "High" and customer.is_active and rng.random() < 0.35:
                status = rng.choice(["open", "open", "in_progress", "done"])
                db.session.add(
                    RetentionAction(
                        customer_id=customer.id,
                        prediction_id=prediction.id,
                        action=result["recommended_action"],
                        status=status,
                        completed_at=now if status == "done" else None,
                    )
                )
                actions += 1

        db.session.commit()
        click.echo(f"Seeded {created} customers, {scored} predictions and {actions} retention actions.")

    @app.cli.command("import-csv")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False))
    @click.option("--mode", type=click.Choice(["skip", "update"]), default="skip", show_default=True)
    @click.option("--score/--no-score", default=True, show_default=True, help="Score imported active customers.")
    def import_csv(path: str, mode: str, score: bool) -> None:
        """Bulk-load a customer CSV (e.g. data/sample/customers_100k.csv) - same engine as the dashboard import."""
        import shutil
        import time

        from app.models.import_job import ImportJob
        from app.services.import_service import ImportService

        folder = os.path.join(app.instance_path, "imports")
        os.makedirs(folder, exist_ok=True)
        job = ImportJob(filename=os.path.basename(path), mode=mode, score_after=score)
        with open(path, "rb") as handle:
            job.total_rows = max(sum(1 for _ in handle) - 1, 0)
        db.session.add(job)
        db.session.commit()
        tmp = os.path.join(folder, f"job_{job.id}.csv")
        shutil.copyfile(path, tmp)
        started = time.time()
        ImportService().run(job.id, tmp)
        db.session.refresh(job)
        click.echo(f"{job.status}: {job.message} ({time.time() - started:.1f}s, scored {job.scored_count:,})")
        for err in job.errors[:5]:
            click.echo(f"  row {err['row']}: {err['message']}")

    @app.cli.command("seed-actions")
    @click.option("--count", default=1500, show_default=True)
    def seed_actions(count: int) -> None:
        """DEMO ONLY: raise retention actions for a sample of High-risk customers, with mixed statuses/outcomes."""
        from sqlalchemy import insert, select

        from app.models.prediction import Prediction
        from app.models.retention_action import RetentionAction
        from app.repositories.prediction_repository import latest_prediction_ids

        rng = random.Random(7)
        now = utcnow()
        rows = db.session.execute(
            select(Prediction.id, Prediction.customer_id, Prediction.recommended_action)
            .where(Prediction.id.in_(latest_prediction_ids()), Prediction.risk_level == "High")
            .order_by(Prediction.churn_probability.desc())
            .limit(count * 3)
        ).all()
        rng.shuffle(rows)
        out = []
        for pid, cid, action in rows[:count]:
            status = rng.choices(["open", "in_progress", "done"], weights=[40, 25, 35])[0]
            done = status == "done"
            out.append(
                {
                    "customer_id": cid,
                    "prediction_id": pid,
                    "action": action,
                    "status": status,
                    "due_date": (now + timedelta(days=rng.randint(-10, 21))).date(),
                    "offer_cost": float(rng.choice([10, 15, 20, 25, 30])),
                    "outcome": rng.choices(["retained", "lost"], weights=[62, 38])[0] if done else None,
                    "created_at": now - timedelta(days=rng.randint(0, 40)),
                    "updated_at": now,
                    "completed_at": now if done else None,
                }
            )
        if out:
            db.session.execute(insert(RetentionAction), out)
            db.session.commit()
        click.echo(f"Created {len(out)} demo actions.")
