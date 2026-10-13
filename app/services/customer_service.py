"""Business logic for customers, their billing transactions and feature extraction."""

from __future__ import annotations

import uuid
from datetime import timedelta

from sqlalchemy.exc import IntegrityError

from app.constants import CONTRACT_TO_RAW, PROFILE_TO_FEATURE
from app.models.customer import Customer
from app.models.transaction import Transaction
from app.repositories.customer_repository import CustomerRepository
from app.repositories.prediction_repository import PredictionRepository
from app.utils.errors import ConflictAPIError, NotFoundAPIError, ValidationAPIError
from app.utils.time import utcnow


class CustomerService:
    def __init__(self, repo: CustomerRepository | None = None, prediction_repo: PredictionRepository | None = None):
        self.repo = repo or CustomerRepository()
        self.prediction_repo = prediction_repo or PredictionRepository(self.repo.session)

    # --- create / read / update / delete ---------------------------------------------
    def create(self, data: dict) -> Customer:
        data = dict(data)
        if self.repo.get_by_email(data["email"]):
            raise ConflictAPIError(f"A customer with email {data['email']!r} already exists.")

        tenure = data.get("tenure_months") or 0
        data["external_id"] = data.get("external_id") or f"CUST-{uuid.uuid4().hex[:8].upper()}"
        if self.repo.get_by_external_id(data["external_id"]):
            raise ConflictAPIError(f"A customer with external_id {data['external_id']!r} already exists.")
        data["signup_date"] = data.get("signup_date") or (utcnow() - timedelta(days=30 * tenure)).date()
        data["plan"] = data.get("plan") or data.get("internet_service") or "Standard"
        if data.get("total_charges") is None:
            data["total_charges"] = round(data["monthly_charges"] * tenure, 2)

        try:
            return self.repo.add(Customer(**data))
        except ValueError as exc:
            raise ValidationAPIError(str(exc)) from exc
        except IntegrityError as exc:  # unique race between the pre-check and the insert
            self.repo.session.rollback()
            raise ConflictAPIError("A customer with this email or external_id already exists.") from exc

    def get_or_404(self, customer_id: int) -> Customer:
        customer = self.repo.get_by_id(customer_id)
        if not customer:
            raise NotFoundAPIError(f"Customer {customer_id} not found.")
        return customer

    def get_with_risk(self, customer_id: int) -> Customer:
        customer = self.get_or_404(customer_id)
        self._attach_latest_risk([customer])
        return customer

    def list_customers(self, page: int, per_page: int, **filters):
        items, total = self.repo.list_filtered(page=page, per_page=per_page, **filters)
        self._attach_latest_risk(items)
        return items, total

    def update(self, customer_id: int, data: dict) -> Customer:
        customer = self.get_or_404(customer_id)
        if "email" in data and data["email"].lower() != customer.email and self.repo.get_by_email(data["email"]):
            raise ConflictAPIError(f"A customer with email {data['email']!r} already exists.")
        try:
            self.repo.update(customer, data)
        except ValueError as exc:
            self.repo.session.rollback()
            raise ValidationAPIError(str(exc)) from exc
        except IntegrityError as exc:
            self.repo.session.rollback()
            raise ConflictAPIError("That email is already in use.") from exc
        self._attach_latest_risk([customer])
        return customer

    def delete(self, customer_id: int) -> None:
        self.repo.delete(self.get_or_404(customer_id))

    def _attach_latest_risk(self, customers: list[Customer]) -> None:
        """One query for the whole page (no N+1): tag each customer with its latest risk + probability."""
        latest = self.prediction_repo.latest_summary_for([c.id for c in customers])
        for customer in customers:
            risk, probability = latest.get(customer.id, (None, None))
            customer.latest_risk_level = risk
            customer.latest_churn_probability = probability

    # --- billing history -------------------------------------------------------------
    def list_transactions(self, customer_id: int) -> list[Transaction]:
        self.get_or_404(customer_id)
        return self.repo.recent_transactions(customer_id)

    def add_transaction(self, customer_id: int, data: dict) -> Transaction:
        self.get_or_404(customer_id)
        try:
            transaction = Transaction(customer_id=customer_id, **data)
        except ValueError as exc:
            raise ValidationAPIError(str(exc)) from exc
        return self.repo.add(transaction)

    # --- ML feature extraction -------------------------------------------------------
    @staticmethod
    def to_features(customer: Customer) -> dict:
        """Stored customer -> raw model features. Unknown profile fields are omitted so the fitted
        pipeline imputes them exactly as it did in training."""
        features: dict = {
            "tenure": customer.tenure_months,
            "MonthlyCharges": customer.monthly_charges,
            "Contract": CONTRACT_TO_RAW.get(customer.contract_type, "Month-to-month"),
        }
        for column, feature in PROFILE_TO_FEATURE.items():
            value = getattr(customer, column)
            if value is not None:
                features[feature] = value
        return features
