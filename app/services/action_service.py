"""Business logic for retention actions (follow-up tasks on at-risk customers)."""

from __future__ import annotations

from sqlalchemy import insert, select

from app.models.prediction import Prediction
from app.models.retention_action import RetentionAction
from app.models.user import User
from app.repositories.action_repository import ActionRepository
from app.repositories.customer_repository import CustomerRepository
from app.repositories.prediction_repository import PredictionRepository, latest_prediction_ids
from app.utils.errors import NotFoundAPIError, ValidationAPIError
from app.utils.time import utcnow


class ActionService:
    def __init__(self):
        self.repo = ActionRepository()
        self.customers = CustomerRepository(self.repo.session)
        self.predictions = PredictionRepository(self.repo.session)

    def create(self, data: dict, user: User | None) -> RetentionAction:
        if not self.customers.get_by_id(data["customer_id"]):
            raise NotFoundAPIError(f"Customer {data['customer_id']} not found.")
        prediction_id = data.get("prediction_id")
        if prediction_id:
            prediction = self.predictions.get_by_id(prediction_id)
            if not prediction or prediction.customer_id != data["customer_id"]:
                raise ValidationAPIError("prediction_id does not belong to this customer.")
        action = RetentionAction(
            customer_id=data["customer_id"],
            prediction_id=prediction_id,
            action=data["action"],
            notes=data.get("notes"),
            due_date=data.get("due_date"),
            offer_cost=data.get("offer_cost"),
            created_by=user.id if user else None,
        )
        self.repo.add(action)
        return self.repo.with_customer_name(action)

    def create_bulk(self, data: dict, user: User | None) -> dict:
        """One open action per ACTIVE customer (default text = the latest prediction's recommended action).
        Customers that already have an open / in-progress action are skipped, so re-running a campaign is safe."""
        ids = list(dict.fromkeys(data["customer_ids"]))
        active = set(self.customers.active_ids(only_ids=ids))
        busy = set(
            self.repo.session.scalars(
                select(RetentionAction.customer_id).where(
                    RetentionAction.customer_id.in_(ids), RetentionAction.status.in_(("open", "in_progress"))
                )
            )
        )
        target = [i for i in ids if i in active and i not in busy]
        latest = {}
        if target:
            latest = {
                p.customer_id: p
                for p in self.repo.session.scalars(
                    select(Prediction).where(Prediction.id.in_(latest_prediction_ids(target)))
                )
            }
        rows = []
        for customer_id in target:
            prediction = latest.get(customer_id)
            text = data.get("action") or (prediction.recommended_action if prediction else "Reach out to the customer")
            rows.append(
                {
                    "customer_id": customer_id,
                    "prediction_id": prediction.id if prediction else None,
                    "action": text[:255],
                    "status": "open",
                    "notes": data.get("notes"),
                    "due_date": data.get("due_date"),
                    "offer_cost": data.get("offer_cost"),
                    "created_by": user.id if user else None,
                    "created_at": utcnow(),
                    "updated_at": utcnow(),
                }
            )
        if rows:
            self.repo.session.execute(insert(RetentionAction), rows)
            self.repo.session.commit()
        return {"requested": len(ids), "created": len(rows), "skipped": len(ids) - len(rows)}

    def get_or_404(self, action_id: int) -> RetentionAction:
        action = self.repo.get_by_id(action_id)
        if not action:
            raise NotFoundAPIError(f"Action {action_id} not found.")
        return self.repo.with_customer_name(action)

    def update(self, action_id: int, data: dict) -> RetentionAction:
        action = self.get_or_404(action_id)
        for key, value in data.items():
            setattr(action, key, value)
        if data.get("outcome") and action.status in ("open", "in_progress"):
            action.status = data["status"] = "done"  # recording an outcome closes the action
        if "status" in data:
            action.completed_at = utcnow() if data["status"] in ("done", "dismissed") else None
        self.repo.commit()
        return action

    def delete(self, action_id: int) -> None:
        self.repo.delete(self.get_or_404(action_id))

    def list_actions(self, page: int, per_page: int, **filters):
        return self.repo.list_filtered(page, per_page, **filters)
