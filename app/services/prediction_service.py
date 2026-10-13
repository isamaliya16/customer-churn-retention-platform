"""Business logic for running + persisting churn predictions."""

from __future__ import annotations

from app.ml.predict import ModelNotTrainedError, features_to_frame
from app.ml.predict import predict as ml_predict
from app.ml.predict import predict_frame as ml_predict_frame
from app.models.prediction import Prediction
from app.repositories.customer_repository import CustomerRepository
from app.repositories.prediction_repository import PredictionRepository
from app.services.customer_service import CustomerService
from app.utils.errors import NotFoundAPIError, ValidationAPIError
from app.utils.time import utcnow


class PredictionService:
    def __init__(
        self,
        prediction_repo: PredictionRepository | None = None,
        customer_repo: CustomerRepository | None = None,
    ):
        self.prediction_repo = prediction_repo or PredictionRepository()
        self.customer_repo = customer_repo or CustomerRepository()
        self.customer_service = CustomerService(self.customer_repo)

    @staticmethod
    def _score(features: dict) -> dict:
        try:
            return ml_predict(features)
        except ModelNotTrainedError as exc:
            raise ValidationAPIError(str(exc), code="model_not_trained") from exc

    @staticmethod
    def _to_row(customer_id: int, result: dict) -> Prediction:
        return Prediction(
            customer_id=customer_id,
            model_version=result["model_version"],
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            key_factors=result["key_factors"],
            recommended_action=result["recommended_action"],
        )

    def predict(self, customer_id: int | None = None, raw_features: dict | None = None) -> dict:
        """Score a stored customer (persisted) or an ad-hoc feature payload (returned, not persisted)."""
        if customer_id is None:
            result = self._score(raw_features or {})
            return {**result, "id": None, "customer_id": None, "created_at": None}

        customer = self.customer_repo.get_by_id(customer_id)
        if not customer:
            raise NotFoundAPIError(f"Customer {customer_id} not found.")
        if not customer.is_active:
            raise ValidationAPIError(
                f"Customer {customer_id} has already churned; scoring only applies to active customers.",
                code="customer_inactive",
            )
        result = self._score(self.customer_service.to_features(customer))
        prediction = self.prediction_repo.add(self._to_row(customer.id, result))
        return {
            **result,
            "id": prediction.id,
            "customer_id": customer.id,
            "created_at": prediction.created_at.isoformat(),
        }

    SCORE_CHUNK = 5000

    def predict_batch(self, customer_ids: list[int] | None = None, scope: str | None = None, limit: int = 500) -> dict:
        """Score many ACTIVE customers (vectorised, chunked). Unknown ids and churned customers count as skipped."""
        if customer_ids:
            ids = self.customer_repo.active_ids(only_ids=customer_ids)
            requested = len(set(customer_ids))
        elif scope == "unscored":
            ids = self.customer_repo.unscored_ids(limit)
            requested = len(ids)
        else:
            ids = self.customer_repo.active_ids(limit=limit, active_only=scope == "active")
            requested = len(ids)
        summary = self.score_ids(ids)
        summary["requested"] = requested
        summary["skipped"] = requested - summary["scored"]
        return summary

    def score_ids(self, ids: list[int], progress=None) -> dict:
        """Score the given (active) customers in chunks with the vectorised model path and bulk-insert the results.

        `progress(done)` is called after each chunk. Returns {scored, by_risk, model_version}."""
        by_risk = {"Low": 0, "Medium": 0, "High": 0}
        model_version = ""
        scored = 0
        for start in range(0, len(ids), self.SCORE_CHUNK):
            chunk = ids[start : start + self.SCORE_CHUNK]
            feature_rows = self.customer_repo.feature_rows(chunk)
            if not feature_rows:
                continue
            try:
                results = ml_predict_frame(features_to_frame([f for _, f in feature_rows]))
            except ModelNotTrainedError as exc:
                raise ValidationAPIError(str(exc), code="model_not_trained") from exc
            now = utcnow()
            rows = []
            for (customer_id, _), result in zip(feature_rows, results):
                by_risk[result["risk_level"]] += 1
                model_version = result["model_version"]
                rows.append(
                    {
                        "customer_id": customer_id,
                        "model_version": result["model_version"],
                        "churn_probability": result["churn_probability"],
                        "risk_level": result["risk_level"],
                        "key_factors": result["key_factors"],
                        "recommended_action": result["recommended_action"],
                        "created_at": now,
                    }
                )
            self.prediction_repo.bulk_insert(rows)
            scored += len(rows)
            if progress:
                progress(scored)
        return {"scored": scored, "by_risk": by_risk, "model_version": model_version}

    def get_or_404(self, prediction_id: int) -> Prediction:
        prediction = self.prediction_repo.get_with_customer_name(prediction_id)
        if not prediction:
            raise NotFoundAPIError(f"Prediction {prediction_id} not found.")
        return prediction

    def list_predictions(self, page: int, per_page: int, **filters):
        return self.prediction_repo.list_filtered(page=page, per_page=per_page, **filters)
