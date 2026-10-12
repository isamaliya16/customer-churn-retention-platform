"""POST /predict, POST /predict/batch, GET /predictions, GET /predictions/<id>."""

from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.extensions import limiter
from app.schemas.common_schema import page_envelope, paginated
from app.schemas.prediction_schema import (
    BatchPredictRequestSchema,
    BatchPredictResponseSchema,
    PredictionQuerySchema,
    PredictionSchema,
    PredictRequestSchema,
    PredictResponseSchema,
)
from app.services.prediction_service import PredictionService
from app.utils.cache import invalidate_analytics_cache
from app.utils.decorators import role_required

blp = Blueprint("predictions", __name__, description="Real-time churn prediction and prediction history")


@blp.route("/predict")
class Predict(MethodView):
    @errors(blp, 400, 401, 403, 404, 429)
    @role_required("admin", "analyst")
    @limiter.limit(lambda: current_app.config["RATE_LIMIT_PREDICT"])
    @blp.arguments(PredictRequestSchema)
    @blp.response(200, PredictResponseSchema)
    def post(self, payload):
        """Score churn risk by `customer_id` (stored + persisted) or by a raw `features` payload (not stored)."""
        result = PredictionService().predict(
            customer_id=payload.get("customer_id"), raw_features=payload.get("features")
        )
        if result["id"] is not None:
            invalidate_analytics_cache()
        return result


@blp.route("/predict/batch")
class PredictBatch(MethodView):
    @errors(blp, 400, 401, 403, 429)
    @role_required("admin", "analyst")
    @limiter.limit(lambda: current_app.config["RATE_LIMIT_BATCH"])
    @blp.arguments(BatchPredictRequestSchema)
    @blp.response(200, BatchPredictResponseSchema)
    def post(self, payload):
        """Score many customers in one call (explicit ids, or scope = active / unscored / all)."""
        result = PredictionService().predict_batch(
            customer_ids=payload.get("customer_ids"), scope=payload.get("scope"), limit=payload["limit"]
        )
        invalidate_analytics_cache()
        return result


@blp.route("/predictions")
class Predictions(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(PredictionQuerySchema, location="query")
    @blp.response(200, paginated(PredictionSchema))
    def get(self, args):
        """Prediction history, filterable by customer / date range / risk level, newest first."""
        page, per_page = args.pop("page"), args.pop("per_page")
        items, total = PredictionService().list_predictions(page, per_page, **args)
        return page_envelope(items, total, page, per_page)


@blp.route("/predictions/<int:prediction_id>")
class PredictionDetail(MethodView):
    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.response(200, PredictionSchema)
    def get(self, prediction_id):
        """A single prediction, including its key factors."""
        return PredictionService().get_or_404(prediction_id)
