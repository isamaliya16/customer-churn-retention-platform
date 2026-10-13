"""CSV exports: customers, predictions, high-risk list (streamed)."""

from flask import Response
from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.repositories.customer_repository import CustomerRepository
from app.repositories.prediction_repository import PredictionRepository
from app.services.analytics_service import AnalyticsService
from app.services.export_service import rows_to_csv
from app.utils.decorators import role_required
from app.utils.time import utcnow

blp = Blueprint("exports", __name__, url_prefix="/export", description="CSV exports")

MAX_EXPORT_ROWS = 250_000


def _csv_response(name: str, header: list[str], rows) -> Response:
    # Materialise rows NOW: the DB session is closed once the request ends, before a lazy stream would run.
    rows = list(rows)
    stamp = utcnow().strftime("%Y%m%d")
    return Response(
        rows_to_csv(header, rows),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}-{stamp}.csv"'},
    )


@blp.route("/customers.csv")
class ExportCustomers(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    def get(self):
        """All customers (with latest risk level) as CSV."""
        repo = CustomerRepository()
        items = repo.export_rows(MAX_EXPORT_ROWS)
        header = [
            "id",
            "external_id",
            "name",
            "email",
            "contract_type",
            "plan",
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "is_active",
            "churned_at",
            "latest_risk_level",
            "latest_churn_probability",
        ]
        rows = (
            [
                c.id,
                c.external_id,
                c.name,
                c.email,
                c.contract_type,
                c.plan,
                c.tenure_months,
                c.monthly_charges,
                c.total_charges,
                c.is_active,
                c.churned_at.isoformat() if c.churned_at else "",
                risk or "",
                prob if prob is not None else "",
            ]
            for c, risk, prob in items
        )
        return _csv_response("customers", header, rows)


@blp.route("/predictions.csv")
class ExportPredictions(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    def get(self):
        """Prediction history as CSV (newest first, max 250,000 rows)."""
        items, _ = PredictionRepository().list_filtered(1, MAX_EXPORT_ROWS)
        header = [
            "id",
            "customer_id",
            "customer_name",
            "churn_probability",
            "risk_level",
            "recommended_action",
            "model_version",
            "created_at",
        ]
        rows = (
            [
                p.id,
                p.customer_id,
                p.customer_name,
                p.churn_probability,
                p.risk_level,
                p.recommended_action,
                p.model_version,
                p.created_at.isoformat(),
            ]
            for p in items
        )
        return _csv_response("predictions", header, rows)


@blp.route("/high-risk.csv")
class ExportHighRisk(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    def get(self):
        """The High-risk customer list as CSV, highest probability first."""
        items, _ = AnalyticsService().high_risk_customers(1, MAX_EXPORT_ROWS)
        header = [
            "customer_id",
            "name",
            "email",
            "contract_type",
            "tenure_months",
            "monthly_charges",
            "churn_probability",
            "recommended_action",
        ]
        rows = (
            [
                i["customer_id"],
                i["name"],
                i["email"],
                i["contract_type"],
                i["tenure_months"],
                i["monthly_charges"],
                i["churn_probability"],
                i["recommended_action"],
            ]
            for i in items
        )
        return _csv_response("high-risk", header, rows)
