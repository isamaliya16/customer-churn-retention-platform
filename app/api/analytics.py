"""GET /analytics/{overview,segments,trends,high-risk,risk-distribution,revenue-at-risk}.

All responses are cached (5-15 min) and invalidated on customer / prediction / action writes.
"""

from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.extensions import cache
from app.schemas.analytics_schema import (
    CampaignQuerySchema,
    CampaignSchema,
    DaysQuerySchema,
    ExplorerQuerySchema,
    ExplorerSchema,
    HighRiskItemSchema,
    HighRiskQuerySchema,
    OutcomesSchema,
    OverviewSchema,
    RevenueAtRiskSchema,
    RiskDistributionSchema,
    SegmentsQuerySchema,
    SegmentsSchema,
    TrendsQuerySchema,
    TrendsSchema,
)
from app.schemas.common_schema import page_envelope, paginated
from app.services.analytics_service import AnalyticsService
from app.services.insights_service import InsightsService
from app.utils.decorators import role_required

blp = Blueprint("analytics", __name__, url_prefix="/analytics", description="Churn analytics and KPIs")


@blp.route("/overview")
class Overview(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(DaysQuerySchema, location="query")
    @blp.response(200, OverviewSchema)
    @cache.cached(timeout=600, query_string=True)
    def get(self, args):
        """KPIs: churn / retention rate, revenue at risk, CLV, prediction volume, open actions."""
        return AnalyticsService().overview(args["days"])


@blp.route("/segments")
class Segments(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(SegmentsQuerySchema, location="query")
    @blp.response(200, SegmentsSchema)
    @cache.cached(timeout=900, query_string=True)
    def get(self, args):
        """Churn by contract type, plan and tenure bucket."""
        return AnalyticsService().segments(args["days"])


@blp.route("/trends")
class Trends(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(TrendsQuerySchema, location="query")
    @blp.response(200, TrendsSchema)
    @cache.cached(timeout=300, query_string=True)
    def get(self, args):
        """Weekly churn rate and prediction volume over time."""
        return AnalyticsService().trends(args["days"])


@blp.route("/risk-distribution")
class RiskDistribution(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.response(200, RiskDistributionSchema)
    @cache.cached(timeout=300)
    def get(self):
        """How many customers sit in each risk tier (by latest prediction)."""
        return AnalyticsService().risk_distribution()


@blp.route("/revenue-at-risk")
class RevenueAtRisk(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.response(200, RevenueAtRiskSchema)
    @cache.cached(timeout=300)
    def get(self):
        """Monthly revenue at risk (High tier), broken down by contract type and plan."""
        return AnalyticsService().revenue_at_risk()


@blp.route("/high-risk")
class HighRisk(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(HighRiskQuerySchema, location="query")
    @blp.response(200, paginated(HighRiskItemSchema))
    @cache.cached(timeout=300, query_string=True)
    def get(self, args):
        """Paginated High-risk customers, sortable by probability."""
        items, total = AnalyticsService().high_risk_customers(
            args["page"], args["per_page"], args["sort"] == "desc", args.get("contract_type")
        )
        return page_envelope(items, total, args["page"], args["per_page"])


@blp.route("/campaign")
class Campaign(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(CampaignQuerySchema, location="query")
    @blp.response(200, CampaignSchema)
    @cache.cached(timeout=300, query_string=True)
    def get(self, args):
        """Campaign planner: whom to contact, expected saves, revenue, cost and ROI, plus a lift / gain curve."""
        return InsightsService().campaign(**args)


@blp.route("/explorer")
class Explorer(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(ExplorerQuerySchema, location="query")
    @blp.response(200, ExplorerSchema)
    @cache.cached(timeout=600, query_string=True)
    def get(self, args):
        """Segment explorer: customers, churn, average risk and revenue at risk for any profile dimension."""
        return InsightsService().explorer(args["dimension"])


@blp.route("/outcomes")
class Outcomes(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.response(200, OutcomesSchema)
    @cache.cached(timeout=300)
    def get(self):
        """Did the retention actions work? Retained vs lost, success rate and revenue retained."""
        return InsightsService().outcomes()
