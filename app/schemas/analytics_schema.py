"""Response schemas for analytics, model info and admin activity (they drive the OpenAPI docs)."""

from marshmallow import Schema, fields, validate

from app.constants import CONTRACT_TYPES
from app.schemas.common_schema import PaginationQuerySchema
from app.services.insights_service import EXPLORER_DIMENSIONS


class DaysQuerySchema(Schema):
    days = fields.Int(
        load_default=30, validate=validate.Range(min=1, max=365), metadata={"description": "Look-back window"}
    )


class SegmentsQuerySchema(Schema):
    days = fields.Int(load_default=90, validate=validate.Range(min=1, max=365))


class TrendsQuerySchema(Schema):
    days = fields.Int(load_default=90, validate=validate.Range(min=7, max=365))


class HighRiskQuerySchema(PaginationQuerySchema):
    sort = fields.Str(
        load_default="desc", validate=validate.OneOf(("asc", "desc")), metadata={"description": "Sort by probability"}
    )
    contract_type = fields.Str(validate=validate.OneOf(CONTRACT_TYPES))


class OverviewSchema(Schema):
    period_days = fields.Int()
    total_customers = fields.Int()
    active_customers = fields.Int()
    churned_customers = fields.Int()
    active_at_period_start = fields.Int()
    churned_in_period = fields.Int()
    churn_rate = fields.Float(metadata={"description": "churned in period / active at period start"})
    retention_rate = fields.Float()
    lifetime_churn_rate = fields.Float()
    revenue_at_risk = fields.Float()
    monthly_recurring_revenue = fields.Float()
    revenue_at_risk_share = fields.Float()
    high_risk_count = fields.Int()
    estimated_clv = fields.Float()
    prediction_volume = fields.Int()
    open_actions = fields.Int()
    completed_actions = fields.Int()


class SegmentRowSchema(Schema):
    segment = fields.Str()
    total_customers = fields.Int()
    lifetime_churned = fields.Int()
    active_at_period_start = fields.Int()
    churned_in_period = fields.Int()
    churn_rate = fields.Float()
    lifetime_churn_rate = fields.Float()


class SegmentsSchema(Schema):
    period_days = fields.Int()
    by_contract_type = fields.List(fields.Nested(SegmentRowSchema))
    by_plan = fields.List(fields.Nested(SegmentRowSchema))
    by_tenure_bucket = fields.List(fields.Nested(SegmentRowSchema))


class TrendWeekSchema(Schema):
    week = fields.Str()
    active_at_week_start = fields.Int()
    churned = fields.Int()
    churn_rate = fields.Float()
    prediction_volume = fields.Int()
    avg_churn_probability = fields.Float()
    high_risk_count = fields.Int()


class TrendsSchema(Schema):
    period_days = fields.Int()
    weekly = fields.List(fields.Nested(TrendWeekSchema))


class RiskDistributionSchema(Schema):
    counts = fields.Dict(keys=fields.Str(), values=fields.Int())
    unscored = fields.Int()
    total_customers = fields.Int()


class RevenueRowSchema(Schema):
    segment = fields.Str()
    customers = fields.Int()
    revenue_at_risk = fields.Float()


class RevenueAtRiskSchema(Schema):
    total = fields.Float()
    by_contract_type = fields.List(fields.Nested(RevenueRowSchema))
    by_plan = fields.List(fields.Nested(RevenueRowSchema))


class HighRiskItemSchema(Schema):
    customer_id = fields.Int()
    prediction_id = fields.Int()
    name = fields.Str()
    email = fields.Str()
    contract_type = fields.Str()
    tenure_months = fields.Int()
    monthly_charges = fields.Float()
    churn_probability = fields.Float()
    recommended_action = fields.Str()
    is_active = fields.Bool()


class ModelInfoSchema(Schema):
    model_version = fields.Str()
    algorithm = fields.Str()
    trained_at = fields.Str()
    n_train_rows = fields.Int()
    n_test_rows = fields.Int()
    n_features = fields.Int()
    churn_rate_in_training = fields.Float(allow_none=True)
    selection_criteria = fields.Raw()
    cv_metrics = fields.Raw()
    held_out_metrics = fields.Raw()
    confusion_matrix = fields.Raw(allow_none=True)
    roc_curves = fields.Raw()
    outlier_report = fields.Raw()
    data_source = fields.Str()
    flag_threshold = fields.Float(allow_none=True)
    calibration = fields.Raw(allow_none=True)
    feature_importance = fields.List(fields.Raw())
    risk_thresholds = fields.Raw()
    recommendation_rules = fields.List(fields.Raw())


class ActivityStatsSchema(Schema):
    days = fields.Int()
    total_requests = fields.Int()
    error_rate = fields.Float()
    avg_latency_ms = fields.Float()
    p95_latency_ms = fields.Float()
    requests_per_day = fields.List(fields.Raw())
    top_endpoints = fields.List(fields.Raw())
    status_codes = fields.List(fields.Raw())


class CampaignQuerySchema(Schema):
    offer_cost = fields.Float(load_default=20.0, validate=validate.Range(min=0.01, max=100000))
    success_rate = fields.Float(
        load_default=0.30,
        validate=validate.Range(min=0.01, max=1),
        metadata={"description": "Share of at-risk customers an offer retains"},
    )
    horizon_months = fields.Int(load_default=12, validate=validate.Range(min=1, max=60))
    budget = fields.Float(validate=validate.Range(min=0), allow_none=True, load_default=None)
    min_probability = fields.Float(load_default=0.0, validate=validate.Range(min=0, max=1))


class CampaignSchema(Schema):
    assumptions = fields.Dict()
    scored_active_customers = fields.Int()
    targeted_customers = fields.Int()
    targeted_share = fields.Float()
    total_cost = fields.Float()
    expected_churners_in_target = fields.Float()
    expected_customers_saved = fields.Float()
    expected_revenue_saved = fields.Float()
    net_value = fields.Float()
    roi = fields.Float()
    avg_probability_targeted = fields.Float()
    customer_ids = fields.List(fields.Int())
    top_customers = fields.List(fields.Dict())
    gain_curve = fields.List(fields.Dict())


class ExplorerQuerySchema(Schema):
    dimension = fields.Str(load_default="contract_type", validate=validate.OneOf(tuple(EXPLORER_DIMENSIONS)))


class ExplorerSchema(Schema):
    dimension = fields.Str()
    segments = fields.List(fields.Dict())


class OutcomesSchema(Schema):
    total_actions = fields.Int()
    closed_actions = fields.Int()
    retained = fields.Int()
    lost = fields.Int()
    success_rate = fields.Float(allow_none=True)
    monthly_revenue_retained = fields.Float()
    total_offer_cost = fields.Float()
    by_action = fields.List(fields.Dict())
