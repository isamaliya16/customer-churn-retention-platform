"""Marshmallow schemas for the prediction endpoints."""

from marshmallow import Schema, ValidationError, fields, validate, validates_schema

from app.constants import (
    ADDON_VALUES,
    GENDERS,
    INTERNET_SERVICES,
    MULTIPLE_LINES,
    PAYMENT_METHODS,
    RISK_LEVELS,
    YES_NO,
)
from app.schemas.common_schema import PaginationQuerySchema


class FeaturesSchema(Schema):
    """Raw model inputs (IBM Telco column names). tenure, MonthlyCharges and Contract are required;
    anything omitted is imputed exactly as during training."""

    tenure = fields.Int(required=True, validate=validate.Range(min=0, max=1200), metadata={"example": 3})
    MonthlyCharges = fields.Float(required=True, validate=validate.Range(min=0, max=100000), metadata={"example": 89.9})
    Contract = fields.Str(required=True, validate=validate.OneOf(("Month-to-month", "One year", "Two year")))
    SeniorCitizen = fields.Int(validate=validate.OneOf((0, 1)))
    gender = fields.Str(validate=validate.OneOf(GENDERS))
    Partner = fields.Str(validate=validate.OneOf(YES_NO))
    Dependents = fields.Str(validate=validate.OneOf(YES_NO))
    PhoneService = fields.Str(validate=validate.OneOf(YES_NO))
    MultipleLines = fields.Str(validate=validate.OneOf(MULTIPLE_LINES))
    InternetService = fields.Str(validate=validate.OneOf(INTERNET_SERVICES))
    OnlineSecurity = fields.Str(validate=validate.OneOf(ADDON_VALUES))
    OnlineBackup = fields.Str(validate=validate.OneOf(ADDON_VALUES))
    DeviceProtection = fields.Str(validate=validate.OneOf(ADDON_VALUES))
    TechSupport = fields.Str(validate=validate.OneOf(ADDON_VALUES))
    StreamingTV = fields.Str(validate=validate.OneOf(ADDON_VALUES))
    StreamingMovies = fields.Str(validate=validate.OneOf(ADDON_VALUES))
    PaperlessBilling = fields.Str(validate=validate.OneOf(YES_NO))
    PaymentMethod = fields.Str(validate=validate.OneOf(PAYMENT_METHODS))


class PredictRequestSchema(Schema):
    """POST /predict takes EITHER a customer_id (stored profile) OR a raw `features` payload."""

    customer_id = fields.Int(validate=validate.Range(min=1))
    features = fields.Nested(FeaturesSchema)

    @validates_schema
    def validate_one_of(self, data, **kwargs):
        if not data.get("customer_id") and not data.get("features"):
            raise ValidationError("Provide either 'customer_id' or 'features'.")
        if data.get("customer_id") and data.get("features"):
            raise ValidationError("Provide only one of 'customer_id' or 'features', not both.")


class KeyFactorSchema(Schema):
    feature = fields.Str(metadata={"example": "Contract"})
    impact = fields.Float(metadata={"example": 0.22})
    direction = fields.Str(metadata={"example": "increases churn risk"})


class PredictResponseSchema(Schema):
    id = fields.Int(allow_none=True, metadata={"description": "Null when scoring a raw payload (not persisted)"})
    customer_id = fields.Int(allow_none=True)
    churn_probability = fields.Float()
    risk_level = fields.Str()
    key_factors = fields.List(fields.Nested(KeyFactorSchema))
    recommended_action = fields.Str()
    model_version = fields.Str()
    created_at = fields.Str(allow_none=True)


class PredictionSchema(Schema):
    id = fields.Int(dump_only=True)
    customer_id = fields.Int(dump_only=True)
    customer_name = fields.Str(dump_only=True, allow_none=True)
    model_version = fields.Str(dump_only=True)
    churn_probability = fields.Float(dump_only=True)
    risk_level = fields.Str(dump_only=True)
    key_factors = fields.List(fields.Nested(KeyFactorSchema), dump_only=True)
    recommended_action = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)


class PredictionQuerySchema(PaginationQuerySchema):
    customer_id = fields.Int(validate=validate.Range(min=1))
    risk_level = fields.Str(validate=validate.OneOf(RISK_LEVELS))
    date_from = fields.Date()
    date_to = fields.Date()


class BatchPredictRequestSchema(Schema):
    """Score many customers at once: explicit ids, or a scope."""

    customer_ids = fields.List(fields.Int(validate=validate.Range(min=1)), validate=validate.Length(min=1, max=5000))
    scope = fields.Str(
        validate=validate.OneOf(("active", "unscored", "all")),
        metadata={"description": "active = still-subscribed customers; unscored = never scored; all = everyone"},
    )
    limit = fields.Int(load_default=500, validate=validate.Range(min=1, max=200000))

    @validates_schema
    def validate_one_of(self, data, **kwargs):
        if bool(data.get("customer_ids")) == bool(data.get("scope")):
            raise ValidationError("Provide exactly one of 'customer_ids' or 'scope'.")


class BatchPredictResponseSchema(Schema):
    requested = fields.Int()
    scored = fields.Int()
    skipped = fields.Int()
    by_risk = fields.Dict(keys=fields.Str(), values=fields.Int())
    model_version = fields.Str()
