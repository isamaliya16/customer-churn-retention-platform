"""Marshmallow schemas for Customer create/update/read + query params."""

from marshmallow import Schema, fields, validate

from app.constants import CONTRACT_TYPES, PROFILE_CHOICES, RISK_LEVELS
from app.schemas.common_schema import PaginationQuerySchema


def _profile_fields() -> dict:
    """Optional service-profile fields, validated against the Telco vocabulary."""
    result = {
        name: fields.Str(validate=validate.OneOf(choices), allow_none=True) for name, choices in PROFILE_CHOICES.items()
    }
    result["senior_citizen"] = fields.Int(validate=validate.OneOf((0, 1)), allow_none=True)
    return result


class CustomerSchema(Schema):
    """Create + read schema. Only name, email, monthly_charges and contract_type are required."""

    id = fields.Int(dump_only=True)
    external_id = fields.Str(validate=validate.Length(min=1, max=64), load_default=None)
    name = fields.Str(required=True, validate=validate.Length(min=1, max=120))
    email = fields.Email(required=True)
    signup_date = fields.Date(load_default=None)
    tenure_months = fields.Int(validate=validate.Range(min=0, max=1200), load_default=0)
    monthly_charges = fields.Float(required=True, validate=validate.Range(min=0, max=100000))
    total_charges = fields.Float(validate=validate.Range(min=0), load_default=None)
    contract_type = fields.Str(required=True, validate=validate.OneOf(CONTRACT_TYPES))
    plan = fields.Str(validate=validate.Length(min=1, max=64), load_default=None)
    is_active = fields.Bool(load_default=True)
    churned_at = fields.DateTime(dump_only=True, allow_none=True)
    latest_risk_level = fields.Str(dump_only=True, allow_none=True)
    latest_churn_probability = fields.Float(dump_only=True, allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    locals().update(_profile_fields())


class CustomerUpdateSchema(Schema):
    """Partial update: every field optional."""

    name = fields.Str(validate=validate.Length(min=1, max=120))
    email = fields.Email()
    tenure_months = fields.Int(validate=validate.Range(min=0, max=1200))
    monthly_charges = fields.Float(validate=validate.Range(min=0, max=100000))
    total_charges = fields.Float(validate=validate.Range(min=0))
    contract_type = fields.Str(validate=validate.OneOf(CONTRACT_TYPES))
    plan = fields.Str(validate=validate.Length(min=1, max=64))
    is_active = fields.Bool()

    locals().update(_profile_fields())


class CustomerQuerySchema(PaginationQuerySchema):
    """Query-string params for GET /customers."""

    q = fields.Str(validate=validate.Length(max=100), metadata={"description": "Search name, email or external id"})
    risk_level = fields.Str(validate=validate.OneOf(RISK_LEVELS))
    contract_type = fields.Str(validate=validate.OneOf(CONTRACT_TYPES))
    is_active = fields.Bool()
    sort_by = fields.Str(
        load_default="id",
        validate=validate.OneOf(
            ("id", "name", "email", "tenure_months", "monthly_charges", "total_charges", "created_at", "signup_date")
        ),
    )
    order = fields.Str(load_default="asc", validate=validate.OneOf(("asc", "desc")))


class TransactionSchema(Schema):
    id = fields.Int(dump_only=True)
    customer_id = fields.Int(dump_only=True)
    amount = fields.Float(required=True, validate=validate.Range(min=0))
    type = fields.Str(load_default="charge", validate=validate.OneOf(("charge", "refund", "credit")))
    status = fields.Str(load_default="completed", validate=validate.OneOf(("completed", "pending", "failed")))
    created_at = fields.DateTime(dump_only=True)
