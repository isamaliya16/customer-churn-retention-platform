"""Schemas for retention actions."""

from marshmallow import Schema, fields, validate

from app.constants import ACTION_OUTCOMES, ACTION_STATUSES
from app.schemas.common_schema import PaginationQuerySchema


class ActionCreateSchema(Schema):
    customer_id = fields.Int(required=True, validate=validate.Range(min=1))
    prediction_id = fields.Int(validate=validate.Range(min=1), allow_none=True, load_default=None)
    action = fields.Str(required=True, validate=validate.Length(min=1, max=255))
    notes = fields.Str(validate=validate.Length(max=2000), allow_none=True, load_default=None)
    due_date = fields.Date(allow_none=True, load_default=None)
    offer_cost = fields.Float(validate=validate.Range(min=0), allow_none=True, load_default=None)


class ActionBulkCreateSchema(Schema):
    customer_ids = fields.List(
        fields.Int(validate=validate.Range(min=1)), required=True, validate=validate.Length(min=1, max=5000)
    )
    action = fields.Str(validate=validate.Length(min=1, max=255), load_default=None, allow_none=True)
    due_date = fields.Date(allow_none=True, load_default=None)
    offer_cost = fields.Float(validate=validate.Range(min=0), allow_none=True, load_default=None)
    notes = fields.Str(validate=validate.Length(max=2000), allow_none=True, load_default=None)


class ActionBulkResultSchema(Schema):
    requested = fields.Int()
    created = fields.Int()
    skipped = fields.Int()


class ActionUpdateSchema(Schema):
    action = fields.Str(validate=validate.Length(min=1, max=255))
    status = fields.Str(validate=validate.OneOf(ACTION_STATUSES))
    notes = fields.Str(validate=validate.Length(max=2000), allow_none=True)
    due_date = fields.Date(allow_none=True)
    offer_cost = fields.Float(validate=validate.Range(min=0), allow_none=True)
    outcome = fields.Str(validate=validate.OneOf(ACTION_OUTCOMES), allow_none=True)


class ActionSchema(Schema):
    id = fields.Int(dump_only=True)
    customer_id = fields.Int(dump_only=True)
    customer_name = fields.Str(dump_only=True, allow_none=True)
    prediction_id = fields.Int(dump_only=True, allow_none=True)
    action = fields.Str(dump_only=True)
    status = fields.Str(dump_only=True)
    notes = fields.Str(dump_only=True, allow_none=True)
    due_date = fields.Date(dump_only=True, allow_none=True)
    offer_cost = fields.Float(dump_only=True, allow_none=True)
    outcome = fields.Str(dump_only=True, allow_none=True)
    created_by = fields.Int(dump_only=True, allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    completed_at = fields.DateTime(dump_only=True, allow_none=True)


class ActionQuerySchema(PaginationQuerySchema):
    status = fields.Str(validate=validate.OneOf(ACTION_STATUSES))
    customer_id = fields.Int(validate=validate.Range(min=1))
