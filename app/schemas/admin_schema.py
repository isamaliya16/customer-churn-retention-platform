"""Schemas for admin endpoints (users, activity log)."""

from marshmallow import Schema, fields, validate

from app.constants import ROLES
from app.schemas.common_schema import PaginationQuerySchema


class UserSchema(Schema):
    id = fields.Int(dump_only=True)
    email = fields.Email(dump_only=True)
    role = fields.Str(dump_only=True)
    is_active = fields.Bool(dump_only=True)
    created_at = fields.DateTime(dump_only=True)


class UserCreateSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=8, max=128), load_only=True)
    role = fields.Str(load_default="analyst", validate=validate.OneOf(ROLES))


class UserUpdateSchema(Schema):
    role = fields.Str(validate=validate.OneOf(ROLES))
    is_active = fields.Bool()
    password = fields.Str(validate=validate.Length(min=8, max=128), load_only=True)


class LogSchema(Schema):
    id = fields.Int(dump_only=True)
    user_id = fields.Int(dump_only=True, allow_none=True)
    user_email = fields.Str(dump_only=True, allow_none=True)
    endpoint = fields.Str(dump_only=True)
    method = fields.Str(dump_only=True)
    status_code = fields.Int(dump_only=True)
    latency_ms = fields.Int(dump_only=True)
    ip_address = fields.Str(dump_only=True, allow_none=True)
    created_at = fields.DateTime(dump_only=True)


class LogQuerySchema(PaginationQuerySchema):
    method = fields.Str(validate=validate.OneOf(("GET", "POST", "PUT", "PATCH", "DELETE")))
    status_class = fields.Int(validate=validate.OneOf((2, 3, 4, 5)), metadata={"description": "2 = 2xx, 4 = 4xx ..."})
    endpoint = fields.Str(validate=validate.Length(max=100), metadata={"description": "Substring match"})
    user_id = fields.Int(validate=validate.Range(min=1))
