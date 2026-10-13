"""Marshmallow schemas for auth endpoints."""

from marshmallow import Schema, fields, validate


class RegisterSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=8, max=128), load_only=True)
    role = fields.Str(load_default="analyst", validate=validate.OneOf(("admin", "analyst")))


class LoginSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True, load_only=True)


class ChangePasswordSchema(Schema):
    current_password = fields.Str(required=True, load_only=True)
    new_password = fields.Str(required=True, validate=validate.Length(min=8, max=128), load_only=True)


class TokenSchema(Schema):
    access_token = fields.Str(dump_only=True)
    refresh_token = fields.Str(dump_only=True)


class AccessTokenSchema(Schema):
    """Distinct from TokenSchema (no refresh_token) so OpenAPI component names stay unique."""

    access_token = fields.Str(dump_only=True)


class LogoutSchema(Schema):
    refresh_token = fields.Str(load_default=None, load_only=True)


class MeSchema(Schema):
    id = fields.Int()
    email = fields.Str()
    role = fields.Str()
    is_active = fields.Bool()
