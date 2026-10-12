"""Consistent JSON error envelope used across the whole API.

Every error response has the shape:
    {"error": {"code": "...", "message": "...", "details": ...}}
"""

from __future__ import annotations

from typing import Any


class APIError(Exception):
    """Base class for all handled application errors.

    Raise this (or a subclass) from services/repositories/routes; the global
    error handler registered in `app/__init__.py` converts it into the
    standard envelope with the right HTTP status.
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, details: Any = None, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.details = details
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code

    def to_envelope(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ValidationAPIError(APIError):
    status_code = 400
    code = "validation_error"


class NotFoundAPIError(APIError):
    status_code = 404
    code = "not_found"


class ConflictAPIError(APIError):
    status_code = 409
    code = "conflict"


class UnauthorizedAPIError(APIError):
    status_code = 401
    code = "unauthorized"


class ForbiddenAPIError(APIError):
    status_code = 403
    code = "forbidden"


class RateLimitedAPIError(APIError):
    status_code = 429
    code = "rate_limited"


def error_envelope(code: str, message: str, details: Any = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}
