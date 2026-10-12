"""Small helpers shared by the API modules."""

from __future__ import annotations

from typing import Callable

from app.schemas.common_schema import ErrorEnvelopeSchema

ERROR_DESCRIPTIONS = {
    400: "Bad request",
    401: "Missing, invalid or expired token / bad credentials",
    403: "Authenticated but the role is not permitted",
    404: "Resource not found",
    409: "Conflict (duplicate resource)",
    429: "Rate limit exceeded (see Retry-After header)",
}


def errors(blp, *codes: int) -> Callable:
    """Document the standard error envelope for the given status codes in OpenAPI."""

    def decorator(fn: Callable) -> Callable:
        for code in codes:
            fn = blp.alt_response(
                code, schema=ErrorEnvelopeSchema, description=ERROR_DESCRIPTIONS[code], success=False
            )(fn)
        return fn

    return decorator
