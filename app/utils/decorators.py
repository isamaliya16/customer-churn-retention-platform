"""Authorization decorators.

`@role_required("admin", "analyst")` enforces a valid JWT AND re-checks the user in the database on every
request, so deactivating a user or changing their role takes effect immediately (not at token expiry).
The authenticated user is available as `flask.g.current_user`.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import g
from flask_jwt_extended import get_jwt_identity, jwt_required

from app.extensions import db
from app.models.user import User
from app.utils.errors import ForbiddenAPIError, UnauthorizedAPIError


def role_required(*roles: str) -> Callable:
    """Restrict an endpoint to one or more roles (implies a valid JWT)."""
    allowed = tuple(roles)

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            try:
                user = db.session.get(User, int(get_jwt_identity()))
            except (TypeError, ValueError):
                user = None
            if user is None or not user.is_active:
                raise UnauthorizedAPIError("This account is disabled or no longer exists.")
            if user.role not in allowed:
                raise ForbiddenAPIError(
                    f"Role '{user.role}' is not permitted to access this resource.",
                    details={"required_roles": list(allowed)},
                )
            g.current_user = user
            return fn(*args, **kwargs)

        return wrapper

    return decorator
