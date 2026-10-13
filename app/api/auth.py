"""POST /auth/register, /auth/login, /auth/refresh, /auth/logout, /auth/change-password and GET /auth/me."""

from flask import current_app, g
from flask.views import MethodView
from flask_jwt_extended import get_jwt, get_jwt_identity, jwt_required, verify_jwt_in_request
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.extensions import db, limiter
from app.models.user import User
from app.schemas.auth_schema import (
    AccessTokenSchema,
    ChangePasswordSchema,
    LoginSchema,
    LogoutSchema,
    MeSchema,
    RegisterSchema,
    TokenSchema,
)
from app.services.auth_service import AuthService
from app.utils.decorators import role_required
from app.utils.errors import ForbiddenAPIError

blp = Blueprint(
    "auth", __name__, url_prefix="/auth", description="User creation, login, token refresh, logout, profile"
)


def _caller_is_admin() -> bool:
    """True only for a valid token whose user is, in the database right now, an active admin."""
    verify_jwt_in_request(optional=True)
    identity = get_jwt_identity()
    if identity is None:
        return False
    try:
        user = db.session.get(User, int(identity))
    except (TypeError, ValueError):
        return False
    return bool(user and user.is_active and user.role == "admin")


@blp.route("/register")
class Register(MethodView):
    @errors(blp, 403, 409)
    @blp.arguments(RegisterSchema)
    @blp.response(201, TokenSchema)
    def post(self, payload):
        """Create a user. Admin-only by default; self-service analyst sign-up only if ALLOW_PUBLIC_REGISTRATION=true."""
        role = payload.get("role", "analyst")
        public_allowed = current_app.config["ALLOW_PUBLIC_REGISTRATION"] and role != "admin"
        if not public_allowed and not _caller_is_admin():
            raise ForbiddenAPIError(
                "Only an admin can create users."
                if role == "admin" or not current_app.config["ALLOW_PUBLIC_REGISTRATION"]
                else "Registration is restricted."
            )
        service = AuthService()
        return service.issue_tokens(service.register(payload["email"], payload["password"], role))


@blp.route("/login")
class Login(MethodView):
    @errors(blp, 401, 429)
    @limiter.limit(lambda: current_app.config["RATE_LIMIT_LOGIN"])
    @blp.arguments(LoginSchema)
    @blp.response(200, TokenSchema)
    def post(self, payload):
        """Exchange email/password for an access + refresh token pair."""
        return AuthService().login(payload["email"], payload["password"])


@blp.route("/refresh")
class Refresh(MethodView):
    @errors(blp, 401)
    @jwt_required(refresh=True)
    @blp.response(200, AccessTokenSchema)
    def post(self):
        """Exchange a valid refresh token for a new access token."""
        return AuthService().refresh(int(get_jwt_identity()))


@blp.route("/logout")
class Logout(MethodView):
    @errors(blp, 401)
    @jwt_required(verify_type=False)
    @blp.arguments(LogoutSchema)
    @blp.response(204)
    def post(self, payload):
        """Revoke the presented token (and optionally the matching `refresh_token`) so it can no longer be used."""
        AuthService().logout(get_jwt(), payload.get("refresh_token"))


@blp.route("/me")
class Me(MethodView):
    @errors(blp, 401)
    @role_required("admin", "analyst")
    @blp.response(200, MeSchema)
    def get(self):
        """The currently authenticated user."""
        return g.current_user


@blp.route("/change-password")
class ChangePassword(MethodView):
    @errors(blp, 401)
    @role_required("admin", "analyst")
    @blp.arguments(ChangePasswordSchema)
    @blp.response(204)
    def post(self, payload):
        """Change your own password."""
        AuthService().change_password(g.current_user, payload["current_password"], payload["new_password"])
        return None
