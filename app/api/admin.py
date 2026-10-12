"""Admin-only endpoints: user management and API activity."""

from flask import g
from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.schemas.admin_schema import LogQuerySchema, LogSchema, UserCreateSchema, UserSchema, UserUpdateSchema
from app.schemas.analytics_schema import ActivityStatsSchema, DaysQuerySchema
from app.schemas.common_schema import PaginationQuerySchema, page_envelope, paginated
from app.services.admin_service import AdminService
from app.utils.decorators import role_required

blp = Blueprint("admin", __name__, url_prefix="/admin", description="User management and API activity (admin only)")


@blp.route("/users")
class Users(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin")
    @blp.arguments(PaginationQuerySchema, location="query")
    @blp.response(200, paginated(UserSchema))
    def get(self, args):
        """List users."""
        items, total = AdminService().list_users(args["page"], args["per_page"])
        return page_envelope(items, total, args["page"], args["per_page"])

    @errors(blp, 401, 403, 409)
    @role_required("admin")
    @blp.arguments(UserCreateSchema)
    @blp.response(201, UserSchema)
    def post(self, payload):
        """Create a user with any role."""
        return AdminService().create_user(payload["email"], payload["password"], payload["role"])


@blp.route("/users/<int:user_id>")
class UserDetail(MethodView):
    @errors(blp, 400, 401, 403, 404)
    @role_required("admin")
    @blp.arguments(UserUpdateSchema)
    @blp.response(200, UserSchema)
    def put(self, payload, user_id):
        """Change a user's role / active flag or reset their password (you cannot demote yourself)."""
        return AdminService().update_user(user_id, payload, g.current_user)


@blp.route("/logs")
class Logs(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin")
    @blp.arguments(LogQuerySchema, location="query")
    @blp.response(200, paginated(LogSchema))
    def get(self, args):
        """The per-request audit trail (who called what, status, latency)."""
        page, per_page = args.pop("page"), args.pop("per_page")
        items, total = AdminService().list_logs(page, per_page, **args)
        return page_envelope(items, total, page, per_page)


@blp.route("/activity")
class Activity(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin")
    @blp.arguments(DaysQuerySchema, location="query")
    @blp.response(200, ActivityStatsSchema)
    def get(self, args):
        """Traffic statistics: volume, error rate, latency (avg / p95), top endpoints."""
        return AdminService().activity_stats(args["days"])
