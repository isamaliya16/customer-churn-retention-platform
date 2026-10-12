"""Retention actions: follow-up tasks raised for at-risk customers."""

from flask import g
from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.schemas.action_schema import (
    ActionBulkCreateSchema,
    ActionBulkResultSchema,
    ActionCreateSchema,
    ActionQuerySchema,
    ActionSchema,
    ActionUpdateSchema,
)
from app.schemas.common_schema import page_envelope, paginated
from app.services.action_service import ActionService
from app.utils.cache import invalidate_analytics_cache
from app.utils.decorators import role_required

blp = Blueprint("actions", __name__, url_prefix="/actions", description="Retention actions (follow-up tasks)")


@blp.route("")
class Actions(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(ActionQuerySchema, location="query")
    @blp.response(200, paginated(ActionSchema))
    def get(self, args):
        """List retention actions, filterable by status / customer, newest first."""
        page, per_page = args.pop("page"), args.pop("per_page")
        items, total = ActionService().list_actions(page, per_page, **args)
        return page_envelope(items, total, page, per_page)

    @errors(blp, 400, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.arguments(ActionCreateSchema)
    @blp.response(201, ActionSchema)
    def post(self, payload):
        """Create an action (e.g. from a prediction's recommended action)."""
        action = ActionService().create(payload, g.current_user)
        invalidate_analytics_cache()
        return action


@blp.route("/bulk")
class ActionsBulk(MethodView):
    @errors(blp, 400, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(ActionBulkCreateSchema)
    @blp.response(201, ActionBulkResultSchema)
    def post(self, payload):
        """Raise one action per customer (e.g. a whole campaign list). Customers with an open action are skipped."""
        result = ActionService().create_bulk(payload, g.current_user)
        invalidate_analytics_cache()
        return result


@blp.route("/<int:action_id>")
class ActionDetail(MethodView):
    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.response(200, ActionSchema)
    def get(self, action_id):
        """Fetch one action."""
        return ActionService().get_or_404(action_id)

    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.arguments(ActionUpdateSchema)
    @blp.response(200, ActionSchema)
    def put(self, payload, action_id):
        """Update the status / notes / text of an action (done + dismissed stamp completed_at)."""
        action = ActionService().update(action_id, payload)
        invalidate_analytics_cache()
        return action

    @errors(blp, 401, 403, 404)
    @role_required("admin")
    @blp.response(204)
    def delete(self, action_id):
        """Delete an action. Admin only."""
        ActionService().delete(action_id)
        invalidate_analytics_cache()
        return None
