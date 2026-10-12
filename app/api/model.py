"""GET /model/info - the deployed model's metrics, importance and decision rules."""

from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.schemas.analytics_schema import ModelInfoSchema
from app.services.model_service import ModelService
from app.utils.decorators import role_required

blp = Blueprint("model", __name__, url_prefix="/model", description="Deployed model information")


@blp.route("/info")
class ModelInfo(MethodView):
    @errors(blp, 400, 401, 403)
    @role_required("admin", "analyst")
    @blp.response(200, ModelInfoSchema)
    def get(self):
        """Version, metrics, ROC curves, feature importance, risk thresholds and recommendation rules."""
        return ModelService().info()
