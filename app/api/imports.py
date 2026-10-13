"""Bulk CSV import: upload a customer file, poll its progress, download a template."""

from flask import Response, g, request
from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.schemas.import_schema import ImportJobSchema, ImportUploadSchema
from app.services.import_service import ImportService, template_csv
from app.utils.decorators import role_required
from app.utils.errors import ValidationAPIError

blp = Blueprint("imports", __name__, url_prefix="/imports", description="Bulk CSV import of customers (100k+ rows)")


@blp.route("")
class Imports(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.response(200, ImportJobSchema(many=True))
    def get(self):
        """The latest import jobs, newest first."""
        return ImportService.list_jobs()

    @errors(blp, 400, 401, 403)
    @role_required("admin")
    @blp.arguments(ImportUploadSchema, location="files", description="multipart/form-data with a `file` field")
    @blp.response(202, ImportJobSchema)
    def post(self, _files):
        """Upload a CSV (multipart field `file`; form fields `mode` = skip|update, `score_after` = true|false).

        Processing runs in the background - poll `GET /imports/<id>` for progress."""
        upload = request.files.get("file")
        if upload is None:
            raise ValidationAPIError("Send the CSV as multipart field 'file'.", code="missing_file")
        mode = request.form.get("mode", "skip")
        score_after = request.form.get("score_after", "true").lower() in ("1", "true", "yes", "on")
        return ImportService().submit(
            upload.read(), upload.filename or "upload.csv", mode, score_after, g.current_user.id
        )


@blp.route("/template.csv")
class ImportTemplate(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    def get(self):
        """A CSV template with every supported column and one example row."""
        return Response(
            template_csv(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=customers_template.csv"},
        )


@blp.route("/<int:job_id>")
class ImportDetail(MethodView):
    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.response(200, ImportJobSchema)
    def get(self, job_id):
        """Status, counters and the first row-level problems of one import."""
        return ImportService.get_or_404(job_id)
