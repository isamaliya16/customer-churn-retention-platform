"""Schemas for the bulk CSV import endpoints."""

from marshmallow import Schema, fields


class ImportJobSchema(Schema):
    id = fields.Int()
    filename = fields.Str()
    status = fields.Str(metadata={"description": "pending | running | completed | failed"})
    phase = fields.Str(metadata={"description": "queued | importing | scoring | done"})
    mode = fields.Str()
    score_after = fields.Bool()
    total_rows = fields.Int()
    processed_rows = fields.Int()
    created_count = fields.Int()
    updated_count = fields.Int()
    skipped_count = fields.Int()
    rejected_count = fields.Int()
    scored_count = fields.Int()
    errors = fields.List(fields.Raw())
    warnings = fields.List(fields.Raw())
    message = fields.Str(allow_none=True)
    created_at = fields.DateTime()
    started_at = fields.DateTime(allow_none=True)
    finished_at = fields.DateTime(allow_none=True)


class ImportUploadSchema(Schema):
    file = fields.Raw(required=True, metadata={"type": "string", "format": "binary"})
