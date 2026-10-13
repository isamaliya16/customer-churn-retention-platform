"""Shared schemas: error envelope, pagination wrapper factory."""

from marshmallow import Schema, fields, validate


class ErrorDetailSchema(Schema):
    code = fields.Str(metadata={"example": "not_found"})
    message = fields.Str(metadata={"example": "Customer 999 not found."})
    details = fields.Raw(allow_none=True)


class ErrorEnvelopeSchema(Schema):
    """The one error envelope used by every endpoint."""

    error = fields.Nested(ErrorDetailSchema)


class PaginationQuerySchema(Schema):
    page = fields.Int(load_default=1, validate=validate.Range(min=1))
    per_page = fields.Int(load_default=20, validate=validate.Range(min=1, max=100))


class PageMetaSchema(Schema):
    page = fields.Int()
    per_page = fields.Int()
    total = fields.Int()
    total_pages = fields.Int()


def paginated(item_schema: type[Schema]) -> type[Schema]:
    """Build a `{items, page, per_page, total, total_pages}` response schema for `item_schema`."""
    return type(
        f"Paginated{item_schema.__name__.replace('Schema', '')}",
        (PageMetaSchema,),
        {"items": fields.List(fields.Nested(item_schema))},
    )


def page_envelope(items, total: int, page: int, per_page: int, schema: Schema | None = None) -> dict:
    """Standard paginated payload; serialises `items` with `schema` when given."""
    return {
        "items": schema.dump(items, many=True) if schema is not None else list(items),
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page if per_page else 0,
    }
