"""CSV export helpers (streamed row by row)."""

from __future__ import annotations

import csv
import io
from typing import Iterable, Iterator

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_cell(value):
    """Neutralise spreadsheet formula injection: text starting with = + - @ is prefixed with a quote so Excel/Sheets
    show it as text instead of evaluating it. Numbers (incl. negatives) are left untouched."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def rows_to_csv(header: list[str], rows: Iterable[list]) -> Iterator[str]:
    """Yield CSV text chunks: header first, then each row."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([safe_cell(h) for h in header])
    yield buffer.getvalue()
    for row in rows:
        buffer.seek(0)
        buffer.truncate(0)
        writer.writerow([safe_cell(cell) for cell in row])
        yield buffer.getvalue()
