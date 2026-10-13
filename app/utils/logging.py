"""Structured JSON logging to stdout, suitable for log aggregators.

Distinct from the per-request `APILog` DB audit trail - this is
process-level operational logging (startup, errors, warnings).
"""

from __future__ import annotations

import json
import logging
import sys


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party loggers unless something's actually wrong.
    for noisy in ("werkzeug", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
