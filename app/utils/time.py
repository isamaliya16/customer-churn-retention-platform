"""Time helpers."""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC 'now' (replacement for the deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
