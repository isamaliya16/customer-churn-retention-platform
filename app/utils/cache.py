"""Cache invalidation helper."""

from app.extensions import cache


def invalidate_analytics_cache() -> None:
    """Drop every cached analytics response (call after customer/prediction writes).

    Endpoints that vary by query string (trends, high-risk) have per-URL keys, so a
    full clear is the simplest correct invalidation.
    """
    cache.clear()
