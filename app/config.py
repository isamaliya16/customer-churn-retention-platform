"""Application configuration classes.

Local development works with ZERO external services: it defaults to a SQLite
file and an in-process cache. Set DATABASE_URL / REDIS_URL in `.env` to use
PostgreSQL / Redis (recommended for production).
"""

from __future__ import annotations

import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CORS = "http://localhost:5000,http://127.0.0.1:5000,http://localhost:8080,http://127.0.0.1:8080"


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class BaseConfig:
    """Shared configuration. Subclassed per environment."""

    SECRET_KEY: str = os.environ.get("SECRET_KEY") or "dev-secret-key"
    DEBUG: bool = False
    TESTING: bool = False

    # --- Database (SQLite by default, PostgreSQL when DATABASE_URL is set) ---
    SQLALCHEMY_DATABASE_URI: str = os.environ.get("DATABASE_URL") or "sqlite:///churn_dev.db"
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {"pool_pre_ping": True}

    # --- Redis / caching (in-process cache when REDIS_URL is not set) ---
    REDIS_URL: str = os.environ.get("REDIS_URL") or ""
    CACHE_TYPE: str = "RedisCache" if REDIS_URL else "SimpleCache"
    CACHE_REDIS_URL: str = REDIS_URL
    CACHE_DEFAULT_TIMEOUT: int = int(os.environ.get("CACHE_DEFAULT_TIMEOUT", 600))

    # --- JWT ---
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY") or "dev-jwt-secret-key-change-me-0123456789"
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(minutes=int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", 60)))
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(days=int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRES_DAYS", 30)))

    # --- Rate limiting (in-memory storage when no Redis) ---
    RATELIMIT_STORAGE_URI: str = os.environ.get("RATELIMIT_STORAGE_URI") or "memory://"
    RATELIMIT_DEFAULT: str = os.environ.get("RATE_LIMIT_DEFAULT") or "300 per minute"
    RATELIMIT_HEADERS_ENABLED: bool = True
    RATE_LIMIT_PREDICT: str = os.environ.get("RATE_LIMIT_PREDICT") or "100 per minute"
    RATE_LIMIT_BATCH: str = os.environ.get("RATE_LIMIT_BATCH") or "10 per minute"
    RATE_LIMIT_LOGIN: str = os.environ.get("RATE_LIMIT_LOGIN") or "20 per minute"

    # Public self-service sign-up is OFF by default: a new account can read every customer's personal data,
    # so accounts are created by an admin (POST /auth/register with an admin token or `flask create-admin`).
    ALLOW_PUBLIC_REGISTRATION: bool = os.environ.get("ALLOW_PUBLIC_REGISTRATION", "").lower() in ("1", "true", "yes")

    TRUST_PROXY: bool = os.environ.get("TRUST_PROXY", "").lower() in ("1", "true", "yes")

    # --- CORS ---
    CORS_ORIGINS: list[str] = _split_csv(os.environ.get("CORS_ORIGINS") or DEFAULT_CORS)

    # --- ML ---
    MODEL_DIR: str = os.environ.get("MODEL_DIR", "app/ml/artifacts")

    # --- API docs (flask-smorest) ---
    API_TITLE: str = "Churn & Retention Platform API"
    API_VERSION: str = "v1"
    OPENAPI_VERSION: str = "3.0.3"
    OPENAPI_URL_PREFIX: str = "/"
    OPENAPI_JSON_PATH: str = "openapi.json"
    OPENAPI_SWAGGER_UI_PATH: str = "/docs"
    OPENAPI_SWAGGER_UI_URL: str = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get("TEST_DATABASE_URL") or "sqlite://"
    RATELIMIT_ENABLED = False
    RATELIMIT_STORAGE_URI = "memory://"
    CACHE_TYPE = "SimpleCache"


class ProductionConfig(BaseConfig):
    DEBUG = False


config_by_name: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
