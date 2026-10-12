"""Flask extension instances.

Instantiated here (unbound) and attached to the app in `create_app()` via
`init_app()`, which is the standard pattern for avoiding circular imports.
"""

from flask_caching import Cache
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman


class Base:
    """Placeholder to satisfy type checkers; SQLAlchemy() supplies the real
    declarative base via `db.Model`."""


db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
cache = Cache()
cors = CORS()
talisman = Talisman()
limiter = Limiter(key_func=get_remote_address)
