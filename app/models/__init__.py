"""Import every model so Alembic's `--autogenerate` and `create_all()` see them."""

from app.models.api_log import APILog  # noqa: F401
from app.models.customer import Customer  # noqa: F401
from app.models.import_job import ImportJob  # noqa: F401
from app.models.prediction import Prediction  # noqa: F401
from app.models.retention_action import RetentionAction  # noqa: F401
from app.models.revoked_token import RevokedToken  # noqa: F401
from app.models.transaction import Transaction  # noqa: F401
from app.models.user import User  # noqa: F401
