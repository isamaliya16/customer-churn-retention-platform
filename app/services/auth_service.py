"""Business logic for registration, login, token refresh and password changes."""

from __future__ import annotations

from datetime import datetime, timezone

from flask_jwt_extended import create_access_token, create_refresh_token, decode_token

from app.extensions import db
from app.models.revoked_token import RevokedToken
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.utils.errors import ConflictAPIError, UnauthorizedAPIError, ValidationAPIError
from app.utils.time import utcnow


class AuthService:
    def __init__(self, user_repo: UserRepository | None = None):
        self.user_repo = user_repo or UserRepository()

    def register(self, email: str, password: str, role: str = "analyst") -> User:
        if self.user_repo.get_by_email(email):
            raise ConflictAPIError(f"A user with email {email!r} already exists.")
        user = User(email=email, role=role)
        user.set_password(password)
        return self.user_repo.add(user)

    def login(self, email: str, password: str) -> dict:
        user = self.user_repo.get_by_email(email)
        if not user or not user.is_active or not user.check_password(password):
            raise UnauthorizedAPIError("Invalid email or password.")
        return self.issue_tokens(user)

    def refresh(self, user_id: int) -> dict:
        user = self.user_repo.get_by_id(user_id)
        if not user or not user.is_active:
            raise UnauthorizedAPIError("User no longer active.")
        return {
            "access_token": create_access_token(identity=str(user.id), additional_claims={"role": user.role}),
        }

    @staticmethod
    def is_revoked(jti: str) -> bool:
        return db.session.query(RevokedToken.id).filter_by(jti=jti).first() is not None

    @staticmethod
    def revoke(jti: str, token_type: str, user_id: int | None, exp: int) -> None:
        """Blacklist one token until it would have expired anyway; opportunistically purge expired rows."""
        if AuthService.is_revoked(jti):
            return
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).replace(tzinfo=None)
        db.session.query(RevokedToken).filter(RevokedToken.expires_at < utcnow()).delete()
        db.session.add(RevokedToken(jti=jti, token_type=token_type, user_id=user_id, expires_at=expires_at))
        db.session.commit()

    def logout(self, claims: dict, refresh_token: str | None = None) -> None:
        """Revoke the presented token and, if supplied, the matching refresh token."""
        user_id = int(claims["sub"]) if str(claims.get("sub", "")).isdigit() else None
        self.revoke(claims["jti"], claims.get("type", "access"), user_id, claims["exp"])
        if refresh_token:
            try:
                other = decode_token(refresh_token)
            except Exception as exc:  # malformed / expired / bad signature
                raise ValidationAPIError("The supplied refresh_token is not valid.") from exc
            if other.get("sub") == claims.get("sub"):  # you can only revoke your own refresh token
                self.revoke(other["jti"], "refresh", user_id, other["exp"])

    def change_password(self, user: User, current_password: str, new_password: str) -> None:
        if not user.check_password(current_password):
            raise UnauthorizedAPIError("Current password is incorrect.")
        if current_password == new_password:
            raise ValidationAPIError("The new password must differ from the current one.")
        user.set_password(new_password)
        self.user_repo.commit()

    @staticmethod
    def issue_tokens(user: User) -> dict:
        claims = {"role": user.role}
        return {
            "access_token": create_access_token(identity=str(user.id), additional_claims=claims),
            "refresh_token": create_refresh_token(identity=str(user.id), additional_claims=claims),
        }
