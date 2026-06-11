"""Null auth/authz provider — development and testing only.

All authentication succeeds. All permission checks return True.
DO NOT use in production. Set APG_AUTH_PROVIDER=null only in dev/test.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..protocols import (
    AuthResult, AuthUser, AuthzProvider, AuthenticationError,
    MFASetup, TokenPair, TokenPayload, UserList,
)

_log = logging.getLogger(__name__)

_NULL_TOKEN = "null-provider-dev-token"
_NULL_REFRESH = "null-provider-dev-refresh"


class NullAuthProvider:
    """Pass-through auth provider for development. Always succeeds."""

    provider_name = "null"

    def __init__(self, dev_user_id: str = "dev-user", dev_email: str = "dev@localhost") -> None:
        _log.warning("NullAuthProvider active — ALL authentication succeeds. NOT for production.")
        self._user_id = dev_user_id
        self._email = dev_email
        self._users: dict[str, AuthUser] = {
            dev_user_id: AuthUser(
                id=dev_user_id, email=dev_email, username="dev",
                first_name="Dev", last_name="User",
                is_active=True, is_email_verified=True, roles=["admin"],
            )
        }

    def _make_tokens(self) -> TokenPair:
        return TokenPair(
            access_token=_NULL_TOKEN,
            refresh_token=_NULL_REFRESH,
            expires_in=86400,
        )

    def _make_user(self) -> AuthUser:
        return self._users[self._user_id]

    async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
        user = self._make_user()
        return AuthResult(user=user, tokens=self._make_tokens())

    async def validate_token(self, token: str) -> TokenPayload:
        return TokenPayload(
            user_id=self._user_id,
            email=self._email,
            roles=["admin"],
            permissions=["*"],
            tenant_id="default",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )

    async def refresh_token(self, refresh_token: str) -> TokenPair:
        return self._make_tokens()

    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        pass

    async def create_user(self, user_data: dict[str, Any]) -> AuthUser:
        from uuid_extensions import uuid7str
        uid = uuid7str()
        user = AuthUser(
            id=uid,
            email=user_data.get("email", f"{uid}@dev.local"),
            username=user_data.get("username", uid),
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            is_active=True,
            is_email_verified=True,
            roles=user_data.get("roles", []),
        )
        self._users[uid] = user
        return user

    async def get_user(self, user_id: str) -> AuthUser:
        user = self._users.get(user_id)
        if user is None:
            raise KeyError(f"User {user_id!r} not found")
        return user

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> AuthUser:
        user = await self.get_user(user_id)
        for k, v in updates.items():
            if hasattr(user, k):
                setattr(user, k, v)
        return user

    async def delete_user(self, user_id: str) -> None:
        self._users.pop(user_id, None)

    async def list_users(self, search: str | None = None, limit: int = 50, page: int = 1) -> UserList:
        users = list(self._users.values())
        if search:
            q = search.lower()
            users = [u for u in users if q in u.email.lower() or q in u.username.lower()]
        return UserList(users=users[:limit], total=len(users), page=page, limit=limit)

    async def send_password_reset(self, email: str) -> None:
        _log.debug("NullAuthProvider: password reset email for %s (no-op)", email)

    async def reset_password(self, token: str, new_password: str) -> None:
        pass

    async def send_magic_link(self, email: str, redirect_url: str) -> None:
        _log.debug("NullAuthProvider: magic link for %s -> %s (no-op)", email, redirect_url)

    async def verify_magic_link(self, token: str) -> AuthResult:
        return AuthResult(user=self._make_user(), tokens=self._make_tokens())

    async def get_oauth_authorization_url(
        self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
    ) -> str:
        return f"{redirect_uri}?code=null-oauth-code&state={state}"

    async def exchange_oauth_code(
        self, code: str, state: str, redirect_uri: str, provider: str
    ) -> AuthResult:
        return AuthResult(user=self._make_user(), tokens=self._make_tokens())

    async def setup_mfa(self, user_id: str, mfa_type: str) -> MFASetup:
        return MFASetup(mfa_type=mfa_type, secret="NULLSECRET", session_token="null-mfa")

    async def verify_mfa(self, user_id: str, code: str, session_token: str) -> AuthResult:
        return AuthResult(user=self._make_user(), tokens=self._make_tokens())

    async def disable_mfa(self, user_id: str, mfa_type: str) -> None:
        pass

    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        return [{"session_id": "null-session", "created_at": datetime.now(timezone.utc).isoformat()}]

    async def revoke_session(self, session_id: str) -> None:
        pass

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "provider": "null", "warning": "development only"}


class NullAuthzProvider:
    """Pass-through authz provider — always allows. Development only."""

    provider_name = "null"

    def __init__(self) -> None:
        _log.warning("NullAuthzProvider active — ALL permission checks pass. NOT for production.")
        self._roles: dict[str, list[str]] = {}
        self._role_registry: dict[str, dict[str, Any]] = {
            "admin": {"role": "admin", "permissions": ["*"]},
            "user": {"role": "user", "permissions": ["read"]},
        }

    async def check_permission(self, user_id: str, permission: str, tenant_id: str = "default",
                               resource_id: str | None = None, resource_type: str | None = None,
                               context: dict[str, Any] | None = None) -> bool:
        return True

    async def check_resource_access(self, user_id: str, resource_type: str, resource_id: str,
                                    action: str, tenant_id: str = "default") -> bool:
        return True

    async def get_user_roles(self, user_id: str, tenant_id: str = "default") -> list[str]:
        return self._roles.get(f"{tenant_id}:{user_id}", ["admin"])

    async def assign_role(self, user_id: str, role: str, tenant_id: str = "default",
                         granted_by: str = "system") -> None:
        key = f"{tenant_id}:{user_id}"
        self._roles.setdefault(key, [])
        if role not in self._roles[key]:
            self._roles[key].append(role)

    async def revoke_role(self, user_id: str, role: str, tenant_id: str = "default",
                          revoked_by: str = "system") -> None:
        key = f"{tenant_id}:{user_id}"
        if key in self._roles:
            self._roles[key] = [r for r in self._roles[key] if r != role]

    async def get_role_permissions(self, role: str, tenant_id: str = "default") -> list[str]:
        return ["*"]

    async def create_role(self, role: str, permissions: list[str], tenant_id: str = "default",
                          description: str = "") -> dict[str, Any]:
        entry = {"role": role, "permissions": permissions, "tenant_id": tenant_id}
        self._role_registry[role] = entry
        return entry

    async def delete_role(self, role: str, tenant_id: str = "default") -> None:
        pass

    async def list_roles(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        return list(self._role_registry.values())

    async def write_relationship(self, resource_type: str, resource_id: str, relation: str,
                                 subject_type: str, subject_id: str) -> None:
        pass

    async def delete_relationship(self, resource_type: str, resource_id: str, relation: str,
                                  subject_type: str, subject_id: str) -> None:
        pass

    async def list_accessible_resources(self, user_id: str, resource_type: str, action: str,
                                        tenant_id: str = "default") -> list[str]:
        return ["*"]

    async def bulk_check_permissions(self, user_id: str, checks: list[dict[str, Any]],
                                     tenant_id: str = "default") -> dict[str, bool]:
        return {c.get("permission", ""): True for c in checks}

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "provider": "null", "warning": "development only"}
