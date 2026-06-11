"""Clerk authentication provider.

Clerk is a hosted auth-as-a-service platform with magic links, passkeys,
social OAuth, MFA, and a developer-friendly API.

Config:
    APG_CLERK_SECRET_KEY        sk_live_... or sk_test_...
    APG_CLERK_PUBLISHABLE_KEY   pk_live_... or pk_test_...
    APG_CLERK_INSTANCE_ID       (optional, from Dashboard)

Docs: https://clerk.com/docs/reference/backend-api
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from capabilities.common.reliability import BoundedCache
from capabilities.common.reliability.circuit_breaker import get_circuit_breaker
from capabilities.common.reliability.timeout import with_timeout
from ..protocols import (
    AuthResult, AuthUser, AuthenticationError, MFASetup,
    ProviderNotImplementedError, TokenPair, TokenPayload, UserList,
)

_log = logging.getLogger(__name__)
_CLERK_API = "https://api.clerk.com/v1"



import hashlib as _hashlib

def _cache_key(token: str) -> str:
    """Blake2b hash of the full token — collision-resistant cache key."""
    return _hashlib.blake2b(token.encode(), digest_size=32).hexdigest()

class ClerkAuthProvider:
    """Clerk authentication provider.

    Uses Clerk Backend API for server-side operations (user management, session
    validation). The Clerk Frontend SDK handles the browser-side login flow.
    """

    provider_name = "clerk"

    def __init__(
        self,
        secret_key: str | None = None,
        publishable_key: str | None = None,
    ) -> None:
        self._secret = secret_key or os.environ.get("APG_CLERK_SECRET_KEY", "")
        self._publishable = publishable_key or os.environ.get("APG_CLERK_PUBLISHABLE_KEY", "")
        self._token_cache = BoundedCache(max_size=5000)
        self._cb = get_circuit_breaker("clerk_auth", failure_threshold=5, reset_timeout=60.0)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._secret}",
            "Content-Type": "application/json",
        }

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=10.0)

    def _parse_user(self, u: dict[str, Any]) -> AuthUser:
        emails = u.get("email_addresses", [])
        primary_email = next(
            (e["email_address"] for e in emails if e["id"] == u.get("primary_email_address_id")),
            emails[0]["email_address"] if emails else "",
        )
        return AuthUser(
            id=u.get("id", ""),
            email=primary_email,
            username=u.get("username") or "",
            first_name=u.get("first_name") or "",
            last_name=u.get("last_name") or "",
            is_active=not u.get("banned", False),
            is_email_verified=any(
                e.get("verification", {}).get("status") == "verified"
                for e in emails
            ),
            roles=u.get("public_metadata", {}).get("roles", []),
            metadata=u.get("public_metadata", {}),
            created_at=datetime.fromtimestamp(u["created_at"] / 1000, tz=timezone.utc)
                if u.get("created_at") else None,
            mfa_enabled=bool(u.get("two_factor_enabled")),
        )

    @with_timeout(15.0)
    async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
        """Authenticate via Clerk.

        For username/password: Clerk doesn't expose this directly via Backend API —
        authentication happens via the Frontend SDK or sign-in tokens.
        We support token-based authentication here (validate a Clerk session token).
        """
        if token := credentials.get("token") or credentials.get("session_token"):
            return AuthResult(
                user=await self._get_user_from_token(token),
                tokens=TokenPair(access_token=token, refresh_token="", expires_in=3600),
            )
        raise AuthenticationError(
            "Clerk authentication requires a session token from the Frontend SDK. "
            "Use credentials={'token': clerk_session_token}",
            "clerk_requires_frontend",
        )

    async def _get_user_from_token(self, token: str) -> AuthUser:
        """Validate a Clerk session JWT and return the user."""
        payload = await self.validate_token(token)
        return await self.get_user(payload.user_id)

    @with_timeout(5.0)
    async def validate_token(self, token: str) -> TokenPayload:
        """Verify a Clerk session token via the sessions API."""
        cached = self._token_cache.get(_cache_key(token))
        if cached:
            return cached

        await self._cb._before_call()
        try:
            async with self._http() as client:
                # Clerk v2: verify session token
                resp = await client.post(
                    f"{_CLERK_API}/sessions/verify",
                    headers=self._headers(),
                    json={"token": token},
                )
                if resp.status_code in (401, 403, 404):
                    raise AuthenticationError("Invalid or expired Clerk session token", "token_invalid")
                resp.raise_for_status()
                session = resp.json()

            user_id = session.get("user_id", "")
            expires_at = None
            if exp := session.get("expire_at"):
                expires_at = datetime.fromtimestamp(exp / 1000, tz=timezone.utc)

            payload = TokenPayload(
                user_id=user_id,
                email=session.get("client_id", ""),
                roles=session.get("public_user_data", {}).get("metadata", {}).get("roles", []),
                tenant_id=session.get("public_user_data", {}).get("metadata", {}).get("tenant_id", "default"),
                expires_at=expires_at,
                extra=session,
            )
            ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds()) if expires_at else 300
            self._token_cache.set(_cache_key(token), payload, ttl=min(ttl, 300))
            await self._cb._on_success()
            return payload
        except AuthenticationError:
            raise
        except Exception as exc:
            await self._cb._on_failure(exc)
            raise AuthenticationError(f"Clerk token validation failed: {exc}") from exc

    @with_timeout(10.0)
    async def refresh_token(self, refresh_token: str) -> TokenPair:
        raise ProviderNotImplementedError(
            "Clerk manages token refresh automatically via the Frontend SDK. "
            "There is no server-side token refresh endpoint."
        )

    @with_timeout(10.0)
    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        """Revoke the Clerk session."""
        self._token_cache.delete(_cache_key(token))
        try:
            payload = await self.validate_token(token)
            session_id = payload.extra.get("id", "")
            if session_id:
                async with self._http() as client:
                    await client.delete(
                        f"{_CLERK_API}/sessions/{session_id}/revoke",
                        headers=self._headers(),
                    )
        except Exception as exc:
            _log.debug("Suppressed %s: %s", type(exc).__name__, exc)

    @with_timeout(10.0)
    async def create_user(self, user_data: dict[str, Any]) -> AuthUser:
        await self._cb._before_call()
        try:
            payload: dict[str, Any] = {
                "email_address": [user_data.get("email", "")],
                "first_name": user_data.get("first_name", ""),
                "last_name": user_data.get("last_name", ""),
                "public_metadata": {"roles": user_data.get("roles", [])},
            }
            if username := user_data.get("username"):
                payload["username"] = username
            if password := user_data.get("password"):
                payload["password"] = password
            async with self._http() as client:
                resp = await client.post(f"{_CLERK_API}/users", json=payload, headers=self._headers())
                resp.raise_for_status()
            await self._cb._on_success()
            return self._parse_user(resp.json())
        except Exception as exc:
            await self._cb._on_failure(exc)
            raise

    @with_timeout(5.0)
    async def get_user(self, user_id: str) -> AuthUser:
        async with self._http() as client:
            resp = await client.get(f"{_CLERK_API}/users/{user_id}", headers=self._headers())
            resp.raise_for_status()
            return self._parse_user(resp.json())

    @with_timeout(10.0)
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> AuthUser:
        payload: dict[str, Any] = {}
        if "first_name" in updates:
            payload["first_name"] = updates["first_name"]
        if "last_name" in updates:
            payload["last_name"] = updates["last_name"]
        if "roles" in updates:
            payload["public_metadata"] = {"roles": updates["roles"]}
        async with self._http() as client:
            resp = await client.patch(
                f"{_CLERK_API}/users/{user_id}", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            return self._parse_user(resp.json())

    @with_timeout(10.0)
    async def delete_user(self, user_id: str) -> None:
        async with self._http() as client:
            resp = await client.delete(f"{_CLERK_API}/users/{user_id}", headers=self._headers())
            resp.raise_for_status()

    @with_timeout(15.0)
    async def list_users(self, search: str | None = None, limit: int = 50, page: int = 1) -> UserList:
        params: dict[str, Any] = {"limit": limit, "offset": (page - 1) * limit}
        if search:
            params["query"] = search
        async with self._http() as client:
            resp = await client.get(f"{_CLERK_API}/users", params=params, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        users = [self._parse_user(u) for u in (data if isinstance(data, list) else data.get("data", []))]
        total_count_resp_data = data.get("total_count", len(users)) if isinstance(data, dict) else len(users)
        return UserList(users=users, total=total_count_resp_data, page=page, limit=limit)

    @with_timeout(10.0)
    async def send_password_reset(self, email: str) -> None:
        async with self._http() as client:
            resp = await client.post(
                f"{_CLERK_API}/users/reset_password",
                json={"email_address": email},
                headers=self._headers(),
            )
            resp.raise_for_status()

    async def reset_password(self, token: str, new_password: str) -> None:
        raise ProviderNotImplementedError("Clerk password reset is handled via Frontend SDK link")

    @with_timeout(10.0)
    async def send_magic_link(self, email: str, redirect_url: str) -> None:
        """Send a Clerk magic link (email code) to the user."""
        # First find or create user
        users = await self.list_users(search=email, limit=1)
        if not users.users:
            raise KeyError(f"No user found with email {email!r}")
        user = users.users[0]
        async with self._http() as client:
            resp = await client.post(
                f"{_CLERK_API}/magic_links",
                json={"user_id": user.id, "email_address_id": email, "redirect_url": redirect_url},
                headers=self._headers(),
            )
            resp.raise_for_status()

    async def verify_magic_link(self, token: str) -> AuthResult:
        raise ProviderNotImplementedError(
            "Clerk magic link verification happens in the browser via Frontend SDK"
        )

    async def get_oauth_authorization_url(
        self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
    ) -> str:
        # Clerk OAuth URLs are generated by the Frontend SDK
        raise ProviderNotImplementedError(
            "Clerk OAuth authorization URLs are generated by @clerk/clerk-js Frontend SDK"
        )

    async def exchange_oauth_code(self, code: str, state: str, redirect_uri: str, provider: str) -> AuthResult:
        raise ProviderNotImplementedError(
            "Clerk OAuth code exchange is handled automatically by the Frontend SDK"
        )

    @with_timeout(10.0)
    async def setup_mfa(self, user_id: str, mfa_type: str) -> MFASetup:
        async with self._http() as client:
            if mfa_type == "totp":
                resp = await client.post(
                    f"{_CLERK_API}/users/{user_id}/totp",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                return MFASetup(
                    mfa_type="totp",
                    secret=data.get("secret"),
                    qr_code_url=data.get("uri"),
                    backup_codes=data.get("backup_codes", []),
                )
            raise ProviderNotImplementedError(f"Clerk MFA type {mfa_type!r} not supported via Backend API")

    async def verify_mfa(self, user_id: str, code: str, session_token: str) -> AuthResult:
        raise ProviderNotImplementedError("Clerk MFA verification happens via Frontend SDK")

    @with_timeout(10.0)
    async def disable_mfa(self, user_id: str, mfa_type: str) -> None:
        async with self._http() as client:
            if mfa_type == "totp":
                await client.delete(f"{_CLERK_API}/users/{user_id}/totp", headers=self._headers())

    @with_timeout(10.0)
    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        async with self._http() as client:
            resp = await client.get(
                f"{_CLERK_API}/sessions",
                params={"user_id": user_id, "status": "active"},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("data", [])

    @with_timeout(10.0)
    async def revoke_session(self, session_id: str) -> None:
        async with self._http() as client:
            await client.delete(
                f"{_CLERK_API}/sessions/{session_id}/revoke", headers=self._headers()
            )

    async def health_check(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{_CLERK_API}/health", headers=self._headers())
                return {"status": "ok" if resp.status_code < 400 else "degraded",
                        "provider": "clerk"}
        except Exception as exc:
            return {"status": "unhealthy", "provider": "clerk", "error": str(exc)}


class ClerkAuthzProvider:
    """Clerk authorization — RBAC via user public_metadata.roles."""

    provider_name = "clerk"

    def __init__(self, auth_provider: ClerkAuthProvider) -> None:
        self._auth = auth_provider
        self._perm_cache = BoundedCache(max_size=10000)

        # Built-in permission map: role -> permissions
        self._role_perms: dict[str, list[str]] = {
            "admin": ["*"],
            "manager": ["read", "write", "delete"],
            "user": ["read", "write"],
            "viewer": ["read"],
        }

    async def check_permission(self, user_id: str, permission: str, tenant_id: str = "default",
                               resource_id: str | None = None, resource_type: str | None = None,
                               context: dict[str, Any] | None = None) -> bool:
        cache_key = f"clerk_perm:{tenant_id}:{user_id}:{permission}"
        cached = self._perm_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        roles = await self.get_user_roles(user_id, tenant_id)
        result = self._has_permission(roles, permission)
        self._perm_cache.set(cache_key, result, ttl=120)
        return result

    def _has_permission(self, roles: list[str], permission: str) -> bool:
        for role in roles:
            perms = self._role_perms.get(role, [])
            if "*" in perms or permission in perms:
                return True
        return False

    async def check_resource_access(self, user_id: str, resource_type: str, resource_id: str,
                                    action: str, tenant_id: str = "default") -> bool:
        return await self.check_permission(user_id, action, tenant_id, resource_id, resource_type)

    async def get_user_roles(self, user_id: str, tenant_id: str = "default") -> list[str]:
        try:
            user = await self._auth.get_user(user_id)
            return user.roles or user.metadata.get("roles", [])
        except Exception as exc:
            _log.debug("Suppressed %s: %s", type(exc).__name__, exc)
            return []

    async def assign_role(self, user_id: str, role: str, tenant_id: str = "default",
                          granted_by: str = "system") -> None:
        user = await self._auth.get_user(user_id)
        roles = list(set(user.roles + [role]))
        await self._auth.update_user(user_id, {"roles": roles})
        self._perm_cache.clear()

    async def revoke_role(self, user_id: str, role: str, tenant_id: str = "default",
                          revoked_by: str = "system") -> None:
        user = await self._auth.get_user(user_id)
        roles = [r for r in user.roles if r != role]
        await self._auth.update_user(user_id, {"roles": roles})
        self._perm_cache.clear()

    async def get_role_permissions(self, role: str, tenant_id: str = "default") -> list[str]:
        return self._role_perms.get(role, [])

    async def create_role(self, role: str, permissions: list[str], tenant_id: str = "default",
                          description: str = "") -> dict[str, Any]:
        self._role_perms[role] = permissions
        return {"role": role, "permissions": permissions}

    async def delete_role(self, role: str, tenant_id: str = "default") -> None:
        self._role_perms.pop(role, None)

    async def list_roles(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        return [{"role": r, "permissions": p} for r, p in self._role_perms.items()]

    async def write_relationship(self, resource_type: str, resource_id: str, relation: str,
                                 subject_type: str, subject_id: str) -> None:
        pass  # Clerk doesn't support relationship tuples

    async def delete_relationship(self, resource_type: str, resource_id: str, relation: str,
                                  subject_type: str, subject_id: str) -> None:
        pass

    async def list_accessible_resources(self, user_id: str, resource_type: str, action: str,
                                        tenant_id: str = "default") -> list[str]:
        return []

    async def bulk_check_permissions(self, user_id: str, checks: list[dict[str, Any]],
                                     tenant_id: str = "default") -> dict[str, bool]:
        roles = await self.get_user_roles(user_id, tenant_id)
        return {c.get("permission", ""): self._has_permission(roles, c.get("permission", ""))
                for c in checks}

    async def health_check(self) -> dict[str, Any]:
        return await self._auth.health_check()
