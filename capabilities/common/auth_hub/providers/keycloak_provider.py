"""Keycloak auth/authz provider.

Keycloak handles both authentication (OIDC/OAuth2) and authorization
(Keycloak Authorization Services with fine-grained policies).

Config (via environment):
    APG_KEYCLOAK_URL           https://auth.example.com
    APG_KEYCLOAK_REALM         apg
    APG_KEYCLOAK_CLIENT_ID     apg-backend
    APG_KEYCLOAK_CLIENT_SECRET <secret>
    APG_KEYCLOAK_ADMIN_USER    admin          (for admin API calls)
    APG_KEYCLOAK_ADMIN_PASS    <password>
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from capabilities.common.reliability import BoundedCache, guard_non_empty_string
from capabilities.common.reliability.circuit_breaker import get_circuit_breaker
from capabilities.common.reliability.timeout import with_timeout
from ..protocols import (
    AuthResult, AuthUser, AuthenticationError, MFASetup,
    ProviderNotImplementedError, TokenPair, TokenPayload, UserList,
)

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10.0



import hashlib as _hashlib

def _cache_key(token: str) -> str:
    """Blake2b hash of the full token — collision-resistant cache key."""
    return _hashlib.blake2b(token.encode(), digest_size=32).hexdigest()

class KeycloakAuthProvider:
    """Keycloak OIDC authentication provider."""

    provider_name = "keycloak"

    def __init__(
        self,
        base_url: str | None = None,
        realm: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        admin_username: str | None = None,
        admin_password: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("APG_KEYCLOAK_URL", "")).rstrip("/")
        self._realm = realm or os.environ.get("APG_KEYCLOAK_REALM", "master")
        self._client_id = client_id or os.environ.get("APG_KEYCLOAK_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("APG_KEYCLOAK_CLIENT_SECRET", "")
        self._admin_user = admin_username or os.environ.get("APG_KEYCLOAK_ADMIN_USER", "admin")
        self._admin_pass = admin_password or os.environ.get("APG_KEYCLOAK_ADMIN_PASS", "")

        self._realm_url = f"{self._base_url}/realms/{self._realm}"
        self._admin_url = f"{self._base_url}/admin/realms/{self._realm}"
        self._oidc_url = f"{self._realm_url}/protocol/openid-connect"

        # Admin token cache
        self._admin_token: str = ""
        self._admin_token_expires: float = 0.0

        # Token validation cache (avoids hitting Keycloak on every request)
        self._token_cache = BoundedCache(max_size=5000)
        self._cb = get_circuit_breaker("keycloak_auth", failure_threshold=5, reset_timeout=60.0)
        self._shared_client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )

    def _http(self, timeout: float = _DEFAULT_TIMEOUT) -> httpx.AsyncClient:
        """Return shared persistent client — connection pool reuse."""
        return self._shared_client

    async def close(self) -> None:
        """Release connection pool. Call from application shutdown."""
        await self._shared_client.aclose()

    async def _get_admin_token(self) -> str:
        """Get or refresh the admin access token."""
        if time.monotonic() < self._admin_token_expires - 30:
            return self._admin_token
        client = self._shared_client
        resp = await client.post(
            f"{self._realm_url}/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": self._admin_user,
                "password": self._admin_pass,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._admin_token = data["access_token"]
        self._admin_token_expires = time.monotonic() + data.get("expires_in", 60)
        return self._admin_token

    def _parse_token_response(self, data: dict[str, Any], user_info: dict[str, Any]) -> AuthResult:
        user = AuthUser(
            id=user_info.get("sub", ""),
            email=user_info.get("email", ""),
            username=user_info.get("preferred_username", ""),
            first_name=user_info.get("given_name", ""),
            last_name=user_info.get("family_name", ""),
            is_active=True,
            is_email_verified=user_info.get("email_verified", False),
            roles=user_info.get("roles", []),
        )
        tokens = TokenPair(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", 3600),
            scope=data.get("scope", ""),
        )
        return AuthResult(user=user, tokens=tokens)

    @with_timeout(15.0)
    async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
        await self._cb._before_call()
        try:
            client = self._shared_client
            resp = await client.post(
                f"{self._oidc_url}/token",
                data={
                    "grant_type": "password",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "username": credentials.get("username", ""),
                    "password": credentials.get("password", ""),
                    "scope": "openid profile email",
                },
            )
            if resp.status_code in (401, 403):
                raise AuthenticationError("Invalid username or password", "invalid_credentials")
            resp.raise_for_status()
            token_data = resp.json()

            # Get user info
            userinfo_resp = await client.get(
                f"{self._oidc_url}/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            userinfo_resp.raise_for_status()
            await self._cb._on_success()
            return self._parse_token_response(token_data, userinfo_resp.json())
        except AuthenticationError:
            raise
        except Exception as exc:
            await self._cb._on_failure(exc)
            raise AuthenticationError(f"Keycloak authentication failed: {exc}") from exc

    @with_timeout(5.0)
    async def validate_token(self, token: str) -> TokenPayload:
        cached = self._token_cache.get(_cache_key(token))
        if cached:
            return cached

        await self._cb._before_call()
        try:
            client = self._shared_client
            resp = await client.post(
                f"{self._oidc_url}/token/introspect",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "token": token,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("active"):
                raise AuthenticationError("Token is inactive or expired", "token_expired")

            realm_access = data.get("realm_access", {})
            roles = realm_access.get("roles", [])
            exp = data.get("exp")
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None

            payload = TokenPayload(
                user_id=data.get("sub", ""),
                email=data.get("email", ""),
                roles=roles,
                tenant_id=data.get("tenant_id", data.get("azp", "default")),
                expires_at=expires_at,
                extra=data,
            )
            ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds()) if expires_at else 300
            self._token_cache.set(_cache_key(token), payload, ttl=min(ttl, 300))
            await self._cb._on_success()
            return payload
        except AuthenticationError:
            raise
        except Exception as exc:
            await self._cb._on_failure(exc)
            raise AuthenticationError(f"Token validation failed: {exc}") from exc

    @with_timeout(10.0)
    async def refresh_token(self, refresh_token: str) -> TokenPair:
        client = self._shared_client
        resp = await client.post(
            f"{self._oidc_url}/token",
            data={
                "grant_type": "refresh_token",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": refresh_token,
            },
        )
        if resp.status_code in (400, 401):
            raise AuthenticationError("Refresh token expired or invalid", "refresh_expired")
        resp.raise_for_status()
        data = resp.json()
        return TokenPair(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_in=data.get("expires_in", 3600),
        )

    @with_timeout(10.0)
    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        self._token_cache.delete(_cache_key(token))
        if refresh_token:
            client = self._shared_client
            await client.post(
                f"{self._oidc_url}/logout",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": refresh_token,
                },
            )

    @with_timeout(10.0)
    async def create_user(self, user_data: dict[str, Any]) -> AuthUser:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        payload = {
            "username": user_data.get("username", user_data.get("email", "")),
            "email": user_data.get("email", ""),
            "firstName": user_data.get("first_name", ""),
            "lastName": user_data.get("last_name", ""),
            "enabled": True,
            "emailVerified": False,
        }
        if password := user_data.get("password"):
            payload["credentials"] = [{"type": "password", "value": password, "temporary": False}]
        resp = await client.post(
            f"{self._admin_url}/users",
            json=payload,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()
        location = resp.headers.get("Location", "")
        user_id = location.split("/")[-1] if location else ""
        return AuthUser(
            id=user_id,
            email=user_data.get("email", ""),
            username=user_data.get("username", user_data.get("email", "")),
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            is_active=True,
        )

    @with_timeout(10.0)
    async def get_user(self, user_id: str) -> AuthUser:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        resp = await client.get(
            f"{self._admin_url}/users/{user_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()
        d = resp.json()
        return AuthUser(
            id=d.get("id", user_id),
            email=d.get("email", ""),
            username=d.get("username", ""),
            first_name=d.get("firstName", ""),
            last_name=d.get("lastName", ""),
            is_active=d.get("enabled", True),
            is_email_verified=d.get("emailVerified", False),
        )

    @with_timeout(10.0)
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> AuthUser:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        payload: dict[str, Any] = {}
        if "email" in updates:
            payload["email"] = updates["email"]
        if "first_name" in updates:
            payload["firstName"] = updates["first_name"]
        if "last_name" in updates:
            payload["lastName"] = updates["last_name"]
        if "is_active" in updates:
            payload["enabled"] = updates["is_active"]
        resp = await client.put(
            f"{self._admin_url}/users/{user_id}",
            json=payload,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()
        return await self.get_user(user_id)

    @with_timeout(10.0)
    async def delete_user(self, user_id: str) -> None:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        resp = await client.delete(
            f"{self._admin_url}/users/{user_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()

    @with_timeout(15.0)
    async def list_users(self, search: str | None = None, limit: int = 50, page: int = 1) -> UserList:
        admin_token = await self._get_admin_token()
        params: dict[str, Any] = {"max": limit, "first": (page - 1) * limit}
        if search:
            params["search"] = search
        client = self._shared_client
        resp = await client.get(
            f"{self._admin_url}/users",
            params=params,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()
        users_data = resp.json()
        count_resp = await client.get(
            f"{self._admin_url}/users/count",
            params={"search": search} if search else {},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        total = count_resp.json() if count_resp.status_code == 200 else len(users_data)

        users = [
            AuthUser(
                id=u.get("id", ""), email=u.get("email", ""),
                username=u.get("username", ""),
                first_name=u.get("firstName", ""), last_name=u.get("lastName", ""),
                is_active=u.get("enabled", True), is_email_verified=u.get("emailVerified", False),
            )
            for u in users_data
        ]
        return UserList(users=users, total=total, page=page, limit=limit, has_more=len(users) >= limit)

    async def send_password_reset(self, email: str) -> None:
        users = await self.list_users(search=email, limit=1)
        if users.users:
            admin_token = await self._get_admin_token()
            client = self._shared_client
            await client.put(
                f"{self._admin_url}/users/{users.users[0].id}/execute-actions-email",
                json=["UPDATE_PASSWORD"],
                headers={"Authorization": f"Bearer {admin_token}"},
            )

    async def reset_password(self, token: str, new_password: str) -> None:
        raise ProviderNotImplementedError("Keycloak handles password reset via email link — use send_password_reset()")

    async def send_magic_link(self, email: str, redirect_url: str) -> None:
        raise ProviderNotImplementedError("Keycloak does not support magic links natively — use Clerk or BetterAuth")

    async def verify_magic_link(self, token: str) -> AuthResult:
        raise ProviderNotImplementedError("Keycloak does not support magic links")

    async def get_oauth_authorization_url(
        self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
    ) -> str:
        scope_str = " ".join(scopes or ["openid", "profile", "email"])
        return (
            f"{self._oidc_url}/auth?response_type=code"
            f"&client_id={self._client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
            f"&scope={scope_str}"
            f"&kc_idp_hint={provider}"
        )

    async def exchange_oauth_code(self, code: str, state: str, redirect_uri: str, provider: str) -> AuthResult:
        client = self._shared_client
        resp = await client.post(
            f"{self._oidc_url}/token",
            data={
                "grant_type": "authorization_code",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        resp.raise_for_status()
        token_data = resp.json()
        userinfo_resp = await client.get(
            f"{self._oidc_url}/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        return self._parse_token_response(token_data, userinfo_resp.json())

    async def setup_mfa(self, user_id: str, mfa_type: str) -> MFASetup:
        # Keycloak TOTP: trigger required action
        admin_token = await self._get_admin_token()
        client = self._shared_client
        await client.put(
            f"{self._admin_url}/users/{user_id}/execute-actions-email",
            json=["CONFIGURE_TOTP"],
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        return MFASetup(mfa_type="totp", session_token=user_id)

    async def verify_mfa(self, user_id: str, code: str, session_token: str) -> AuthResult:
        raise ProviderNotImplementedError("Keycloak MFA verification happens in the login flow, not via API")

    async def disable_mfa(self, user_id: str, mfa_type: str) -> None:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        resp = await client.get(
            f"{self._admin_url}/users/{user_id}/credentials",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()
        for cred in resp.json():
            if cred.get("type") == "otp":
                await client.delete(
                    f"{self._admin_url}/users/{user_id}/credentials/{cred['id']}",
                    headers={"Authorization": f"Bearer {admin_token}"},
                )

    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        resp = await client.get(
            f"{self._admin_url}/users/{user_id}/sessions",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        resp.raise_for_status()
        return resp.json()

    async def revoke_session(self, session_id: str) -> None:
        admin_token = await self._get_admin_token()
        client = self._shared_client
        await client.delete(
            f"{self._admin_url}/sessions/{session_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    async def health_check(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._realm_url}/.well-known/openid-configuration")
                return {"status": "ok" if resp.status_code == 200 else "degraded",
                        "provider": "keycloak", "realm": self._realm}
        except Exception as exc:
            return {"status": "unhealthy", "provider": "keycloak", "error": str(exc)}


class KeycloakAuthzProvider:
    """Keycloak Authorization Services — RBAC + UMA fine-grained policies."""

    provider_name = "keycloak"

    def __init__(self, auth_provider: KeycloakAuthProvider) -> None:
        self._auth = auth_provider
        self._perm_cache = BoundedCache(max_size=10000)

    async def _get_resource_token(self, access_token: str, resource: str | None = None) -> dict[str, Any]:
        """Request an RPT (Requesting Party Token) from Keycloak authorization endpoint."""
        data: dict[str, Any] = {
            "grant_type": "urn:ietf:params:oauth:grant-type:uma-ticket",
            "audience": self._auth._client_id,
            "response_include_resource_name": "false",
        }
        if resource:
            data["permission"] = resource
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self._auth._oidc_url}/token",
                data=data,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code == 403:
                return {"result": False}
            resp.raise_for_status()
            return {"result": True, "rpt": resp.json().get("access_token")}

    async def check_permission(self, user_id: str, permission: str, tenant_id: str = "default",
                               resource_id: str | None = None, resource_type: str | None = None,
                               context: dict[str, Any] | None = None) -> bool:
        cache_key = f"kc_perm:{tenant_id}:{user_id}:{permission}:{resource_id}"
        cached = self._perm_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        # Use Keycloak realm roles as a simple permission check
        roles = await self.get_user_roles(user_id, tenant_id)
        result = permission in roles or "admin" in roles
        self._perm_cache.set(cache_key, result, ttl=60)
        return result

    async def check_resource_access(self, user_id: str, resource_type: str, resource_id: str,
                                    action: str, tenant_id: str = "default") -> bool:
        cache_key = f"kc_res:{tenant_id}:{user_id}:{resource_type}/{resource_id}:{action}"
        cached = self._perm_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        # Simplified: check if user has the role for this action
        roles = await self.get_user_roles(user_id, tenant_id)
        result = "admin" in roles or f"{resource_type}:{action}" in roles
        self._perm_cache.set(cache_key, result, ttl=60)
        return result

    async def get_user_roles(self, user_id: str, tenant_id: str = "default") -> list[str]:
        admin_token = await self._auth._get_admin_token()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{self._auth._admin_url}/users/{user_id}/role-mappings/realm",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()
            return [r["name"] for r in resp.json()]

    async def assign_role(self, user_id: str, role: str, tenant_id: str = "default",
                          granted_by: str = "system") -> None:
        admin_token = await self._auth._get_admin_token()
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Get role representation
            roles_resp = await client.get(
                f"{self._auth._admin_url}/roles/{role}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            roles_resp.raise_for_status()
            role_rep = roles_resp.json()
            await client.post(
                f"{self._auth._admin_url}/users/{user_id}/role-mappings/realm",
                json=[role_rep],
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        self._perm_cache.clear()

    async def revoke_role(self, user_id: str, role: str, tenant_id: str = "default",
                          revoked_by: str = "system") -> None:
        admin_token = await self._auth._get_admin_token()
        async with httpx.AsyncClient(timeout=5.0) as client:
            roles_resp = await client.get(
                f"{self._auth._admin_url}/roles/{role}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            roles_resp.raise_for_status()
            await client.delete(
                f"{self._auth._admin_url}/users/{user_id}/role-mappings/realm",
                json=[roles_resp.json()],
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        self._perm_cache.clear()

    async def get_role_permissions(self, role: str, tenant_id: str = "default") -> list[str]:
        return [role]

    async def create_role(self, role: str, permissions: list[str], tenant_id: str = "default",
                          description: str = "") -> dict[str, Any]:
        admin_token = await self._auth._get_admin_token()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self._auth._admin_url}/roles",
                json={"name": role, "description": description},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()
        return {"role": role, "permissions": permissions}

    async def delete_role(self, role: str, tenant_id: str = "default") -> None:
        admin_token = await self._auth._get_admin_token()
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.delete(
                f"{self._auth._admin_url}/roles/{role}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

    async def list_roles(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        admin_token = await self._auth._get_admin_token()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{self._auth._admin_url}/roles",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()
            return [{"role": r["name"], "description": r.get("description", "")} for r in resp.json()]

    async def write_relationship(self, resource_type: str, resource_id: str, relation: str,
                                 subject_type: str, subject_id: str) -> None:
        pass  # Keycloak doesn't support SpiceDB-style relationship tuples

    async def delete_relationship(self, resource_type: str, resource_id: str, relation: str,
                                  subject_type: str, subject_id: str) -> None:
        pass

    async def list_accessible_resources(self, user_id: str, resource_type: str, action: str,
                                        tenant_id: str = "default") -> list[str]:
        return []

    async def bulk_check_permissions(self, user_id: str, checks: list[dict[str, Any]],
                                     tenant_id: str = "default") -> dict[str, bool]:
        results = {}
        for check in checks:
            perm = check.get("permission", "")
            results[perm] = await self.check_permission(user_id, perm, tenant_id)
        return results

    async def health_check(self) -> dict[str, Any]:
        return await self._auth.health_check()
