"""BetterAuth provider — proxies to a BetterAuth Node.js service.

BetterAuth (https://better-auth.com) is a TypeScript auth library. Since APG
is Python, BetterAuth runs as a separate microservice and this provider
communicates with it via HTTP. This is the recommended pattern for using
Node.js auth libraries in a Python backend.

Setup:
    1. Run BetterAuth as a Node service:
       npx @better-auth/cli init
       node server.js  # listens on APG_BETTERAUTH_URL

    2. Configure APG:
       APG_BETTERAUTH_URL     http://localhost:3001
       APG_BETTERAUTH_SECRET  <shared-secret>

The BetterAuth service exposes:
    POST /auth/sign-in/email         → {user, session}
    POST /auth/sign-in/magic-link    → {url}
    POST /auth/sign-up/email         → {user, session}
    GET  /auth/get-session           → {session, user}
    POST /auth/sign-out              → {}
    GET  /auth/user/:id              → {user}

Docs: https://better-auth.com/docs
"""
from __future__ import annotations

import logging
import os
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



import hashlib as _hashlib

def _cache_key(token: str) -> str:
    """Blake2b hash of the full token — collision-resistant cache key."""
    return _hashlib.blake2b(token.encode(), digest_size=32).hexdigest()

class BetterAuthProvider:
    """BetterAuth authentication provider (via HTTP proxy to Node.js service)."""

    provider_name = "betterauth"

    def __init__(
        self,
        base_url: str | None = None,
        secret: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("APG_BETTERAUTH_URL", "http://localhost:3001")).rstrip("/")
        self._secret = secret or os.environ.get("APG_BETTERAUTH_SECRET", "")
        self._token_cache = BoundedCache(max_size=5000)
        self._cb = get_circuit_breaker("betterauth", failure_threshold=5, reset_timeout=60.0)

    def _headers(self, session_token: str | None = None) -> dict[str, str]:
        h = {"X-APG-Secret": self._secret, "Content-Type": "application/json"}
        if session_token:
            h["Authorization"] = f"Bearer {session_token}"
        return h

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=10.0)

    def _parse_user(self, u: dict[str, Any]) -> AuthUser:
        return AuthUser(
            id=u.get("id", ""),
            email=u.get("email", ""),
            username=u.get("name", u.get("email", "")),
            first_name=u.get("name", "").split()[0] if u.get("name") else "",
            last_name=" ".join(u.get("name", "").split()[1:]) if u.get("name") else "",
            is_active=True,
            is_email_verified=u.get("emailVerified") is not None,
            roles=u.get("role", "user").split(",") if u.get("role") else ["user"],
            metadata=u,
        )

    def _parse_session(self, session: dict[str, Any], user: dict[str, Any]) -> AuthResult:
        token = session.get("token", session.get("id", ""))
        expires_at = session.get("expiresAt")
        expires_in = 3600
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                expires_in = max(0, int((exp_dt - datetime.now(timezone.utc)).total_seconds()))
            except Exception as exc:
                _log.debug("Suppressed %s: %s", type(exc).__name__, exc)
        return AuthResult(
            user=self._parse_user(user),
            tokens=TokenPair(
                access_token=token,
                refresh_token=session.get("refresh_token", ""),
                expires_in=expires_in,
            ),
        )

    @with_timeout(15.0)
    async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
        await self._cb._before_call()
        try:
            async with self._http() as client:
                resp = await client.post(
                    f"{self._base_url}/auth/sign-in/email",
                    json={
                        "email": credentials.get("email") or credentials.get("username", ""),
                        "password": credentials.get("password", ""),
                    },
                    headers=self._headers(),
                )
                if resp.status_code in (401, 403):
                    raise AuthenticationError("Invalid email or password", "invalid_credentials")
                resp.raise_for_status()
                data = resp.json()
            await self._cb._on_success()
            return self._parse_session(data.get("session", {}), data.get("user", {}))
        except AuthenticationError:
            raise
        except Exception as exc:
            await self._cb._on_failure(exc)
            raise AuthenticationError(f"BetterAuth authentication failed: {exc}") from exc

    @with_timeout(5.0)
    async def validate_token(self, token: str) -> TokenPayload:
        cached = self._token_cache.get(_cache_key(token))
        if cached:
            return cached

        await self._cb._before_call()
        try:
            async with self._http() as client:
                resp = await client.get(
                    f"{self._base_url}/auth/get-session",
                    headers=self._headers(session_token=token),
                )
                if resp.status_code in (401, 403, 404):
                    raise AuthenticationError("Invalid or expired session token", "token_invalid")
                resp.raise_for_status()
                data = resp.json()

            user = data.get("user", {})
            session = data.get("session", {})
            expires_at_str = session.get("expiresAt")
            expires_at = None
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                except Exception as exc:
                    _log.debug("Suppressed %s: %s", type(exc).__name__, exc)

            payload = TokenPayload(
                user_id=user.get("id", ""),
                email=user.get("email", ""),
                roles=user.get("role", "user").split(",") if user.get("role") else ["user"],
                tenant_id=user.get("tenantId", "default"),
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
            raise AuthenticationError(f"BetterAuth token validation failed: {exc}") from exc

    @with_timeout(10.0)
    async def refresh_token(self, refresh_token: str) -> TokenPair:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/refresh",
                json={"refreshToken": refresh_token},
                headers=self._headers(),
            )
            if resp.status_code in (400, 401):
                raise AuthenticationError("Refresh token invalid or expired", "refresh_expired")
            resp.raise_for_status()
            data = resp.json()
            return TokenPair(
                access_token=data.get("token", data.get("accessToken", "")),
                refresh_token=data.get("refreshToken", refresh_token),
                expires_in=data.get("expiresIn", 3600),
            )

    @with_timeout(10.0)
    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        self._token_cache.delete(_cache_key(token))
        try:
            async with self._http() as client:
                await client.post(
                    f"{self._base_url}/auth/sign-out",
                    headers=self._headers(session_token=token),
                )
        except Exception as exc:
            _log.debug("Suppressed %s: %s", type(exc).__name__, exc)

    @with_timeout(10.0)
    async def create_user(self, user_data: dict[str, Any]) -> AuthUser:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/sign-up/email",
                json={
                    "email": user_data.get("email", ""),
                    "password": user_data.get("password", ""),
                    "name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
                },
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_user(data.get("user", data))

    @with_timeout(5.0)
    async def get_user(self, user_id: str) -> AuthUser:
        async with self._http() as client:
            resp = await client.get(f"{self._base_url}/auth/user/{user_id}", headers=self._headers())
            resp.raise_for_status()
            return self._parse_user(resp.json())

    @with_timeout(10.0)
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> AuthUser:
        async with self._http() as client:
            resp = await client.patch(
                f"{self._base_url}/auth/user/{user_id}",
                json=updates,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return self._parse_user(resp.json())

    @with_timeout(10.0)
    async def delete_user(self, user_id: str) -> None:
        async with self._http() as client:
            resp = await client.delete(f"{self._base_url}/auth/user/{user_id}", headers=self._headers())
            resp.raise_for_status()

    @with_timeout(15.0)
    async def list_users(self, search: str | None = None, limit: int = 50, page: int = 1) -> UserList:
        params: dict[str, Any] = {"limit": limit, "page": page}
        if search:
            params["search"] = search
        async with self._http() as client:
            resp = await client.get(f"{self._base_url}/auth/users", params=params, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        users_data = data.get("users", data) if isinstance(data, dict) else data
        users = [self._parse_user(u) for u in users_data]
        return UserList(
            users=users,
            total=data.get("total", len(users)) if isinstance(data, dict) else len(users),
            page=page, limit=limit,
        )

    @with_timeout(10.0)
    async def send_password_reset(self, email: str) -> None:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/forget-password",
                json={"email": email},
                headers=self._headers(),
            )
            resp.raise_for_status()

    @with_timeout(10.0)
    async def reset_password(self, token: str, new_password: str) -> None:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/reset-password",
                json={"token": token, "newPassword": new_password},
                headers=self._headers(),
            )
            resp.raise_for_status()

    @with_timeout(10.0)
    async def send_magic_link(self, email: str, redirect_url: str) -> None:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/sign-in/magic-link",
                json={"email": email, "callbackURL": redirect_url},
                headers=self._headers(),
            )
            resp.raise_for_status()

    async def verify_magic_link(self, token: str) -> AuthResult:
        async with self._http() as client:
            resp = await client.get(
                f"{self._base_url}/auth/magic-link/verify?token={token}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_session(data.get("session", {}), data.get("user", {}))

    async def get_oauth_authorization_url(
        self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
    ) -> str:
        async with self._http() as client:
            resp = await client.get(
                f"{self._base_url}/auth/sign-in/social",
                params={"provider": provider, "callbackURL": redirect_uri, "state": state},
                headers=self._headers(),
                follow_redirects=False,
            )
            location = resp.headers.get("Location", "")
            if location:
                return location
            resp.raise_for_status()
            return resp.json().get("url", "")

    async def exchange_oauth_code(self, code: str, state: str, redirect_uri: str, provider: str) -> AuthResult:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/callback/{provider}",
                json={"code": code, "state": state, "redirectUri": redirect_uri},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_session(data.get("session", {}), data.get("user", {}))

    async def setup_mfa(self, user_id: str, mfa_type: str) -> MFASetup:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/two-factor/enable",
                json={"userId": user_id, "type": mfa_type},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return MFASetup(
                mfa_type=mfa_type,
                secret=data.get("secret"),
                qr_code_url=data.get("qrCode"),
                backup_codes=data.get("backupCodes", []),
            )

    async def verify_mfa(self, user_id: str, code: str, session_token: str) -> AuthResult:
        async with self._http() as client:
            resp = await client.post(
                f"{self._base_url}/auth/two-factor/verify",
                json={"userId": user_id, "code": code},
                headers=self._headers(session_token=session_token),
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_session(data.get("session", {}), data.get("user", {}))

    async def disable_mfa(self, user_id: str, mfa_type: str) -> None:
        async with self._http() as client:
            await client.post(
                f"{self._base_url}/auth/two-factor/disable",
                json={"userId": user_id, "type": mfa_type},
                headers=self._headers(),
            )

    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        async with self._http() as client:
            resp = await client.get(
                f"{self._base_url}/auth/user/{user_id}/sessions", headers=self._headers()
            )
            resp.raise_for_status()
            return resp.json()

    async def revoke_session(self, session_id: str) -> None:
        async with self._http() as client:
            await client.post(
                f"{self._base_url}/auth/revoke-session",
                json={"sessionId": session_id},
                headers=self._headers(),
            )

    async def health_check(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/auth/health", headers=self._headers())
                return {"status": "ok" if resp.status_code < 400 else "degraded",
                        "provider": "betterauth", "url": self._base_url}
        except Exception as exc:
            return {"status": "unhealthy", "provider": "betterauth", "error": str(exc)}
