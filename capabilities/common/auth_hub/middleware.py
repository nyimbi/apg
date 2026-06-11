"""Flask middleware for APG auth hub — decorators and request context.

Usage:
    from capabilities.common.auth_hub.middleware import require_auth, require_permission

    @app.get("/api/accounts")
    @require_auth
    async def list_accounts():
        user = get_current_user()
        ...

    @app.post("/api/payments")
    @require_permission("payments:write")
    async def create_payment():
        ...

    @app.delete("/api/users/<user_id>")
    @require_permission("users:delete", resource_type="user")
    async def delete_user(user_id: str):
        ...
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable

_log = logging.getLogger(__name__)

_CURRENT_USER_KEY = "_auth_hub_user"
_CURRENT_TOKEN_KEY = "_auth_hub_token"


def _get_token_from_request() -> str | None:
    """Extract bearer token from Authorization header or cookie."""
    from flask import request
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    # Fallback: check cookie
    return request.cookies.get("apg_session_token")


def get_current_user() -> Any | None:
    """Return the authenticated user from the current request context."""
    from flask import g
    return getattr(g, _CURRENT_USER_KEY, None)


def get_current_token_payload() -> Any | None:
    """Return the validated token payload from the current request context."""
    from flask import g
    return getattr(g, _CURRENT_TOKEN_KEY, None)


def require_auth(fn: Callable) -> Callable:
    """Decorator: require a valid auth token. Returns 401 if missing/invalid."""
    @functools.wraps(fn)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        from flask import g, jsonify
        from .factory import get_auth_provider
        from .protocols import AuthenticationError

        token = _get_token_from_request()
        if not token:
            return jsonify({"error": "Authentication required", "code": "missing_token"}), 401

        try:
            payload = await get_auth_provider().validate_token(token)
            g._auth_hub_token = payload
            g._auth_hub_user = payload  # lightweight — full user fetch done lazily
        except AuthenticationError as exc:
            return jsonify({"error": str(exc), "code": exc.code}), 401
        except Exception as exc:
            _log.error("Token validation error: %s", exc)
            return jsonify({"error": "Authentication service error"}), 503

        return await fn(*args, **kwargs)

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        finally:
            loop.close()

    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper


def require_permission(
    permission: str,
    resource_type: str | None = None,
    resource_id_param: str | None = None,
) -> Callable:
    """Decorator factory: require auth + specific permission.

    Args:
        permission: Permission name, e.g. "payments:write"
        resource_type: Optional resource type for ReBAC checks
        resource_id_param: URL parameter name containing the resource ID
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        @require_auth
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from flask import jsonify
            from .factory import get_authz_provider
            from .protocols import AuthenticationError, AuthorizationError

            payload = get_current_token_payload()
            if payload is None:
                return jsonify({"error": "Not authenticated"}), 401

            resource_id = kwargs.get(resource_id_param) if resource_id_param else None
            tenant_id = payload.tenant_id

            try:
                allowed = await get_authz_provider().check_permission(
                    user_id=payload.user_id,
                    permission=permission,
                    tenant_id=tenant_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                )
            except Exception as exc:
                _log.error("Permission check error: %s", exc)
                return jsonify({"error": "Authorization service error"}), 503

            if not allowed:
                return jsonify({
                    "error": "Forbidden",
                    "code": "permission_denied",
                    "required_permission": permission,
                }), 403

            return await fn(*args, **kwargs)

        return wrapper
    return decorator


def require_role(role: str) -> Callable:
    """Decorator: require auth + specific role membership."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        @require_auth
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from flask import jsonify
            payload = get_current_token_payload()
            if payload is None:
                return jsonify({"error": "Not authenticated"}), 401
            if role not in payload.roles and "admin" not in payload.roles:
                return jsonify({
                    "error": "Forbidden",
                    "code": "role_required",
                    "required_role": role,
                }), 403
            return await fn(*args, **kwargs)
        return wrapper
    return decorator
