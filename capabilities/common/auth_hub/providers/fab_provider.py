"""Flask-AppBuilder (FAB) auth/authz provider.

Wraps the existing Flask-AppBuilder security manager that APG already uses.
This is the zero-dependency option — works without any external service.

FAB supports:
  - Database (username/password stored in apg DB)
  - LDAP/Active Directory
  - OAuth (GitHub, Google, Azure AD, etc.)
  - REMOTE_USER (reverse-proxy auth)
  - OpenID

Config: no extra env vars — uses existing Flask-AppBuilder configuration.
Set APG_AUTH_PROVIDER=fab to use this provider.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from capabilities.common.reliability import BoundedCache
from ..protocols import (
    AuthResult, AuthUser, AuthenticationError, MFASetup,
    ProviderNotImplementedError, TokenPair, TokenPayload, UserList,
)

_log = logging.getLogger(__name__)

try:
	from uuid_extensions import uuid7str
except ImportError:
	try:
		from uuid6 import uuid7
		def uuid7str() -> str:
			return str(uuid7())
	except ImportError:
		import uuid
		def uuid7str() -> str:  # type: ignore[misc]
			return str(uuid.uuid4())


def _get_security_manager() -> Any:
	"""Get the FAB security manager from the current Flask app context."""
	try:
		from flask import current_app
		return current_app.appbuilder.sm
	except Exception as exc:
		raise RuntimeError(
			"FABAuthProvider requires a Flask application context. "
			"Ensure this is called from within a Flask request or app context."
		) from exc


def _make_jwt_token(user_id: str, email: str, roles: list[str], expires_in: int = 3600) -> str:
	"""Generate a signed JWT for FAB users (uses PyJWT if available)."""
	import os
	try:
		import jwt
		secret = os.environ.get("APG_JWT_SECRET", os.environ.get("SECRET_KEY", "apg-dev-secret"))
		now = datetime.now(timezone.utc)
		payload = {
			"sub": user_id,
			"email": email,
			"roles": roles,
			"iat": now,
			"exp": now + timedelta(seconds=expires_in),
		}
		return jwt.encode(payload, secret, algorithm="HS256")
	except ImportError:
		# Fallback: opaque token (store in cache)
		return f"fab-{uuid7str()}"


class FABAuthProvider:
	"""Flask-AppBuilder authentication provider.

	Wraps the FAB SecurityManager to provide the same interface as
	Keycloak/Clerk/BetterAuth. Works out-of-the-box with no extra services.
	"""

	provider_name = "fab"

	def __init__(self) -> None:
		self._token_cache = BoundedCache(max_size=5000)
		# Store opaque tokens → user_id for providers that can't do JWT
		self._session_store: dict[str, dict[str, Any]] = {}

	def _user_to_auth_user(self, user: Any) -> AuthUser:
		"""Convert FAB User model to AuthUser."""
		if user is None:
			raise AuthenticationError("User not found", "user_not_found")
		roles = [r.name for r in getattr(user, "roles", [])]
		return AuthUser(
			id=str(user.id),
			email=getattr(user, "email", ""),
			username=getattr(user, "username", ""),
			first_name=getattr(user, "first_name", ""),
			last_name=getattr(user, "last_name", ""),
			is_active=getattr(user, "active", True),
			is_email_verified=True,  # FAB doesn't track this by default
			roles=roles,
		)

	async def authenticate(self, credentials: dict[str, Any]) -> AuthResult:
		sm = _get_security_manager()
		username = credentials.get("username") or credentials.get("email", "")
		password = credentials.get("password", "")
		try:
			user = sm.auth_user_db(username, password)
			if user is None:
				# Try email-based login
				user = sm.auth_user_db(username, password)
			if user is None:
				raise AuthenticationError("Invalid username or password", "invalid_credentials")
		except AuthenticationError:
			raise
		except Exception as exc:
			raise AuthenticationError(f"FAB authentication failed: {exc}") from exc

		auth_user = self._user_to_auth_user(user)
		token = _make_jwt_token(auth_user.id, auth_user.email, auth_user.roles)

		# Cache the session
		self._session_store[token] = {
			"user_id": auth_user.id,
			"email": auth_user.email,
			"roles": auth_user.roles,
			"created_at": datetime.now(timezone.utc).isoformat(),
		}

		return AuthResult(
			user=auth_user,
			tokens=TokenPair(
				access_token=token,
				refresh_token=f"refresh-{uuid7str()}",
				expires_in=3600,
			),
		)

	async def validate_token(self, token: str) -> TokenPayload:
		cached = self._token_cache.get(token[:32])
		if cached:
			return cached

		# Try JWT validation first
		try:
			import os, jwt as pyjwt
			secret = os.environ.get("APG_JWT_SECRET", os.environ.get("SECRET_KEY", "apg-dev-secret"))
			payload_data = pyjwt.decode(token, secret, algorithms=["HS256"])
			payload = TokenPayload(
				user_id=payload_data["sub"],
				email=payload_data.get("email", ""),
				roles=payload_data.get("roles", []),
				expires_at=datetime.fromtimestamp(payload_data["exp"], tz=timezone.utc),
			)
			self._token_cache.set(token[:32], payload, ttl=300)
			return payload
		except ImportError:
			pass
		except Exception as exc:
			_log.debug("Suppressed JWT decode: %s: %s", type(exc).__name__, exc)

		# Fallback: check session store
		session = self._session_store.get(token)
		if session:
			payload = TokenPayload(
				user_id=session["user_id"],
				email=session["email"],
				roles=session["roles"],
				expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
			)
			self._token_cache.set(token[:32], payload, ttl=300)
			return payload

		raise AuthenticationError("Invalid or expired token", "token_invalid")

	async def refresh_token(self, refresh_token: str) -> TokenPair:
		# Find session by refresh token mapping
		for token, session in self._session_store.items():
			if session.get("refresh_token") == refresh_token:
				new_token = _make_jwt_token(session["user_id"], session["email"], session["roles"])
				return TokenPair(access_token=new_token, refresh_token=f"refresh-{uuid7str()}", expires_in=3600)
		raise AuthenticationError("Refresh token invalid or expired", "refresh_expired")

	async def logout(self, token: str, refresh_token: str | None = None) -> None:
		self._token_cache.delete(token[:32])
		self._session_store.pop(token, None)

	async def create_user(self, user_data: dict[str, Any]) -> AuthUser:
		sm = _get_security_manager()
		user = sm.add_user(
			username=user_data.get("username", user_data.get("email", "")),
			first_name=user_data.get("first_name", ""),
			last_name=user_data.get("last_name", ""),
			email=user_data.get("email", ""),
			role=sm.find_role(user_data.get("roles", ["User"])[0] if user_data.get("roles") else "User"),
			password=user_data.get("password", uuid7str()),
		)
		if user is None:
			raise ValueError(f"Failed to create user {user_data.get('email', '')!r}")
		return self._user_to_auth_user(user)

	async def get_user(self, user_id: str) -> AuthUser:
		sm = _get_security_manager()
		user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
		if user is None:
			raise KeyError(f"User {user_id!r} not found")
		return self._user_to_auth_user(user)

	async def update_user(self, user_id: str, updates: dict[str, Any]) -> AuthUser:
		sm = _get_security_manager()
		user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
		if user is None:
			raise KeyError(f"User {user_id!r} not found")
		if "first_name" in updates:
			user.first_name = updates["first_name"]
		if "last_name" in updates:
			user.last_name = updates["last_name"]
		if "email" in updates:
			user.email = updates["email"]
		if "is_active" in updates:
			user.active = updates["is_active"]
		sm.update_user(user)
		return self._user_to_auth_user(user)

	async def delete_user(self, user_id: str) -> None:
		sm = _get_security_manager()
		user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
		if user:
			sm.del_register_user(user)

	async def list_users(self, search: str | None = None, limit: int = 50, page: int = 1) -> UserList:
		sm = _get_security_manager()
		all_users = sm.get_all_users()
		if search:
			q = search.lower()
			all_users = [
				u for u in all_users
				if q in getattr(u, "email", "").lower()
				or q in getattr(u, "username", "").lower()
				or q in getattr(u, "first_name", "").lower()
			]
		start = (page - 1) * limit
		page_users = all_users[start:start + limit]
		return UserList(
			users=[self._user_to_auth_user(u) for u in page_users],
			total=len(all_users),
			page=page,
			limit=limit,
			has_more=start + limit < len(all_users),
		)

	async def send_password_reset(self, email: str) -> None:
		raise ProviderNotImplementedError(
			"FAB does not include built-in password reset emails. "
			"Implement a custom reset flow or use Clerk/BetterAuth."
		)

	async def reset_password(self, token: str, new_password: str) -> None:
		raise ProviderNotImplementedError("FAB password reset requires custom implementation")

	async def send_magic_link(self, email: str, redirect_url: str) -> None:
		raise ProviderNotImplementedError("FAB does not support magic links")

	async def verify_magic_link(self, token: str) -> AuthResult:
		raise ProviderNotImplementedError("FAB does not support magic links")

	async def get_oauth_authorization_url(
		self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
	) -> str:
		raise ProviderNotImplementedError(
			"FAB OAuth URLs are generated by Flask-AppBuilder views, not via API. "
			"Use Keycloak or Clerk for programmatic OAuth."
		)

	async def exchange_oauth_code(self, code: str, state: str, redirect_uri: str, provider: str) -> AuthResult:
		raise ProviderNotImplementedError("FAB OAuth code exchange is handled by Flask views")

	async def setup_mfa(self, user_id: str, mfa_type: str) -> MFASetup:
		raise ProviderNotImplementedError("FAB does not include built-in MFA. Use Keycloak or Clerk.")

	async def verify_mfa(self, user_id: str, code: str, session_token: str) -> AuthResult:
		raise ProviderNotImplementedError("FAB does not include built-in MFA")

	async def disable_mfa(self, user_id: str, mfa_type: str) -> None:
		pass

	async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
		sessions = [
			{**s, "token": t}
			for t, s in self._session_store.items()
			if s.get("user_id") == user_id
		]
		return sessions

	async def revoke_session(self, session_id: str) -> None:
		self._session_store.pop(session_id, None)
		self._token_cache.delete(session_id[:32])

	async def health_check(self) -> dict[str, Any]:
		try:
			sm = _get_security_manager()
			return {"status": "ok", "provider": "fab", "backend": type(sm).__name__}
		except Exception as exc:
			return {"status": "degraded", "provider": "fab", "note": str(exc)}


class FABAuthzProvider:
	"""Flask-AppBuilder authorization provider using FAB roles/permissions."""

	provider_name = "fab"

	def __init__(self) -> None:
		self._perm_cache = BoundedCache(max_size=10000)

	async def check_permission(self, user_id: str, permission: str, tenant_id: str = "default",
							   resource_id: str | None = None, resource_type: str | None = None,
							   context: dict[str, Any] | None = None) -> bool:
		cache_key = f"fab_perm:{tenant_id}:{user_id}:{permission}"
		cached = self._perm_cache.get(cache_key)
		if cached is not None:
			return bool(cached)
		try:
			sm = _get_security_manager()
			user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
			if user is None:
				return False
			# Check FAB permission
			result = sm.has_access(permission, resource_type or "APG")
			self._perm_cache.set(cache_key, result, ttl=60)
			return result
		except Exception as exc:
			_log.debug("Suppressed %s: %s", type(exc).__name__, exc)
			return False

	async def check_resource_access(self, user_id: str, resource_type: str, resource_id: str,
									action: str, tenant_id: str = "default") -> bool:
		return await self.check_permission(user_id, action, tenant_id, resource_id, resource_type)

	async def get_user_roles(self, user_id: str, tenant_id: str = "default") -> list[str]:
		try:
			sm = _get_security_manager()
			user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
			if user is None:
				return []
			return [r.name for r in getattr(user, "roles", [])]
		except Exception as exc:
			_log.debug("Suppressed %s: %s", type(exc).__name__, exc)
			return []

	async def assign_role(self, user_id: str, role: str, tenant_id: str = "default",
						  granted_by: str = "system") -> None:
		sm = _get_security_manager()
		user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
		fab_role = sm.find_role(role)
		if user and fab_role:
			user.roles.append(fab_role)
			sm.update_user(user)
		self._perm_cache.clear()

	async def revoke_role(self, user_id: str, role: str, tenant_id: str = "default",
						  revoked_by: str = "system") -> None:
		sm = _get_security_manager()
		user = sm.get_user_by_id(int(user_id)) if user_id.isdigit() else sm.find_user(username=user_id)
		if user:
			user.roles = [r for r in user.roles if r.name != role]
			sm.update_user(user)
		self._perm_cache.clear()

	async def get_role_permissions(self, role: str, tenant_id: str = "default") -> list[str]:
		try:
			sm = _get_security_manager()
			fab_role = sm.find_role(role)
			if fab_role:
				return [f"{p.view_menu.name}:{p.permission.name}" for p in fab_role.permissions]
			return []
		except Exception as exc:
			_log.debug("Suppressed %s: %s", type(exc).__name__, exc)
			return []

	async def create_role(self, role: str, permissions: list[str], tenant_id: str = "default",
						  description: str = "") -> dict[str, Any]:
		sm = _get_security_manager()
		fab_role = sm.add_role(role)
		return {"role": role, "permissions": permissions, "fab_role_id": fab_role.id if fab_role else None}

	async def delete_role(self, role: str, tenant_id: str = "default") -> None:
		sm = _get_security_manager()
		fab_role = sm.find_role(role)
		if fab_role:
			sm.del_role(fab_role)

	async def list_roles(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		try:
			sm = _get_security_manager()
			return [{"role": r.name, "id": r.id} for r in sm.get_all_roles()]
		except Exception as exc:
			_log.debug("Suppressed %s: %s", type(exc).__name__, exc)
			return []

	async def write_relationship(self, resource_type: str, resource_id: str, relation: str,
								 subject_type: str, subject_id: str) -> None:
		pass  # FAB doesn't support relationship tuples

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
		try:
			sm = _get_security_manager()
			return {"status": "ok", "provider": "fab", "backend": type(sm).__name__}
		except Exception as exc:
			return {"status": "degraded", "provider": "fab", "note": str(exc)}
