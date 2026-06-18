"""auth_rbac service stubs."""
from __future__ import annotations
from typing import Any, Callable
from functools import wraps


class AuthService:
	"""Stub auth service."""
	async def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
		return None

	async def authorize(self, user_id: str, permission: str, resource: str | None = None) -> bool:
		return True


class AuthRBACService(AuthService):
	"""Stub RBAC service."""
	async def check_permission(self, user_id: str, permission: str) -> bool:
		return True

	async def get_user_roles(self, user_id: str) -> list[str]:
		return []


class UserManagementService:
	"""Stub user management service."""
	async def get_user(self, user_id: str) -> dict[str, Any] | None:
		return None


def get_current_user() -> dict[str, Any] | None:
	return None


def get_current_tenant() -> str | None:
	return None


def require_permission(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def require_permissions(*permissions: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			return fn(*args, **kwargs)
		return wrapper
	return decorator


class APGSecurityManager:
	"""Stub security manager."""
	def has_access(self, resource: str, action: str) -> bool:
		return True


__all__ = [
	"AuthService", "AuthRBACService", "UserManagementService",
	"APGSecurityManager",
	"get_current_user", "get_current_tenant",
	"require_permission", "require_permissions",
]
