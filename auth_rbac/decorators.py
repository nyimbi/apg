"""auth_rbac decorators stubs."""
from .service import require_permission, require_permissions, get_current_user, get_current_tenant

__all__ = ["require_permission", "require_permissions", "get_current_user", "get_current_tenant"]
