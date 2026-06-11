"""APG Auth Hub — interchangeable auth/authz provider adapter.

Supported providers:
    Auth:   keycloak | clerk | betterauth | fab | null
    Authz:  spicedb | keycloak | clerk | fab | null

Mix and match via environment variables:
    APG_AUTH_PROVIDER=clerk APG_AUTHZ_PROVIDER=spicedb
"""
from .service import AuthHubService
from .factory import get_auth_provider, get_authz_provider, provider_info
from .protocols import (
    AuthProvider, AuthzProvider,
    AuthResult, AuthUser, TokenPair, TokenPayload, MFASetup, UserList,
    AuthenticationError, AuthorizationError, ProviderNotImplementedError,
)
from .middleware import require_auth, require_permission, require_role, get_current_user

__all__ = [
    "AuthHubService",
    "get_auth_provider", "get_authz_provider", "provider_info",
    "AuthProvider", "AuthzProvider",
    "AuthResult", "AuthUser", "TokenPair", "TokenPayload", "MFASetup", "UserList",
    "AuthenticationError", "AuthorizationError", "ProviderNotImplementedError",
    "require_auth", "require_permission", "require_role", "get_current_user",
]
