"""Auth hub protocols — structural typing for interchangeable auth/authz providers.

Any class implementing these methods can serve as a provider, regardless of
inheritance. This enables mixing providers: Clerk for auth + SpiceDB for authz,
or Keycloak for both, or FAB for zero-dependency development.

Providers are selected at runtime via environment variables:
    APG_AUTH_PROVIDER   = keycloak | clerk | betterauth | fab | null
    APG_AUTHZ_PROVIDER  = spicedb | keycloak | clerk | betterauth | fab | null

The null provider approves all requests — useful for development only.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthProvider(Protocol):
    """Authentication provider — handles identity verification and session management.

    Implementations must be async throughout. Every external call must be
    protected by a circuit breaker and timeout.
    """

    async def authenticate(
        self,
        credentials: dict[str, Any],
    ) -> "AuthResult":
        """Verify credentials and return tokens.

        credentials: {username, password} OR {token} OR {code, redirect_uri} for OAuth
        """
        ...

    async def validate_token(self, token: str) -> "TokenPayload":
        """Validate an access token and return its claims.

        Raises AuthenticationError if token is invalid or expired.
        Should be cached — this is called on every API request.
        """
        ...

    async def refresh_token(self, refresh_token: str) -> "TokenPair":
        """Exchange a refresh token for a new access/refresh pair."""
        ...

    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        """Invalidate the session and blacklist the token."""
        ...

    async def create_user(self, user_data: dict[str, Any]) -> "AuthUser":
        """Create a new user account."""
        ...

    async def get_user(self, user_id: str) -> "AuthUser":
        """Retrieve user profile by ID."""
        ...

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> "AuthUser":
        """Update user profile fields."""
        ...

    async def delete_user(self, user_id: str) -> None:
        """Permanently delete a user account."""
        ...

    async def list_users(
        self, search: str | None = None, limit: int = 50, page: int = 1
    ) -> "UserList":
        """List users with optional search filter."""
        ...

    async def send_password_reset(self, email: str) -> None:
        """Send password reset email."""
        ...

    async def reset_password(self, token: str, new_password: str) -> None:
        """Complete password reset."""
        ...

    async def send_magic_link(self, email: str, redirect_url: str) -> None:
        """Send a magic link (passwordless) sign-in email."""
        ...

    async def verify_magic_link(self, token: str) -> "AuthResult":
        """Complete magic link sign-in."""
        ...

    async def get_oauth_authorization_url(
        self, provider: str, redirect_uri: str, state: str, scopes: list[str] | None = None
    ) -> str:
        """Build OAuth2 authorization URL for a social provider."""
        ...

    async def exchange_oauth_code(
        self, code: str, state: str, redirect_uri: str, provider: str
    ) -> "AuthResult":
        """Exchange OAuth2 code for tokens."""
        ...

    async def setup_mfa(self, user_id: str, mfa_type: str) -> "MFASetup":
        """Initiate MFA enrollment. mfa_type: totp | sms | email"""
        ...

    async def verify_mfa(self, user_id: str, code: str, session_token: str) -> "AuthResult":
        """Complete MFA verification during login."""
        ...

    async def disable_mfa(self, user_id: str, mfa_type: str) -> None:
        """Remove MFA method from user account."""
        ...

    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """List active sessions for a user."""
        ...

    async def revoke_session(self, session_id: str) -> None:
        """Revoke a specific session."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Return provider health status."""
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...


@runtime_checkable
class AuthzProvider(Protocol):
    """Authorization provider — handles permissions, roles, and resource access.

    Implementations may use RBAC (role-based), ABAC (attribute-based),
    or ReBAC (relationship-based, e.g. SpiceDB/Zanzibar) models.
    """

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        tenant_id: str = "default",
        resource_id: str | None = None,
        resource_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Return True if user has the named permission, optionally scoped to a resource."""
        ...

    async def check_resource_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        tenant_id: str = "default",
    ) -> bool:
        """ReBAC-style check: can user perform action on resource? (SpiceDB pattern)"""
        ...

    async def get_user_roles(self, user_id: str, tenant_id: str = "default") -> list[str]:
        """Return all roles assigned to a user in a tenant."""
        ...

    async def assign_role(
        self, user_id: str, role: str, tenant_id: str = "default", granted_by: str = "system"
    ) -> None:
        """Grant a role to a user."""
        ...

    async def revoke_role(
        self, user_id: str, role: str, tenant_id: str = "default", revoked_by: str = "system"
    ) -> None:
        """Remove a role from a user."""
        ...

    async def get_role_permissions(self, role: str, tenant_id: str = "default") -> list[str]:
        """Return all permissions granted by a role."""
        ...

    async def create_role(
        self, role: str, permissions: list[str], tenant_id: str = "default", description: str = ""
    ) -> dict[str, Any]:
        """Create a new role with permissions."""
        ...

    async def delete_role(self, role: str, tenant_id: str = "default") -> None:
        """Delete a role."""
        ...

    async def list_roles(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        """List all roles in a tenant."""
        ...

    async def write_relationship(
        self,
        resource_type: str,
        resource_id: str,
        relation: str,
        subject_type: str,
        subject_id: str,
    ) -> None:
        """Write a SpiceDB-style relationship tuple. No-op for RBAC providers."""
        ...

    async def delete_relationship(
        self,
        resource_type: str,
        resource_id: str,
        relation: str,
        subject_type: str,
        subject_id: str,
    ) -> None:
        """Delete a relationship tuple."""
        ...

    async def list_accessible_resources(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        tenant_id: str = "default",
    ) -> list[str]:
        """Return IDs of all resources user can perform action on."""
        ...

    async def bulk_check_permissions(
        self,
        user_id: str,
        checks: list[dict[str, Any]],
        tenant_id: str = "default",
    ) -> dict[str, bool]:
        """Batch permission check: {permission: bool, ...}"""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Return provider health status."""
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...


# ── Result types (used by all providers) ─────────────────────────────────────

from dataclasses import dataclass, field
from datetime import datetime


class AuthenticationError(Exception):
    """Raised on invalid credentials or expired token."""
    def __init__(self, msg: str, code: str = "authentication_failed") -> None:
        self.code = code
        super().__init__(msg)


class AuthorizationError(Exception):
    """Raised when a user lacks a required permission."""
    def __init__(self, msg: str, required_permission: str = "") -> None:
        self.required_permission = required_permission
        super().__init__(msg)


class ProviderNotImplementedError(NotImplementedError):
    """Raised when a provider does not support the requested feature."""


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600  # seconds
    scope: str = ""


@dataclass
class AuthResult:
    user: "AuthUser"
    tokens: TokenPair
    mfa_required: bool = False
    mfa_session_token: str | None = None


@dataclass
class TokenPayload:
    user_id: str
    email: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    tenant_id: str = "default"
    expires_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(self.expires_at.tzinfo) > self.expires_at


@dataclass
class AuthUser:
    id: str
    email: str
    username: str = ""
    first_name: str = ""
    last_name: str = ""
    is_active: bool = True
    is_email_verified: bool = False
    roles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    mfa_enabled: bool = False

    @property
    def display_name(self) -> str:
        name = f"{self.first_name} {self.last_name}".strip()
        return name or self.username or self.email


@dataclass
class MFASetup:
    mfa_type: str              # totp | sms | email
    secret: str | None = None  # TOTP secret (show to user once)
    qr_code_url: str | None = None
    backup_codes: list[str] = field(default_factory=list)
    session_token: str = ""


@dataclass
class UserList:
    users: list[AuthUser]
    total: int
    page: int
    limit: int
    has_more: bool = False
