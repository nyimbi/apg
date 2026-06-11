"""Auth hub provider factory.

Reads APG_AUTH_PROVIDER and APG_AUTHZ_PROVIDER from the environment and
instantiates the correct provider pair. Providers are singletons per process.

Supported combinations (mix and match):

    Auth + Authz from same provider:
        APG_AUTH_PROVIDER=keycloak   APG_AUTHZ_PROVIDER=keycloak
        APG_AUTH_PROVIDER=fab        APG_AUTHZ_PROVIDER=fab
        APG_AUTH_PROVIDER=clerk      APG_AUTHZ_PROVIDER=clerk
        APG_AUTH_PROVIDER=betterauth APG_AUTHZ_PROVIDER=betterauth

    Auth from one, fine-grained authz from SpiceDB (recommended for production):
        APG_AUTH_PROVIDER=keycloak   APG_AUTHZ_PROVIDER=spicedb
        APG_AUTH_PROVIDER=clerk      APG_AUTHZ_PROVIDER=spicedb
        APG_AUTH_PROVIDER=betterauth APG_AUTHZ_PROVIDER=spicedb

    Development (no external services):
        APG_AUTH_PROVIDER=null       APG_AUTHZ_PROVIDER=null
        APG_AUTH_PROVIDER=fab        APG_AUTHZ_PROVIDER=fab
"""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

_auth_provider: Any = None
_authz_provider: Any = None


def get_auth_provider() -> Any:
    """Return the singleton AuthProvider instance for this process."""
    global _auth_provider
    if _auth_provider is None:
        _auth_provider = _create_auth_provider()
    return _auth_provider


def get_authz_provider() -> Any:
    """Return the singleton AuthzProvider instance for this process."""
    global _authz_provider
    if _authz_provider is None:
        _authz_provider = _create_authz_provider()
    return _authz_provider


def reset_providers() -> None:
    """Reset provider singletons — used in tests."""
    global _auth_provider, _authz_provider
    _auth_provider = None
    _authz_provider = None


def _create_auth_provider() -> Any:
    name = os.environ.get("APG_AUTH_PROVIDER", "null").lower().strip()
    _log.info("Initialising auth provider: %s", name)

    if name == "keycloak":
        from .providers.keycloak_provider import KeycloakAuthProvider
        return KeycloakAuthProvider()

    if name in ("clerk",):
        from .providers.clerk_provider import ClerkAuthProvider
        return ClerkAuthProvider()

    if name in ("betterauth", "better_auth", "better-auth"):
        from .providers.betterauth_provider import BetterAuthProvider
        return BetterAuthProvider()

    if name in ("fab", "flask_appbuilder", "flask-appbuilder"):
        from .providers.fab_provider import FABAuthProvider
        return FABAuthProvider()

    if name in ("null", "dev", "test", ""):
        from .providers.null_provider import NullAuthProvider
        return NullAuthProvider()

    raise ValueError(
        f"Unknown APG_AUTH_PROVIDER={name!r}. "
        f"Valid options: keycloak, clerk, betterauth, fab, null"
    )


def _create_authz_provider() -> Any:
    name = os.environ.get("APG_AUTHZ_PROVIDER", "null").lower().strip()
    auth_name = os.environ.get("APG_AUTH_PROVIDER", "null").lower().strip()
    _log.info("Initialising authz provider: %s", name)

    if name == "spicedb":
        from .providers.spicedb_provider import SpiceDBAuthzProvider
        return SpiceDBAuthzProvider()

    if name == "keycloak":
        from .providers.keycloak_provider import KeycloakAuthProvider, KeycloakAuthzProvider
        # Reuse existing auth provider if it's also keycloak, else create new
        auth = get_auth_provider() if auth_name == "keycloak" else KeycloakAuthProvider()
        return KeycloakAuthzProvider(auth)

    if name in ("clerk",):
        from .providers.clerk_provider import ClerkAuthProvider, ClerkAuthzProvider
        auth = get_auth_provider() if auth_name == "clerk" else ClerkAuthProvider()
        return ClerkAuthzProvider(auth)

    if name in ("betterauth", "better_auth", "better-auth"):
        # BetterAuth doesn't have a separate authz provider — fall through to null
        # or use SpiceDB alongside BetterAuth
        _log.warning("BetterAuth has no standalone authz provider — using NullAuthzProvider")
        from .providers.null_provider import NullAuthzProvider
        return NullAuthzProvider()

    if name in ("fab", "flask_appbuilder", "flask-appbuilder"):
        from .providers.fab_provider import FABAuthzProvider
        return FABAuthzProvider()

    if name in ("null", "dev", "test", ""):
        from .providers.null_provider import NullAuthzProvider
        return NullAuthzProvider()

    raise ValueError(
        f"Unknown APG_AUTHZ_PROVIDER={name!r}. "
        f"Valid options: spicedb, keycloak, clerk, fab, null"
    )


def provider_info() -> dict[str, str]:
    """Return the configured provider names without instantiating."""
    return {
        "auth_provider": os.environ.get("APG_AUTH_PROVIDER", "null"),
        "authz_provider": os.environ.get("APG_AUTHZ_PROVIDER", "null"),
    }
