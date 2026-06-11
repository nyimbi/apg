# Auth Hub — User Guide

**Version**: 1.1.0  
**Capability path**: `capabilities/common/auth_hub`  
**Copyright**: © 2025 Datacraft

---

## Overview

Auth Hub is the single entry point for all authentication and authorization in APG. It wraps multiple identity providers behind a unified facade so application code never imports a provider SDK directly — only `AuthHubService`.

The service enforces:
- Per-identity adaptive rate limiting on every authentication attempt
- Structured audit events on every mutating or sensitive operation
- Token validation caching to keep hot-path latency below 1 ms on warm requests
- Guard assertions (`guard_tenant_id`, `guard_non_empty_string`) at every public method boundary

---

## Installation and Setup

### 1. Environment variables

Set exactly two variables before starting the app:

```bash
export APG_AUTH_PROVIDER=keycloak      # or clerk | betterauth | fab | null
export APG_AUTHZ_PROVIDER=spicedb      # or keycloak | clerk | fab | null
```

The `null` provider is development-only. Using it in `APG_ENV=production` raises a `RuntimeError` at startup.

### 2. Provider-specific configuration

Refer to the provider-specific sections in README.md for required env vars per provider.

### 3. Instantiation

```python
from capabilities.common.auth_hub import AuthHubService

# Default — reads providers from environment
svc = AuthHubService()

# Explicit providers (testing)
svc = AuthHubService(
    auth_provider=my_auth_provider,
    authz_provider=my_authz_provider,
    tenant_id="acme",
)

# Tuned caching and rate limiting
svc = AuthHubService(
    token_cache_positive_ttl=120.0,   # cache valid tokens for 2 minutes
    token_cache_negative_ttl=30.0,    # cache invalid tokens for 30 seconds
    rate_limit_max_failures=3,        # lock after 3 consecutive failures
    rate_limit_lockout_seconds=600.0, # 10-minute lockout
)
```

---

## Authentication

### Password / username sign-in

```python
result = await svc.authenticate({"email": "alice@example.com", "password": "s3cr3t"})
access_token  = result.tokens.access_token
refresh_token = result.tokens.refresh_token
user_id       = result.user.id

if result.mfa_required:
    # Prompt user for TOTP code and call verify_mfa
    mfa_result = await svc.verify_mfa(user_id, totp_code, result.mfa_session_token)
```

### Token validation

```python
try:
    payload = await svc.validate_token(bearer_token)
    # payload.user_id, payload.email, payload.roles, payload.tenant_id
except AuthenticationError as exc:
    return 401, exc.code  # "token_invalid" | "token_expired" | "rate_limited"
```

Token results are cached by SHA-256 fingerprint:
- Valid tokens: cached for `token_cache_positive_ttl` (default 60 s)
- Invalid/revoked tokens: cached for `token_cache_negative_ttl` (default 10 s)

Logout evicts the token immediately:
```python
await svc.logout(access_token, refresh_token)
```

Force-evict from cache without a full logout (e.g., after a privilege change):
```python
await svc.invalidate_token_cache(access_token)
```

### Magic links (Clerk, BetterAuth)

```python
await svc.send_magic_link("alice@example.com", "https://app.example.com/auth/callback")
# User clicks link → browser POSTs token to your callback endpoint
result = await svc.verify_magic_link(magic_token)
```

### OAuth 2.0 (social login)

```python
# Step 1: redirect user to provider
url = await svc.get_oauth_url("github", "https://app/callback", state="csrf-token", scopes=["user:email"])
return redirect(url)

# Step 2: handle callback
result = await svc.exchange_oauth_code(code, state, redirect_uri, provider="github")
```

---

## Multi-Factor Authentication

### TOTP setup

```python
setup = await svc.setup_mfa("alice", mfa_type="totp")
# setup.secret → show to user once (QR code or manual entry)
# setup.qr_code_url → otpauth:// URI for QR rendering
# setup.backup_codes → one-time recovery codes, store encrypted
print(setup.qr_code_url)

# Verify enrollment
result = await svc.verify_mfa("alice", totp_code, setup.session_token)
```

### Disable MFA

```python
await svc.disable_mfa("alice", mfa_type="totp")
```

---

## Passkeys / WebAuthn

Passkeys require Keycloak 22+, Clerk, or Hanko as the auth provider. Other providers raise `ProviderNotImplementedError`.

### Registration flow

```python
# 1. Your frontend calls navigator.credentials.create() with a server-generated challenge
# 2. Frontend sends the attestation object to your API
# 3. Your API calls:
result = await svc.register_passkey(
    user_id="alice",
    credential_data=attestation_dict,  # decoded from browser response
    device_name="Alice's iPhone",
)
print(result["passkey_id"])
```

### Authentication flow

```python
# 1. Your frontend calls navigator.credentials.get()
# 2. Frontend sends assertion to your API
auth_result = await svc.verify_passkey_assertion("alice", assertion_dict)
# Returns full AuthResult — same as password login
```

### List registered devices

```python
passkeys = await svc.list_passkeys("alice")
for pk in passkeys:
    print(f"{pk['device_name']} — last used {pk['last_used_at']}")
```

---

## Authorization

### Permission check

```python
allowed = await svc.check_permission(
    user_id="alice",
    permission="payments:write",
    tenant_id="acme",  # omit to use service default tenant
)
```

### Permission inheritance

```python
# Define once at startup (or from DB config)
await svc.register_permission_hierarchy(
    "platform_admin",
    ["users:delete", "payments:write", "reports:read", "tenants:manage"],
)
await svc.register_permission_hierarchy(
    "tenant_admin",
    ["users:write", "payments:read", "reports:read"],
)

# Check resolves inheritance automatically
allowed = await svc.check_permission_with_inheritance("alice", "payments:write", tenant_id="acme")
# Returns True if alice has "platform_admin" OR "tenant_admin" OR direct "payments:write"
```

### ReBAC resource access (SpiceDB)

```python
await svc.write_relationship("document", "doc-123", "owner", "user", "alice")
await svc.write_relationship("document", "doc-123", "viewer", "user", "bob")

can_edit = await svc.check_resource_access("alice", "document", "doc-123", "edit")
can_view = await svc.check_resource_access("bob",  "document", "doc-123", "view")

all_my_docs = await svc.list_accessible_resources("alice", "document", "view", tenant_id="acme")
```

### Bulk permission check

```python
results = await svc.bulk_check_permissions(
    "alice",
    [
        {"permission": "payments:write"},
        {"permission": "reports:read"},
        {"permission": "users:delete"},
    ],
    tenant_id="acme",
)
# {"payments:write": True, "reports:read": True, "users:delete": False}
```

---

## Role Management

```python
# Create a role
await svc.create_role(
    "analyst",
    ["reports:read", "dashboards:read"],
    tenant_id="acme",
    description="Read-only analyst access",
)

# Assign / revoke
await svc.assign_role("alice", "analyst", tenant_id="acme", granted_by="admin-bob")
await svc.revoke_role("alice", "analyst", tenant_id="acme", revoked_by="admin-bob")

# Inspect
roles = await svc.get_user_roles("alice", tenant_id="acme")
perms = await svc.get_role_permissions("analyst", tenant_id="acme")
all_roles = await svc.list_roles(tenant_id="acme")
```

---

## Multi-Tenancy

### Tenant context manager

The cleanest way to scope a batch of operations to a single tenant without threading `tenant_id` through every call:

```python
async with svc.tenant_context("acme") as tsvc:
    roles  = await tsvc.get_user_roles("alice")
    perms  = await tsvc.get_role_permissions("admin")
    result = await tsvc.authenticate({"username": "alice", "password": "s3cr3t"})
```

The previous `tenant_id` is restored on exit, even if the block raises.

### Guard enforcement

`guard_tenant_id` rejects `None`, empty strings, and the literal `"default"` when `APG_ENV` is `production` or `staging`. This catches cross-tenant data leakage at the call site rather than at the database query.

---

## Identity Federation (Enterprise SSO)

```python
# Azure AD — accept an id_token from an enterprise customer's tenant
result = await svc.federate_user(
    external_token=azure_id_token,
    external_provider="azure-ad",
    tenant_id="enterprise-acme",
    auto_provision=True,  # create local user if not found
)

# Google Workspace
result = await svc.federate_user(
    external_token=google_id_token,
    external_provider="google-workspace",
    tenant_id="enterprise-acme",
)

# SAML 2.0 assertion (base64-encoded XML)
result = await svc.federate_user(
    external_token=saml_assertion_b64,
    external_provider="saml",
    tenant_id="enterprise-acme",
)
```

Supported providers for federation: Keycloak (Identity Brokering), Clerk (SAML SSO). Other providers raise `ProviderNotImplementedError`.

---

## Token Exchange (Service-to-Service)

Used when a backend service needs to call another service on behalf of a user without sharing the user's token:

```python
# Supported by Keycloak 20+ and Auth0 (RFC 8693)
service_token = await svc.exchange_token(
    subject_token=user_access_token,
    target_service="payments-service",
    scope=["payments:read"],  # narrow down scopes
)
# Use service_token.access_token in Authorization header for the downstream call
```

---

## Session Risk Scoring

```python
score = await svc.score_session_risk(
    user_id="alice",
    session_id="sess-abc123",
    event_context={
        "ip_address": "41.206.10.1",
        "country_code": "KE",
        "user_agent": "Mozilla/5.0 ...",
        "previous_ip": "41.206.10.1",   # same IP — low risk
        "action_type": "payment_initiate",
    },
)

match score["action"]:
    case "allow":
        pass  # proceed normally
    case "challenge":
        # Force step-up authentication
        return redirect(f"/auth/challenge?session={session_id}")
    case "revoke":
        await svc.revoke_session(session_id)
        raise AuthorizationError("Session revoked due to anomalous activity")
```

---

## Rate Limiting

Rate limiting is automatic — `authenticate` checks and records per-identity failures internally. No configuration required beyond the constructor parameters.

### Behavior

1. Successful `authenticate` → clears failure counter for that identity
2. Failed `authenticate` → increments counter; audit event emitted
3. Counter reaches `rate_limit_max_failures` (default 5) → identity locked for `rate_limit_lockout_seconds` (default 300 s)
4. Locked identity calling `authenticate` → `AuthenticationError(code="rate_limited")`

### Admin operations

```python
# Inspect
status = await svc.get_rate_limit_status("alice@example.com")
# {
#   "identity": "alice@example.com",
#   "failures": 5,
#   "locked": True,
#   "locked_until": 1234567890.0,
#   "seconds_remaining": 247,
# }

# Unlock after verifying identity via support channel
await svc.unlock_identity("alice@example.com", unlocked_by="support-agent-007")
```

---

## Flask Middleware

```python
from capabilities.common.auth_hub.middleware import (
    require_auth,
    require_permission,
    require_role,
    get_current_user,
    get_current_token_payload,
)

@app.get("/api/profile")
@require_auth
async def get_profile():
    user = get_current_user()        # TokenPayload
    payload = get_current_token_payload()
    return {"user_id": payload.user_id, "roles": payload.roles}

@app.post("/api/payments")
@require_permission("payments:write")
async def create_payment():
    ...

@app.delete("/api/users/<user_id>")
@require_permission("users:delete", resource_type="user", resource_id_param="user_id")
async def delete_user(user_id: str):
    ...

@app.get("/api/admin")
@require_role("admin")
async def admin_panel():
    ...
```

All decorators:
- Return `401` if no token present or token invalid
- Return `403` if permission/role not held
- Return `503` if the auth service is unreachable

---

## Health and Observability

```python
health = await svc.health_check()
# {
#   "status": "ok",           # "ok" | "degraded"
#   "auth_provider": {...},
#   "authz_provider": {...},
#   "config": {"auth": "keycloak", "authz": "spicedb"},
#   "token_cache": {
#       "entries": 142,
#       "expired_pending_eviction": 3,
#       "positive_ttl_seconds": 60.0,
#       "negative_ttl_seconds": 10.0,
#   },
#   "rate_limits": {
#       "tracked_identities": 8,
#       "locked_identities": 1,
#       "max_failures_before_lockout": 5,
#       "lockout_seconds": 300.0,
#   },
#   "permission_hierarchy": {
#       "registered_parents": 4,
#   },
# }

info = await svc.describe()
# Lists all enabled features

# Periodic cache maintenance (call from a background task every few minutes)
evicted = await svc.purge_token_cache()
_log.info("Evicted %d expired token cache entries", evicted)
```

---

## Error Reference

| Exception | When raised | Key attributes |
|-----------|-------------|----------------|
| `AuthenticationError` | Invalid credentials, expired token, rate-limited identity | `code: str` — `authentication_failed` \| `token_invalid` \| `token_expired` \| `rate_limited` |
| `AuthorizationError` | User lacks required permission | `required_permission: str` |
| `ProviderNotImplementedError` | Feature not supported by active provider | `str(exc)` — names a compatible provider |
| `ValueError` | Missing required argument (e.g., no email or username) | standard |

---

## Security Checklist

- [ ] Set `APG_ENV=production` in production deployments — this enables dev-provider guard and production-only validations
- [ ] Use `APG_AUTH_PROVIDER=keycloak` or `clerk` in production; never `null`
- [ ] Run `await svc.purge_token_cache()` every 5 minutes from a background task to prevent unbounded memory growth
- [ ] Plug in a real `audit_sink` that forwards to your SIEM (Datadog, Splunk, CloudWatch) — `_log.info` is not a compliant audit trail
- [ ] Set `rate_limit_max_failures=3` and `rate_limit_lockout_seconds=900` (15 min) in consumer-facing deployments (NIST 800-63B recommendation)
- [ ] Register `permission_hierarchy` from a database config on startup — hardcoded hierarchies make permission changes require redeploys
- [ ] Use `tenant_context()` when processing batch jobs or background tasks that span multiple tenants — prevents `tenant_id="default"` bleed

---

## Testing

```python
from capabilities.common.auth_hub.providers.null_provider import NullAuthProvider, NullAuthzProvider
from capabilities.common.auth_hub.service import AuthHubService

async def test_permission_hierarchy():
    svc = AuthHubService(
        auth_provider=NullAuthProvider(),
        authz_provider=NullAuthzProvider(),
        tenant_id="test",
    )
    await svc.register_permission_hierarchy("admin", ["payments:write", "reports:read"])
    # NullAuthzProvider grants all permissions, so this is a sanity check for the call path
    allowed = await svc.check_permission_with_inheritance("user-1", "payments:write")
    assert allowed is True

async def test_rate_limiting():
    svc = AuthHubService(
        auth_provider=NullAuthProvider(),
        authz_provider=NullAuthzProvider(),
        rate_limit_max_failures=2,
        rate_limit_lockout_seconds=60.0,
    )
    # NullAuthProvider always succeeds, so manually record failures
    await svc._record_auth_failure("eve@example.com")
    await svc._record_auth_failure("eve@example.com")
    status = await svc.get_rate_limit_status("eve@example.com")
    assert status["locked"] is True

    await svc.unlock_identity("eve@example.com", unlocked_by="admin")
    status = await svc.get_rate_limit_status("eve@example.com")
    assert status["locked"] is False
```

Run tests:
```bash
uv run pytest -vxs tests/ci/test_auth_hub.py
```

---

*© 2025 Datacraft — www.datacraft.co.ke*
