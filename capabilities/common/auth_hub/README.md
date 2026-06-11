# Auth Hub (`auth_hub`)

Interchangeable authentication + authorization adapter. Switch providers without changing application code.

## Supported Providers

| Provider | Auth | Authz | Notes |
|----------|------|-------|-------|
| `keycloak` | ✓ | ✓ | Self-hosted OIDC/OAuth2 + Keycloak Authorization Services |
| `spicedb` | ✗ | ✓ | Google Zanzibar-style ReBAC — pair with any auth provider |
| `clerk` | ✓ | ✓ | Hosted auth-as-a-service. RBAC via user metadata. |
| `betterauth` | ✓ | ✗ | TypeScript service via HTTP proxy. Pair with SpiceDB for authz. |
| `fab` | ✓ | ✓ | Flask-AppBuilder built-in. Zero dependencies. |
| `null` | ✓ | ✓ | Dev/test only — all auth succeeds, all permissions granted. |

## Configuration

```bash
# Auth + Authz from same provider
APG_AUTH_PROVIDER=keycloak   APG_AUTHZ_PROVIDER=keycloak
APG_AUTH_PROVIDER=fab        APG_AUTHZ_PROVIDER=fab     # zero-dependency dev

# Best production combo: Clerk (hosted, developer UX) + SpiceDB (fine-grained)
APG_AUTH_PROVIDER=clerk      APG_AUTHZ_PROVIDER=spicedb

# Self-hosted everything
APG_AUTH_PROVIDER=keycloak   APG_AUTHZ_PROVIDER=spicedb

# Development
APG_AUTH_PROVIDER=null       APG_AUTHZ_PROVIDER=null
```

### Keycloak
```bash
APG_KEYCLOAK_URL=https://auth.example.com
APG_KEYCLOAK_REALM=apg
APG_KEYCLOAK_CLIENT_ID=apg-backend
APG_KEYCLOAK_CLIENT_SECRET=<secret>
APG_KEYCLOAK_ADMIN_USER=admin
APG_KEYCLOAK_ADMIN_PASS=<password>
```

### SpiceDB
```bash
APG_SPICEDB_URL=grpc://spicedb:50051
APG_SPICEDB_TOKEN=<pre-shared-key>
```

### Clerk
```bash
APG_CLERK_SECRET_KEY=sk_live_...
APG_CLERK_PUBLISHABLE_KEY=pk_live_...
```

### BetterAuth (Node.js service)
```bash
APG_BETTERAUTH_URL=http://localhost:3001
APG_BETTERAUTH_SECRET=<shared-secret>
```

## Usage

### Service layer

```python
from capabilities.common.auth_hub import AuthHubService

svc = AuthHubService()  # reads APG_AUTH_PROVIDER from env

# Authenticate
result = await svc.authenticate({"username": "alice", "password": "s3cr3t"})
print(result.tokens.access_token)

# Validate a token on every request
payload = await svc.validate_token(request.headers["Authorization"].split()[1])
print(payload.user_id, payload.roles)

# Check permission
allowed = await svc.check_permission(payload.user_id, "payments:write", tenant_id="acme")

# SpiceDB fine-grained access
allowed = await svc.check_resource_access("alice", "document", "doc-123", "edit")

# Write a relationship (SpiceDB only, no-op on others)
await svc.write_relationship("document", "doc-123", "owner", "user", "alice")
```

### Flask middleware

```python
from capabilities.common.auth_hub.middleware import require_auth, require_permission

@app.get("/api/accounts")
@require_auth
async def list_accounts():
    from capabilities.common.auth_hub.middleware import get_current_user
    user = get_current_user()
    ...

@app.delete("/api/users/<user_id>")
@require_permission("users:delete", resource_type="user", resource_id_param="user_id")
async def delete_user(user_id: str):
    ...
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/sign-in` | Authenticate (password or token) |
| POST | `/api/auth/sign-out` | Invalidate session |
| POST | `/api/auth/token/refresh` | Refresh access token |
| POST | `/api/auth/token/validate` | Validate token |
| GET | `/api/auth/users` | List users |
| POST | `/api/auth/users` | Create user |
| GET | `/api/auth/users/<id>` | Get user |
| PATCH | `/api/auth/users/<id>` | Update user |
| DELETE | `/api/auth/users/<id>` | Delete user |
| POST | `/api/auth/password/reset-request` | Send reset email |
| POST | `/api/auth/magic-link/send` | Send magic link |
| GET | `/api/auth/oauth/authorize` | Get OAuth URL |
| POST | `/api/auth/oauth/callback` | Exchange OAuth code |
| POST | `/api/auth/users/<id>/mfa/setup` | Set up MFA |
| POST | `/api/auth/mfa/verify` | Verify MFA code |
| GET | `/api/auth/roles` | List roles |
| POST | `/api/auth/users/<id>/roles` | Assign role |
| DELETE | `/api/auth/users/<id>/roles/<role>` | Revoke role |
| POST | `/api/auth/permissions/check` | Check permission |
| POST | `/api/auth/permissions/bulk-check` | Bulk permission check |
| POST | `/api/auth/relationships` | Write SpiceDB relationship |
| DELETE | `/api/auth/relationships` | Delete relationship |
| GET | `/api/auth/health` | Health check |
| GET | `/api/auth/info` | Provider info |

## Provider Feature Matrix

| Feature | keycloak | clerk | betterauth | fab | null |
|---------|---------|-------|-----------|-----|------|
| Username/password | ✓ | ✗* | ✓ | ✓ | ✓ |
| Token validation | ✓ | ✓ | ✓ | ✓ | ✓ |
| Magic links | ✗ | ✓ | ✓ | ✗ | ✓ |
| Social OAuth | ✓ | ✓* | ✓ | ✓ | ✓ |
| MFA/TOTP | ✓ | ✓ | ✓ | ✗ | ✓ |
| Fine-grained authz | ✓† | RBAC | RBAC | RBAC | ✓ |
| Relationship-based | ✗ | ✗ | ✗ | ✗ | ✓ |
| Self-hosted | ✓ | ✗ | ✓ | ✓ | ✓ |

*Clerk password/OAuth handled by Frontend SDK
†Pair with SpiceDB for true fine-grained authz
