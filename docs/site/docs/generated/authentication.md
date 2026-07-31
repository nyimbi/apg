# Authentication

The generated app supports three authentication modes that can be combined.

## 1. Session (form login)

The HTML management UI at `/ui` uses server-signed Flask sessions. Navigate to `/login`, enter credentials, and a secure cookie is issued.

Configure users:

```bash
# Single user (simple)
export APG_AUTH_USERNAME=admin
export APG_AUTH_PASSWORD=changeme

# Multiple users (JSON)
export APG_AUTH_USERS='[{"username":"alice","password_hash":"$2b$12$...","role":"admin"}]'
```

Generate a bcrypt hash:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'mypassword', bcrypt.gensalt()).decode())"
```

## 2. API key (Bearer token)

For machine-to-machine access:

```bash
export APG_API_KEY="$(openssl rand -hex 32)"
```

Clients send:

```
Authorization: Bearer <key>
# or
X-APG-Api-Key: <key>
```

Without `APG_API_KEY` set, the API is unauthenticated (development mode).

## 3. JWT (token-based)

The generated app validates JWTs on every request.

```bash
# HS256
export APG_JWT_SECRET="my-shared-secret"

# RS256 (verify with public key only)
export APG_JWT_PUBLIC_KEY="$(cat public.pem)"
```

The `sub` claim is used as the authenticated user identity. The `role` claim (if present) is used for ACL decisions.

## Admin key

A separate key grants elevated access to admin-only routes:

```bash
export APG_ADMIN_KEY="$(openssl rand -hex 32)"
```

Pass it the same way as `APG_API_KEY`. An admin key bypasses row-ownership checks.

## Auth mode detection

```
GET /entities/Foo/config
```

Returns:

```json
{
  "mode": "api_key",
  "header": "Authorization: Bearer <key> or X-APG-API-Key"
}
```

Or `"mode": "open"` when no key is configured.

## Multi-user role model

| Role | Permissions |
|------|-------------|
| `admin` | Full CRUD + admin routes |
| `editor` | Create, read, update |
| `viewer` | Read-only |

Roles are assigned in `APG_AUTH_USERS` and enforced by the Column ACL middleware.

## Session security

- `SESSION_COOKIE_HTTPONLY = True`
- `SESSION_COOKIE_SAMESITE = "Lax"` (configurable via `APG_SESSION_COOKIE_SAMESITE`)
- `SESSION_COOKIE_SECURE = True` when `APG_PRODUCTION=1`
- Session fixation protection: new session ID issued after login
- Login throttle: 5 failed attempts per IP triggers a 60-second lockout
