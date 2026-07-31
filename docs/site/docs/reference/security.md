# Security

APG generated apps ship with security controls enabled by default. No option flags are required for basic hardening — everything activates via environment variables in production.

## Session authentication

The generated app uses Flask signed sessions. In development the session secret is auto-generated at startup. **In production you must set `APG_SECRET_KEY`.**

```bash
export APG_SECRET_KEY="$(openssl rand -hex 32)"
export APG_PRODUCTION=1
```

The login form at `/login` authenticates against the user list configured in `APG_AUTH_USERS`.

## API key authentication

Any request carrying a valid `Authorization: Bearer <key>` or `X-APG-Api-Key: <key>` header is authenticated as an API client:

```bash
export APG_API_KEY="$(openssl rand -hex 32)"
```

Without a key set, the API is open (useful for development).

## JWT authentication

The generated app accepts RS256 or HS256 JWTs:

```bash
# HS256 (shared secret)
export APG_JWT_SECRET="my-secret"

# RS256 (public key verification)
export APG_JWT_PUBLIC_KEY="$(cat public.pem)"
```

JWT claims are validated on every request. The `sub` claim is used as the user identity.

## Admin key

A separate admin key grants access to admin-only endpoints (e.g. `/admin/users`):

```bash
export APG_ADMIN_KEY="$(openssl rand -hex 32)"
```

## User list

For apps without an external identity provider, define users inline:

```bash
# Simple single user
export APG_AUTH_USERNAME=admin
export APG_AUTH_PASSWORD=changeme

# Multiple users (JSON array)
export APG_AUTH_USERS='[
  {"username":"alice","password_hash":"$2b$12$...","role":"admin"},
  {"username":"bob","password_hash":"$2b$12$...","role":"viewer"}
]'
```

Generate password hashes:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'mypassword', bcrypt.gensalt()).decode())"
```

## Security headers

All responses include:

- `Content-Security-Policy` with per-request nonce
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Strict-Transport-Security` (when `APG_PRODUCTION=1`)
- `Referrer-Policy: strict-origin-when-cross-origin`

## Rate limiting

Built-in rate limiter (no Redis required):

- Default: 100 requests per minute per IP
- Returns `429 Too Many Requests` when exceeded

Configurable via `APG_RATE_LIMIT` environment variable.

## CSRF protection

All state-changing form submissions (POST/PUT/PATCH/DELETE from the HTML UI) include a CSRF token validated server-side.

## Audit log

Every write operation (create, update, delete) is recorded in the audit log:

```bash
export APG_AUDIT_LOG_FILE="/var/log/apg/audit.jsonl"
```

Each entry contains: timestamp, user identity, entity name, record ID, operation, diff of changed fields.

## Column ACL (field-level access control)

Restrict which fields are visible per role:

```bash
export APG_FIELD_ACL='{"viewer": {"User": ["id","name","email"], "Order": ["id","status"]}}'
```

Fields not in the ACL list are stripped from responses for that role.

## Row ownership

When a record has an `owner_id` field, non-admin users can only read and modify their own records. Override with `APG_ADMIN_KEY`.

## Production checklist

| Variable | Required in prod | Purpose |
|----------|-----------------|---------|
| `APG_SECRET_KEY` | **Yes** | Session signing key |
| `APG_PRODUCTION` | **Yes** | Enables HSTS, cookie Secure flag |
| `APG_AUTH_USERS` | Yes (if no JWT/API key) | User credentials |
| `APG_AUDIT_LOG_FILE` | Recommended | Audit trail |
| `APG_API_KEY` | For API clients | Bearer token auth |
