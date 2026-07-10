# Generated App Runtime Security Baseline — 2025/2026

**Scope**: What the app a code generator *emits* must do out-of-the-box to meet or exceed world-class standards. Generator-internal quality is out of scope. All analysis assumes a stdlib-constrained Flask app (no third-party security packages in the emitted app).

**Date**: 2026-07-10
**Status**: Complete (single-pass deep research)

---

## 1. Executive Summary

The 2025/2026 world-class baseline for a generated/scaffolded web application is defined by three converging sources:

1. **OWASP ASVS 5.0** (released May 2025) — 17-chapter, ~350-requirement specification, organized across three assurance levels. L1 = basic, checklist-detectable. L2 = standard commercial app. L3 = high-security/financial.
2. **OWASP Cheat Sheet Series** (continuously updated 2025) — specific implementation guidance for session management, authentication, password storage, logging.
3. **Framework defaults from Django, Rails 8, Laravel 12, Phoenix** — the actual competitive bar, i.e., what a peer framework ships to every developer without configuration.

APG currently implements security headers (CSP, X-Frame-Options, etc.), CSRF tokens on session forms, and hardened session cookies. This puts it at roughly ASVS L1 on web-frontend security (V3) but significantly below L1 on authentication hardening (V6), session management (V7), logging (V16), and production ops.

**The gaps that matter most** (ranked by exploitability and peer-framework coverage):

| Priority | Gap | ASVS Ref | Complexity |
|---|---|---|---|
| P0 | Password hashing with KDF (not raw SHA/MD5) | V6 implied, Password Storage CS | Low — stdlib only |
| P0 | Session fixation: regenerate session ID on login | 7.2.4 (L1) | Low |
| P0 | Generic error messages on auth failures | V6.3.8 (L1) | Low |
| P0 | Timing-safe hash comparison (`hmac.compare_digest`) | Auth CS | Low |
| P1 | Login rate limiting / per-IP throttle | 6.1.1, 6.3.1 (L1) | Medium — in-memory OK for single-process |
| P1 | Request size limit (`MAX_CONTENT_LENGTH`) | — | Trivial |
| P1 | Health endpoints (`/healthz`, `/readyz`) | Ops baseline | Low |
| P1 | Structured JSON logging with security event fields | 16.2.1–16.3.3 (L2) | Medium |
| P2 | Request ID propagation (`X-Request-ID`) | Ops baseline | Low |
| P2 | Inactivity session timeout (server-enforced) | 7.3.1 (L2) | Low |
| P2 | Graceful degradation / fail-secure error handler | 16.5.2–16.5.3 (L2) | Medium |
| P2 | Password length minimum (8, recommended 15) | 6.2.1 (L1) | Trivial |
| P3 | Breached-password check against top-N list | 6.2.4 (L1) | Medium — needs bundled wordlist |
| P3 | Account lockout with observable counter | 6.3.1 (L1) | Medium |

---

## 2. OWASP ASVS 5.0 — Applicable Requirements

### Chapter Structure (v5.0, released May 2025)

| Chapter | Title | Relevance to generated Flask app |
|---|---|---|
| V3 | Web Frontend Security | Security headers (already done) |
| V6 | Authentication | Password policy, rate limiting, error messages |
| V7 | Session Management | Session token generation, fixation, timeouts |
| V11 | Cryptography | KDF algorithm selection |
| V12 | Secure Communication | TLS enforcement (deploy-time concern, not app-time) |
| V13 | Configuration | Secret management, no debug in prod |
| V16 | Security Logging & Error Handling | What to log and how |

### V6 Authentication — Key L1/L2 Requirements

| Req | Level | Requirement |
|---|---|---|
| 6.1.1 | L1 | Controls such as rate limiting, anti-automation, and adaptive response are used to defend against credential stuffing and password brute force. Must be documented and configured. |
| 6.2.1 | L1 | User-set passwords are at least 8 characters; 15+ strongly recommended. |
| 6.2.4 | L1 | Passwords checked against at least the top 3,000 passwords matching the application's password policy. |
| 6.2.5 | L1 | Any composition of characters is permitted; no mandatory complexity rules. |
| 6.2.6 | L1 | Password fields use `type="password"` to mask entry. |
| 6.2.7 | L1 | Allow paste functionality (password manager support). |
| 6.2.8 | L1 | Verify password exactly as received — no truncation, no case transformation. |
| 6.2.9 | L2 | Permit passwords of at least 64 characters. |
| 6.2.12 | L2 | Check passwords against a set of known breached passwords. |
| 6.3.1 | L1 | Implement controls against credential stuffing and brute force per the documentation. |
| 6.3.2 | L1 | Default accounts (root, admin, sa) are absent or disabled. |
| 6.3.8 | L1 | Valid users cannot be deduced from failed authentication challenges. (Generic error messages.) |

### V7 Session Management — Key L1/L2 Requirements

| Req | Level | Requirement |
|---|---|---|
| 7.2.1 | L1 | All session token verification performed by a trusted backend service (not client-side). |
| 7.2.2 | L1 | Dynamically generated tokens; no static secrets used as session identifiers. |
| 7.2.3 | L1 | Session tokens possess at least 128 bits of entropy via CSPRNG. |
| 7.2.4 | L1 | **New session token generated on user authentication; previous token terminated.** (Session fixation defense.) |
| 7.3.1 | L2 | Inactivity timeout: re-authentication enforced per risk analysis. |
| 7.3.2 | L2 | Absolute maximum session lifetime enforced. |
| 7.4.1 | L1 | Session termination invalidates the token/session on the server side. |
| 7.4.2 | L1 | Terminating a user account disallows all active sessions. |

**Session fixation (7.2.4) is L1.** This is the highest-priority session gap for APG.

### V16 Security Logging & Error Handling — Key L2 Requirements

All V16 requirements are L2 (no L1 requirements). However, L2 is the "standard commercial application" bar that all frameworks aspire to:

| Req | Requirement |
|---|---|
| 16.2.1 | Each log entry captures: when, where, who, what. |
| 16.2.2 | All logging components use synchronized time sources; UTC timestamps preferred. |
| 16.2.4 | Logs in a common parseable format (structured/JSON preferred). |
| 16.2.5 | Sensitive data (credentials, tokens) hashed/masked in logs. |
| 16.3.1 | All authentication operations logged — successful and failed — with auth type metadata. |
| 16.3.2 | Failed authorization attempts logged. |
| 16.3.3 | Attempts to bypass security controls (input validation failures, anti-automation) logged. |
| 16.3.4 | Unexpected errors and security control failures logged. |
| 16.4.1 | Log output encoded to prevent log injection. |
| 16.5.1 | Generic message returned on unexpected/security-sensitive errors (no stack traces, queries, keys). |
| 16.5.2 | Application continues operating securely when external resource access fails (circuit breaker / graceful degradation). |
| 16.5.3 | Application fails secure; no fail-open conditions. |
| 16.5.4 | (L3) Last-resort error handler catches all unhandled exceptions. |

---

## 3. Password Storage Best Practice (2025/2026)

### Algorithm Hierarchy (OWASP Password Storage Cheat Sheet)

| Rank | Algorithm | Recommended Parameters | Notes |
|---|---|---|---|
| 1 | **Argon2id** | m=19456 (19 MiB), t=2, p=1 OR m=47104 (46 MiB), t=1, p=1 | PHC winner; memory-hard. Not in stdlib. |
| 2 | **scrypt** | N=2^17 (131072), r=8, p=1 | Available in `hashlib.scrypt()` (Python 3.6+, requires OpenSSL). |
| 3 | **bcrypt** | work factor ≥ 10; max 72 bytes input | Not in stdlib. |
| 4 | **PBKDF2-HMAC-SHA256** | 600,000+ iterations | FIPS-140 compliant. In stdlib via `hashlib.pbkdf2_hmac()`. |
| 4 | **PBKDF2-HMAC-SHA512** | 210,000+ iterations | In stdlib. |

### Python Stdlib Implementation

**Option A — scrypt** (preferred stdlib choice; memory-hard):

```python
import hashlib, os, base64

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=2**17,   # CPU/memory cost; OWASP minimum 2^17
        r=8,        # block size
        p=1,        # parallelism
        dklen=64,
    )
    return f"scrypt$131072$8$1${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def verify_password(password: str, stored: str) -> bool:
    parts = stored.split("$")
    n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
    salt = base64.b64decode(parts[4])
    stored_dk = base64.b64decode(parts[5])
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=64)
    return hmac.compare_digest(dk, stored_dk)
```

**Option B — PBKDF2-HMAC-SHA256** (fallback if OpenSSL unavailable):

```python
import hashlib, os, base64, hmac

PBKDF2_ITERATIONS = 600_000

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2$sha256${PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def verify_password(password: str, stored: str) -> bool:
    parts = stored.split("$")
    algo, iterations = parts[1], int(parts[2])
    salt = base64.b64decode(parts[3])
    stored_dk = base64.b64decode(parts[4])
    dk = hashlib.pbkdf2_hmac(algo, password.encode(), salt, iterations)
    return hmac.compare_digest(dk, stored_dk)
```

**Critical**: Always use `hmac.compare_digest()` (or `secrets.compare_digest()`, identical) for hash comparison. Standard `==` short-circuits and leaks timing information that allows byte-by-byte guessing.

**NIST SP 800-63B Rev 4 (2024)** aligns with OWASP: minimum 8 characters, support 64+ characters, no mandatory complexity rules, no periodic rotation unless compromised, screen against breached-password lists.

---

## 4. Login Hardening

### Rate Limiting Architecture

**The problem with in-memory counters**: each worker process has its own dict. With N workers, effective limit = configured_limit × N. For single-process dev servers this is fine; for production with gunicorn/uwsgi it breaks.

**OWASP recommendation (6.1.1, 6.3.1)**: rate limiting must be *documented* and *consistent*. No specific algorithm mandated.

**Patterns (best to worst for stdlib-only generated apps)**:

| Pattern | Implementation | Pros | Cons |
|---|---|---|---|
| Per-IP fixed window (in-memory) | `dict[ip] = (count, window_start)` | Zero dependencies | Breaks with multiple workers; memory leak risk |
| Per-user fixed window (in-memory) | `dict[username] = (count, window_start)` | Survives IP rotation/VPN | Enables username enumeration DoS |
| Combined per-IP + per-user | Both dicts | More robust | Double memory; still per-process |
| Progressive delay | Exponential sleep on failures | No lockout DoS; graceful | Holds worker threads |
| Flask-Limiter + Redis | External dependency | Production-grade | Breaks stdlib-only constraint |

**Recommended for APG-generated apps**: Per-IP fixed window in-memory as the minimum, with a clearly generated `# TODO: replace with Redis-backed limiter in production` comment. The generated app must not ship with *no* rate limiting — that is a ASVS L1 failure.

**Suggested defaults**:
- Max 5 login failures per IP per 15-minute window before returning 429
- Lock window resets on successful login from that IP
- Account lockout (softer): after 10 failures on a specific username, require a CAPTCHA or notify admin (do not hard-lock to avoid DoS)
- Progressive delay: 0, 0, 0, 1s, 2s, 4s, 8s per-IP failure sequence

### Session Fixation Defense (ASVS 7.2.4, L1)

**What it is**: An attacker pre-sets a known session ID, lures the victim to authenticate, then hijacks the now-authenticated session.

**Defense**: Immediately after authentication succeeds, generate a new session ID and invalidate the old one.

In Flask with `flask.session`:

```python
from flask import session
import secrets

def login_user(user):
    # Copy any pre-auth session data to preserve across regeneration
    old_data = dict(session)
    session.clear()  # Invalidates the old session cookie
    session.update(old_data)
    session["user_id"] = user.id
    session["authenticated_at"] = time.time()
    # Flask will issue a new signed cookie with a new value
```

Note: Flask's signed cookie sessions don't have a server-side session store, so "regeneration" means clearing and reissuing the cookie. For reference-token (server-side) sessions, explicitly delete the old session record from the store and create a new one.

### Generic Error Messages

ASVS 6.3.8 (L1): Do not distinguish between "wrong username" and "wrong password."

**Correct**: `"Invalid username or password."`  
**Incorrect** (fail): `"User 'alice' does not exist."` / `"Incorrect password for alice."`

This applies to both the HTTP response body and any HTTP status codes (both cases must return 200 with the error in the body, or both return 401 — never 404 for missing user vs 401 for wrong password).

### Timing Attack Defense

On a login attempt where the user does not exist, the app must still perform a dummy password hash comparison to consume equivalent time. Without this, response time leaks whether the username is valid.

```python
DUMMY_HASH = hash_password("this_is_a_dummy_password_for_timing")

def authenticate(username, password):
    user = db.get_user(username)
    if user is None:
        verify_password(password, DUMMY_HASH)  # consume time
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
```

---

## 5. Production Ops Baseline

### Health Endpoints

Kubernetes and all cloud-native deployment targets expect:

| Endpoint | Purpose | Correct behavior |
|---|---|---|
| `GET /healthz` or `/health/live` | **Liveness**: is the process responsive? | Returns 200 if the process can handle requests. Never fail because a *dependency* is down — that causes crash-loop restarts. |
| `GET /readyz` or `/health/ready` | **Readiness**: can the process serve traffic? | Returns 200 only if all required dependencies (DB, cache) are reachable. Return 503 if not. |
| `GET /health` | Combined (simpler apps) | Acceptable; document what it checks. |

**Critical design rule**: Liveness and readiness must be separate. Failing liveness because the database is down causes Kubernetes to restart the pod in a loop, making the outage worse.

Response format:

```json
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "cache": "degraded"
  },
  "version": "1.2.3",
  "timestamp": "2026-07-10T12:00:00Z"
}
```

### Structured JSON Logging

OWASP Logging Cheat Sheet + ASVS 16.2.x mandates structured, machine-parseable log output. The minimum required fields per security event:

**When**: `timestamp` (ISO-8601 UTC), `event_id` (UUID)  
**Where**: `app_name`, `app_version`, `endpoint` (URL path), `http_method`  
**Who**: `source_ip`, `user_id` (if authenticated), `session_id` (hashed — never raw), `request_id`  
**What**: `event_type` (enum: `auth.login.success`, `auth.login.failure`, `session.created`, `session.destroyed`, `authz.denied`, `security.rate_limit_hit`, etc.), `result` (`success`|`failure`|`error`), `reason` (non-sensitive explanation)

```python
import json, logging, time, uuid

class SecurityLogger:
    def __init__(self, app_name: str, app_version: str):
        self._logger = logging.getLogger("security")
        self._app_name = app_name
        self._app_version = app_version

    def log_event(self, event_type: str, result: str, **kwargs):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_id": str(uuid.uuid4()),
            "app": self._app_name,
            "version": self._app_version,
            "event_type": event_type,
            "result": result,
            **kwargs,
        }
        self._logger.info(json.dumps(entry))
```

**Events that must be logged**:
- `auth.login.success` — user_id, source_ip, request_id
- `auth.login.failure` — username (hashed), source_ip, reason
- `auth.logout` — user_id, session_id (hashed)
- `auth.rate_limit` — source_ip, endpoint
- `session.created` — user_id
- `session.fixation_defense` — old_session_id (hashed), new_session_id (hashed)
- `authz.denied` — user_id, resource, action
- `security.validation_failure` — field, input_length (not input value)

**Never log**: plaintext passwords, session tokens, CSRF tokens, API keys, or PII beyond what's minimally necessary.

### Request ID Propagation

The `X-Request-ID` header is the de-facto standard (Heroku-originated, now universal). Pattern:

1. On incoming request: if `X-Request-ID` header present and valid UUID, use it; otherwise generate a new UUID4.
2. Store in `flask.g.request_id` for the duration of the request.
3. Include in all log entries as `request_id`.
4. Echo back in the response as `X-Request-ID`.

This enables correlating a single user-visible error ID with server-side log entries across all services.

### Request Size Limits

Flask does not set `MAX_CONTENT_LENGTH` by default. An unset value allows clients to exhaust server memory with a single large request.

**Generated app must set**:

```python
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MiB (adjust per use case)
```

Flask returns HTTP 413 automatically when this is exceeded.

### Graceful Degradation (ASVS 16.5.2–16.5.3)

The generated app should:
- Wrap database calls in try/except; return 503 on DB failure, not 500 with stack trace
- Never expose stack traces, query text, or internal paths in HTTP responses
- Register a Flask `@app.errorhandler(Exception)` that logs the full exception internally and returns a generic JSON/HTML error page externally

```python
@app.errorhandler(Exception)
def handle_unexpected(e):
    security_logger.log_event("error.unhandled", "error",
        error_type=type(e).__name__,
        request_id=g.get("request_id"))
    return {"error": "An unexpected error occurred."}, 500
```

---

## 6. Competitive Bar — Framework Default Features

| Feature | Django 5.x | Rails 8.x | Laravel 12 | Phoenix 1.8 | APG (current) |
|---|---|---|---|---|---|
| **CSRF protection** | Default on (`CsrfViewMiddleware`) | Default on (`protect_from_forgery`) | Default on (middleware) | Default on (`:protect_from_forgery` plug) | Done (session forms) |
| **Session cookie HttpOnly** | Yes | Yes | Yes | Yes | Done |
| **Session cookie SameSite** | `Lax` | `Lax` | `Lax` | `Lax` | Done |
| **Session cookie Secure** | Production-enforced via `SESSION_COOKIE_SECURE=True` | `force_ssl=true` in prod | `secure=true` config | `https_only: true` in prod | Done |
| **Security headers** | `SECURE_BROWSER_XSS_FILTER`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `HSTS` | `X-Frame-Options: SAMEORIGIN`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `HSTS` | `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `HSTS` | `put_secure_browser_headers` plug | Done (CSP, headers) |
| **Password hashing** | PBKDF2-SHA256 (600k iter) + Argon2 optional | bcrypt (cost 12) via `has_secure_password` | bcrypt or Argon2 via `Hash` facade | bcrypt via `Bcrypt` | **MISSING** |
| **Password policy validators** | 4 built-in validators (length, common, similarity, numeric) | None built-in (rely on gems) | None built-in | None built-in | **MISSING** |
| **Login rate limiting** | django-axes (third-party, common) | `rate_limit` in Rails 7.2+ (built-in!) | Throttle middleware built-in | `Hammer` (third-party) | **MISSING** |
| **Session fixation defense** | Yes — `django.contrib.auth.login()` calls `cycle_key()` | Yes — Rails rotates session on login | Yes — `session.regenerate_token` | Manual — must call `configure_session` | **MISSING** |
| **Generic auth error messages** | Yes (default login view) | Yes (default) | Yes (default) | Manual | **MISSING** |
| **Request size limits** | `DATA_UPLOAD_MAX_MEMORY_SIZE` = 2.5 MiB default | `config.max_param_bytes_size` | `post_max_size` config | `max_length` in Plug | **MISSING** |
| **Health endpoints** | None built-in | None built-in | None built-in | None built-in | **MISSING** |
| **Structured logging** | Django logging config; no default JSON format | Lograge (third-party) commonly | No default JSON | Logger (text format) | **MISSING** |
| **Request ID propagation** | None built-in | `X-Request-Id` header set by default! | None built-in | None built-in | **MISSING** |
| **Inactivity timeout** | `SESSION_COOKIE_AGE` = 2 weeks (not activity-based) | `config.session_store` — no default idle timeout | Session `lifetime` = 120 min | No default | **MISSING** |
| **Secret key validation** | Warns on weak/default key | Raises on missing `secret_key_base` | Raises on missing `APP_KEY` | Raises on missing key | Unknown |
| **Debug-mode protection** | `DEBUG=False` required; raises in prod | `config.consider_all_requests_local = false` | `APP_DEBUG=false` | `config :phoenix, :debug_errors, false` | Unknown |

**Rails 8 is the most notable**: it ships rate limiting built-in (Rails 7.2+), sets `X-Request-Id` by default, and in 8.2+ switches CSRF to `Sec-Fetch-Site` header verification (browser-native, no token needed for modern browsers). This is where the bar is moving.

---

## 7. Prioritized Recommendations for APG-Generated Flask Apps

### Tier 1 — Must Ship (ASVS L1 gaps, exploitable day-one)

1. **Password hashing with KDF**
   - Default: `hashlib.scrypt(N=131072, r=8, p=1)` with 16-byte random salt
   - Fallback (if OpenSSL not available): `hashlib.pbkdf2_hmac("sha256", ..., 600_000)`
   - Store as encoded string with algorithm/params embedded (PHC string format)
   - Use `hmac.compare_digest()` for all hash comparisons

2. **Session fixation defense**
   - On successful login: `session.clear()` then repopulate + set `user_id`
   - Flask's signed cookie model: clearing + reissuing achieves equivalent protection
   - For reference-token sessions: delete old record, create new one

3. **Generic authentication error messages**
   - Single message for all auth failures: `"Invalid username or password."`
   - Same HTTP status (401 or 200 redirect) regardless of whether username exists

4. **Timing-safe comparison + dummy hash**
   - `hmac.compare_digest()` for all hash comparisons
   - Run dummy `verify_password()` when user not found to equalize timing

5. **Request size limit**
   - `app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024`

6. **Password length validation**
   - Minimum 8 characters (15 recommended) — ASVS 6.2.1 (L1)
   - Maximum 1024 characters (prevent denial-of-service via KDF exhaustion)
   - No truncation — validate and reject, do not silently truncate

### Tier 2 — Should Ship (competitive parity, L2 or strong L1)

7. **In-memory login rate limiter**
   - Per-IP: 5 failures per 15-minute window → 429 Too Many Requests
   - Thread-safe counter using `threading.Lock()`
   - Comment in generated code: replace with Redis-backed limiter for multi-process deployments

8. **Health endpoints**
   - `GET /healthz` → liveness (process alive, no dep checks), returns `{"status": "ok"}`
   - `GET /readyz` → readiness (DB ping), returns 503 with reason if not ready
   - Protected from auth middleware

9. **Security event logging**
   - Structured JSON logger for: login success/failure, session create/destroy, auth errors, rate limit triggers
   - Fields: timestamp (UTC), event_type, result, source_ip, user_id (hashed), request_id
   - Never log credentials or raw session tokens

10. **Request ID propagation**
    - Middleware: accept `X-Request-ID` header or generate UUID4
    - Store in `flask.g.request_id`
    - Include in all log entries and echo in response headers

11. **Server-enforced session inactivity timeout**
    - Store `last_active` in session; check on each request
    - Default: 30 minutes inactivity → clear session, redirect to login

12. **Generic error handler**
    - `@app.errorhandler(Exception)`: log internally, return generic message externally
    - No stack traces, query text, or file paths in HTTP responses

### Tier 3 — Nice to Have (L2 beyond competitive bar)

13. **Common password blocklist**
    - Bundle a compressed list of top-3000 passwords (ASVS 6.2.4, L1 literal requirement)
    - Check on registration and password change
    - ~15 KB compressed

14. **Progressive login delay**
    - After N failures from same IP, add exponential sleep: 1s, 2s, 4s, 8s...
    - Complements rate limiting without lockout DoS risk

15. **Debug mode assertion**
    - On startup in production: if `DEBUG=True` and `FLASK_ENV=production`, raise `RuntimeError`

---

## 8. Open Questions

1. **scrypt availability**: `hashlib.scrypt()` requires Python compiled with OpenSSL. Is this guaranteed in APG's target deployment environments? If not, PBKDF2 fallback logic must be automatic, not manual.

2. **In-memory rate limiter race conditions**: Threading.Lock is sufficient for CPython (GIL) but the counter still leaks across worker processes. Should APG emit a warning banner on startup when running with multiple workers and no Redis configured?

3. **Cookie prefix (`__Host-`)**: OWASP Session Management Cheat Sheet recommends the `__Host-` cookie prefix (forces `Secure`, no `Domain`, `Path=/`). Flask does not support this natively via `SESSION_COOKIE_NAME`. Is a middleware wrapper worth the complexity?

4. **Rails 8.2 `Sec-Fetch-Site` CSRF**: This browser-native approach eliminates token injection entirely for modern browsers. Should APG adopt this as a complementary CSRF check alongside tokens? It requires no JS changes and is zero-cost to add.

5. **Breached password database**: ASVS 6.2.4 (L1) requires checking against top-3000 passwords. Should APG bundle a wordlist or implement a HIBP API call (requires network access at registration time)?

6. **Session backend**: APG currently uses Flask's signed cookie sessions. These have no server-side revocation capability, making ASVS 7.4.2 (terminate all sessions on account disable) impossible without a server-side session store. This is a fundamental architecture decision.

7. **HSTS**: Included in Django/Rails/Laravel defaults. APG's security headers should include `Strict-Transport-Security: max-age=31536000; includeSubDomains` — but only if TLS is guaranteed (emitting it for non-HTTPS deployments trains browsers to expect HTTPS and breaks HTTP-only dev). Generate HSTS conditionally based on a `FORCE_HTTPS` config flag.
