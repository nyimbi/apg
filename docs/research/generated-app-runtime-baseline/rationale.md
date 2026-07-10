# Design Rationale — Generated App Runtime Security Baseline

*Why these decisions, not just what they are.*

---

## 1. Stdlib-Only Constraint

**Decision**: All security primitives in the generated app must use Python stdlib only (no `argon2-cffi`, `bcrypt`, `Flask-Limiter`, etc. as emitted dependencies).

**Why**: A code generator's output must be deployable without the generator's opinionated dependency set bleeding into the application's runtime. If APG emits `argon2-cffi` as a dependency, the operator must maintain it, pin it, audit it, and upgrade it — for a library they didn't choose. This is a bad DX pattern.

**Tradeoff accepted**: scrypt (`hashlib.scrypt`) is available in stdlib but is second-best to Argon2id. PBKDF2 (`hashlib.pbkdf2_hmac`) is third-best but universally available. Both are cryptographically adequate for 2025/2026. The gap between scrypt and Argon2id is measurable in attack cost (Argon2id has better side-channel resistance) but not exploitable in practice against properly-parameterized scrypt.

**What we reject**: shipping with no password hashing, or with a simple `hashlib.sha256(password)`, because that is exploitable immediately (rainbow tables, GPU cracking). This is worse than suboptimal; it is actively insecure.

**Future path**: Generate a comment in the emitted code indicating that `argon2-cffi` can be dropped in as an upgrade, with no API change, once the operator adds the dependency.

---

## 2. scrypt as the Default KDF (not PBKDF2)

**Decision**: `hashlib.scrypt(N=131072, r=8, p=1)` as default, with PBKDF2-SHA256 (600,000 iter) as automatic fallback if `hashlib.scrypt` raises `ValueError` (OpenSSL not available).

**Why scrypt over PBKDF2**:
- PBKDF2 is CPU-hard only. Modern GPUs can run millions of PBKDF2 iterations per second even at 600k iterations per hash.
- scrypt is memory-hard (N=131072 requires ~128 MiB RAM). GPU attacks are bottlenecked by memory bandwidth, not compute. Cost to crack is orders of magnitude higher.
- OWASP explicitly recommends scrypt when Argon2id is unavailable.

**Why N=2^17 (131072) not N=2^15 (32768)**:
- OWASP minimum is N=2^17. This is non-negotiable if we claim OWASP compliance.
- At N=2^17 on a typical server, one hash takes ~200-300ms. This is acceptable for login (users tolerate up to ~1s). It is a P0 denial-of-service risk without a request size limit on the password field — a 10MB password at this cost exhausts a worker. **This is why password max-length validation and `MAX_CONTENT_LENGTH` are mandatory co-requirements with KDF selection.**

**Fallback logic**:
```python
try:
    hashlib.scrypt(b"test", salt=b"test"*2, n=2, r=8, p=1)
    _KDF = "scrypt"
except (ValueError, AttributeError):
    _KDF = "pbkdf2"
```
This probe runs at module import time. Zero runtime cost.

---

## 3. Session Fixation as P0 (not P1)

**Decision**: Session regeneration on login is Priority 0, mandatory, not optional.

**Why P0**: ASVS 7.2.4 is L1 — the lowest assurance level. L1 means "basic, should be detectable by automated scanning." Session fixation is a well-known, easy-to-exploit attack with documented CVSS scores in the 6-8 range. Every peer framework (Django, Rails, Laravel) implements this automatically in their auth helpers. For APG to not implement it would put APG-generated apps below what a developer gets for free with any mainstream framework.

**Flask complication**: Flask's signed cookie sessions don't have a server-side record. The "session ID" is the signed cookie value. Regeneration = clearing the cookie and reissuing with new content (causing Flask's serializer to produce a new signature). This is semantically equivalent to server-side session ID rotation for the cookie-session model.

**What we explicitly do NOT do**: Change the secret key on login. That would invalidate all other users' sessions simultaneously — a global logout, which is a DoS.

---

## 4. In-Memory Rate Limiter (not "requires Redis")

**Decision**: Generate a basic per-IP in-memory rate limiter rather than requiring Flask-Limiter + Redis.

**Why not require Redis**: A generated app with a Redis dependency at startup won't run in a simple `python app.py` invocation. For many APG use cases (internal tools, prototypes, small deployments), Redis is not available. Requiring it breaks the "runs anywhere" promise.

**Why not skip it**: ASVS 6.3.1 (L1) requires rate limiting controls. Shipping with zero rate limiting is a compliance failure.

**Why in-memory over nothing**: Even with multi-worker caveats, in-memory rate limiting provides:
- Protection against single-script attacks (the most common brute force pattern)
- Compliance documentation ("we have rate limiting")
- A clear upgrade path (replace the dict with a Redis call)

**Acknowledged limitation**: The generated code must include a prominent comment:
```python
# WARNING: This in-memory rate limiter is per-process. With multiple
# worker processes (gunicorn --workers N), the effective limit is
# N * RATE_LIMIT. For production deployments with multiple workers,
# replace this with a Redis-backed implementation (e.g., Flask-Limiter).
```

**Why per-IP (not per-username) as primary**:
- Per-username enables lockout DoS if attacker knows valid usernames
- Per-IP is harder to abuse for DoS (attacker would need to attack from their own IP range)
- Combined is best; per-IP is the safe default for a generated app

---

## 5. Health Endpoint Separation (liveness vs readiness)

**Decision**: Emit two separate endpoints: `/healthz` (liveness, no dependency checks) and `/readyz` (readiness, checks DB connectivity).

**Why separate**: This is the most important design decision for health checks. A single `/health` endpoint that checks the database will cause Kubernetes crash-loop restarts when the database is unavailable. The correct behavior when the DB is down: stop sending traffic to the pod (readiness=503) but don't restart the pod (liveness=200). A pod that isn't restarted can reconnect when the DB recovers; a pod in crash-loop cannot.

**Why these names specifically**: `/healthz` is the Kubernetes de-facto convention (used by kube-apiserver, etcd, etc.). The `-z` suffix is Kubernetes-internal convention. `/readyz` follows the same pattern. Alternative `/health/live` and `/health/ready` are equally valid but less widely recognized.

**Auth exclusion**: Health endpoints must be excluded from authentication middleware. If the auth middleware depends on the database, and the database is down, the health endpoint returns 401/403 instead of 503, masking the real failure. Bootstrap-order the health blueprint before auth middleware.

---

## 6. Structured Logging at L2 (not optional)

**Decision**: Generate structured JSON logging for security events even though ASVS V16 is all L2.

**Why, given it's L2**: Every serious operator of a generated app will need to answer "what happened at 14:32 yesterday?" within months of deployment. Apps without structured security logging are unauditable. The cost to generate it (a few hundred lines) is much lower than the cost to retrofit it later (requires touching every auth code path).

**Why JSON specifically**: Machine-parseable. Compatible with every log aggregator (CloudWatch, Loki, Datadog, Splunk, ELK). Plain text logs are a dead end for any non-trivial deployment.

**Why not use Python's `logging` module's default format**: The default `%(asctime)s %(levelname)s %(message)s` is not structured. We need JSON output. The simplest implementation: `logging.info(json.dumps(event_dict))`. This requires no extra libraries and produces parseable output.

**Sensitive data in logs**: Session tokens and passwords must never appear in logs. This is not just OWASP guidance — it's a breach amplifier. If logs are exported to a SIEM (which they should be), a leaked log file becomes a session hijacking kit. Rule: log `sha256(session_id)[:16]` not `session_id`. Log "auth.login.failure" not "wrong password for alice@example.com".

---

## 7. Generic Error Messages — Implementation Approach

**Decision**: All authentication failures return `"Invalid username or password."` regardless of failure reason. Both cases (user not found, wrong password) return the same HTTP status.

**Why**: ASVS 6.3.8 (L1). This is basic. But the subtlety is HTTP status: returning 404 for "user not found" vs 401 for "wrong password" leaks the same information as a distinct message. The generated code must normalize status codes as well as messages.

**Performance implication**: The "user not found" path must be as slow as the "wrong password" path. Skipping the password hash computation for non-existent users creates a timing oracle. The dummy hash `verify_password(candidate, DUMMY_HASH)` call ensures uniform response time. This dummy hash must be computed at module load time (not per-request) to avoid revealing timing information about the hash computation itself.

---

## 8. `MAX_CONTENT_LENGTH` as Security Feature

**Decision**: Set `app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024` (16 MiB) in all generated apps.

**Why it matters specifically for password hashing**: With scrypt at N=2^17, hashing a 10 MB "password" takes seconds and pins a worker thread. An attacker can exhaust all workers with a handful of concurrent requests. Without this limit, enabling a memory-hard KDF creates a DoS vector on the `/login` endpoint itself.

**Why 16 MiB default**: Generous for form submissions (a 16 MiB form is pathological), tight enough to prevent the DoS. Operators can raise it if their app handles large file uploads.

**Additional password-specific limit**: Set `MAX_PASSWORD_LENGTH = 1024` in the auth code. OWASP recommends this specifically to prevent KDF-exhaustion DoS on the authentication endpoint. Reject passwords > 1024 bytes before hashing.

---

## 9. HSTS Conditional Generation

**Decision**: Emit HSTS header (`Strict-Transport-Security`) only when a `FORCE_HTTPS=true` environment variable is set, not unconditionally.

**Why not unconditional**: If HSTS is emitted over HTTP, Chrome/Firefox persist it. The next HTTP request will be browser-upgraded to HTTPS, which may not exist in a dev or HTTP-only deployment. This breaks the app and is hard to undo (requires clearing browser HSTS state manually or waiting for max-age to expire).

**Why not skip it**: HSTS is in Django's, Rails's, and Laravel's production checklists. Generating apps without it puts them below the competitive bar in production.

**Implementation pattern**: Check environment variable in the response post-processor, not in the security headers blueprint. This makes it clearly conditional and easy to find.

---

## 10. What We Explicitly Defer

**Argon2id**: Requires `argon2-cffi`. The right algorithm; wrong dependency boundary for a generated app. Document as the recommended upgrade path.

**Server-side session store**: ASVS 7.4.2 (invalidate all sessions on account disable) is architecturally incompatible with Flask's signed cookie sessions. This is a known limitation. Document it. If APG needs full ASVS L2 session management, it must emit a database-backed session store. That's a bigger architectural decision than this research can resolve.

**Breached password list bundling**: ASVS 6.2.4 (L1) requires top-3000 password check. Bundling a compressed wordlist is ~15KB and low-complexity, but it's a content decision (which wordlist? which version?) that should be a separate implementation decision, not buried in this research.

**MFA/TOTP**: ASVS 6.3.3 (L2) requires MFA. Out of scope for baseline session auth. Noted for roadmap.

**`__Host-` cookie prefix**: Technically superior to standard cookie naming but requires Flask internals work. Low priority relative to P0/P1 gaps.
