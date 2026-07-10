# Raw Reasoning — Generated App Runtime Security Baseline

*Stream-of-consciousness notes from the research session. Not polished prose.*

---

## Initial Framing

The question is: what does a *generated* Flask app need to ship so that it meets or exceeds what the best peer frameworks (Django, Rails, Laravel, Phoenix) ship by default, and what ASVS 5.0 requires at L1/L2?

APG already has: security headers (CSP, X-Frame-Options etc.), CSRF tokens on session forms, hardened session cookies (HttpOnly, SameSite, Secure, ephemeral secret fallback). That's basically V3 (Web Frontend Security) at L1. Good start.

What is obviously missing based on first principles before any research:
- Password hashing (this is table-stakes; any framework that doesn't hash passwords is disqualified from the "world-class" conversation immediately)
- Session fixation defense (rotate session on login)
- Rate limiting on the login endpoint
- Health endpoints for Kubernetes/cloud deployment
- Structured logging
- Request ID propagation

The research should pin down exact parameters and ASVS requirement numbers, not just confirm the obvious.

---

## ASVS 5.0 Structure Discovery

First thing to confirm: does ASVS 5.0 really exist and when was it released?

Yes — released May 2025. 17 chapters (added V17 WebRTC). ~350 requirements. The key chapters for us:
- V6: Authentication
- V7: Session Management  
- V16: Security Logging
- V3: Web Frontend (already done)
- V11: Cryptography (algorithm selection)

Searched for V6 authentication requirements directly — got the GitHub markdown. Good extraction.

Key surprise: ASVS 6.3.8 (L1) — "valid users cannot be deduced from failed authentication challenges." This is the generic error message requirement. A lot of apps get this wrong. Django gets this right by default; APG almost certainly doesn't (it will either 404 on missing username or give different messages).

Key finding for session management: 7.2.4 is L1, not L2. I expected session fixation defense to be L2 ("advanced"), but it's explicitly L1. That makes it P0 for APG.

V16 logging: all requirements are L2. This is interesting — OWASP considers structured security logging "standard" not "basic." But all the peer frameworks aspire to it via plugins (Lograge for Rails, etc.) so it's still competitive table stakes.

---

## Password Storage Deep Dive

The OWASP Password Storage Cheat Sheet is clear:
1. Argon2id — preferred (PHC winner, memory-hard, side-channel resistant)
2. scrypt — second choice (memory-hard, available in Python stdlib)
3. bcrypt — legacy only (72-byte truncation is a real attack surface)
4. PBKDF2-HMAC-SHA256 — FIPS compliance only (CPU-hard only, no memory hardness)

For stdlib-only constraint: Argon2id requires `argon2-cffi` or similar. Not stdlib. So our real options are:
- `hashlib.scrypt()` — Python 3.6+, but requires OpenSSL. Most production environments have this. Best stdlib choice.
- `hashlib.pbkdf2_hmac()` — pure Python fallback (actually requires OpenSSL since 3.12 too... checking)

Wait — the Python docs say pbkdf2_hmac is "only available when Python is compiled with OpenSSL" since Python 3.12. Before 3.12 it was available without OpenSSL. This is a complication.

Practical reality: both scrypt and pbkdf2_hmac require OpenSSL in modern Python. Any deployment environment with Python 3.6+ and OpenSSL (which is essentially all of them) can use scrypt. Only exotic embedded or stripped builds would lack it.

Decision: default to scrypt with N=2^17 (OWASP minimum), PBKDF2 as documented fallback. Store format should encode algorithm + params so future parameter upgrades are non-breaking (PHC string format pattern).

PBKDF2 iteration count: OWASP says 600,000+ for SHA-256. Python docs show 500k as example. Django (the gold standard) uses 600k as of recent versions. Use 600,000.

Timing attack defense: `hmac.compare_digest()` — this needs to be emphasized heavily. It's in stdlib (Python 3.3+). `secrets.compare_digest()` is identical. The failure mode (using `==` on hash output) is a classic mistake. The generated app must use this everywhere, not just where the developer remembers to.

Also critical: dummy hash on missing username. If the user doesn't exist, skip password hashing → response comes back in microseconds vs ~200ms for a real hash. That's a username enumeration oracle. Fix: run the hash computation anyway, then discard.

---

## Login Rate Limiting Analysis

The problem with pure in-memory rate limiting for a generated app:
- gunicorn default: multiple worker processes
- Each process has its own counter dict
- With 4 workers and limit=5, effective limit is 20

For a *generated* app with "stdlib only" constraint, we can't ship Redis as a dependency. Options:
1. Ship with in-memory limiter + conspicuous `# PRODUCTION: replace with Redis-backed limiter` comment
2. Use a file-based shared counter (bad — file locking is complex)
3. Use a process-shared counter via multiprocessing.Manager (ugly)

Decision: in-memory with a very clear generated comment. The alternative (no rate limiting) is worse. At minimum it protects against naive single-process attacks and complies with "must have some rate limiting" for ASVS 6.3.1.

Should it be per-IP or per-username?
- Per-IP: works for targeted attacks; fails against distributed botnets
- Per-username: works against distributed attacks; creates lockout DoS if attacker knows username
- Combined (per-IP AND per-username): best, but double the complexity

OWASP Authentication Cheat Sheet says counter should be tied to the *account itself* to survive distributed attacks, but account lockout creates DoS risk → need reset via email. This is complex.

Practical recommendation: per-IP as primary control (simple, no DoS risk), per-username as secondary with softer threshold (just adds delay, no hard lockout). For APG's generated app, per-IP with 5/15-min window is the right baseline.

---

## Session Fixation — Why It Matters for Flask

Flask's default session is a signed cookie. The session "ID" is the cookie value itself (signed with the app secret key). There's no server-side session record to look up.

Session fixation in this model: attacker can't easily pre-set a session ID because Flask signs cookies — a cookie without a valid signature is rejected. BUT: if Flask's signing key is weak/known (the ephemeral fallback generates a random key each restart, so this isn't an issue in APG's case), or if a pre-auth session cookie is issued without signing requirements, fixation is possible.

The more practical defense: on login, `session.clear()` and re-populate. This generates a new signed cookie value because the cookie contents changed → Flask's serializer produces a different signed output. Not identical to server-side session ID rotation, but equivalent protection for cookie-based sessions.

If APG adds server-side sessions (in-database or Redis-backed), then true session ID rotation is needed: delete old DB record, create new one, set cookie to new ID.

---

## Health Endpoints — Design Considerations

Standard naming:
- `/healthz` — liveness (kubernetes legacy; Google's convention)
- `/readyz` — readiness (kubernetes legacy)
- `/health/live` and `/health/ready` — more readable alternative

The critical design rule I found: NEVER fail liveness because an external dependency is down. If DB is down and liveness fails → k8s restarts pod → DB is still down → pod restarts again → crash loop. Liveness should only fail if the *process itself* is broken (stuck, OOM, deadlocked).

For APG's simple Flask apps: readiness should check DB connectivity. Liveness should just return 200.

JSON response format with version info is useful — makes it easy to confirm what's deployed.

These endpoints must be excluded from authentication middleware, or they break monitoring in unauthenticated contexts.

---

## Structured Logging

OWASP Logging Cheat Sheet is specific about fields. The "who/what/when/where" model:
- When: ISO-8601 UTC timestamp, event_id
- Where: app name, endpoint URL, HTTP method, code location
- Who: source IP, user_id (after auth), session_id (hashed — NEVER raw), user type
- What: event_type, severity, result, reason, HTTP status

Key constraint: `session_id` must be hashed in logs. If raw session IDs appear in logs and logs are compromised, all active sessions are compromised. Use `hashlib.sha256(session_id.encode()).hexdigest()` for logging.

Same for username in failed login logs — arguably PII and enables enumeration. Hash it: `hashlib.sha256(username.lower().encode()).hexdigest()[:16]` (truncated for readability).

Event type enum to generate:
- auth.login.success
- auth.login.failure  
- auth.login.rate_limited
- auth.logout
- auth.session.created
- auth.session.destroyed
- auth.session.fixation_defense (emitted when we clear-and-regenerate on login)
- authz.denied
- security.validation_failure
- security.csrf_failure
- error.unhandled
- error.dependency_failure (DB down, etc.)

---

## Competitive Analysis — Key Surprises

**Rails 8 is farther ahead than I expected**:
1. Built-in rate limiting (Rails 7.2+) — not third-party
2. `X-Request-Id` set by default — no plugin needed
3. Rails 8.0: authentication generator ships with bcrypt out of the box
4. Rails 8.2: CSRF via `Sec-Fetch-Site` header — eliminates CSRF tokens for modern browsers entirely. This is a paradigm shift.
5. ReDoS protection (1-second regex timeout) — novel

**Django is thorough but verbose**: requires explicit configuration of most security settings. Good defaults exist but aren't always enabled out of the box.

**Laravel**: bcrypt/Argon2 via Hash facade — can't mess this up. Rate throttle middleware built-in. But no structured logging by default.

**Phoenix**: secure by default on XSS and CSRF, but session security is fairly manual. No built-in rate limiting.

**The gap APG needs to close**: password hashing and session fixation are the two features literally every peer framework ships by default. Everything else (rate limiting, health endpoints, structured logging) is "competitive parity" rather than "table stakes."

---

## Open Questions I Couldn't Fully Resolve

1. **hashlib.scrypt OpenSSL availability**: Python 3.6+ + OpenSSL is nearly universal but not guaranteed. Need to decide if APG should probe at startup and fall back gracefully to PBKDF2.

2. **Cookie prefix `__Host-`**: OWASP recommends it strongly. But Flask's `SESSION_COOKIE_NAME` config doesn't let you set `__Host-session` easily — the `__Host-` prefix has browser-enforced requirements (no Domain, Path=/, Secure). Flask doesn't natively support this. Requires a custom response wrapper. Worth doing but complex.

3. **Server-side session revocation**: Flask signed cookies can't be revoked without the secret key changing (which revokes ALL sessions). ASVS 7.4.2 says terminating a user account must disallow all active sessions — impossible with pure cookie sessions. This is a fundamental limitation. APG should either accept this limitation (document it) or emit a server-side session store.

4. **Breached password list**: ASVS 6.2.4 (L1!) requires checking against top-3000 passwords. Bundling a wordlist in the generated app adds ~15KB but makes the L1 requirement true. The alternative (HIBP API call) adds network dependency. Lean toward bundled compressed list.

5. **Sec-Fetch-Site CSRF**: Rails 8.2 is doing this. Python/Flask community hasn't adopted it. Should APG add it as an additional CSRF check? It's browser-native, requires no JS changes, and is impossible to spoof by JS. Could coexist with existing token-based CSRF. Worth investigating as a future enhancement.

6. **HSTS conditional on HTTPS**: If APG generates HSTS unconditionally and someone runs the app over HTTP (dev mode), browsers will remember the HSTS policy and break HTTP. Need a `FORCE_HTTPS` flag that controls HSTS emission. Currently APG might be emitting HSTS unconditionally — needs verification.
