# Sources

All URLs accessed: 2026-07-10

---

## OWASP Standards & Cheat Sheets

1. **OWASP ASVS 5.0 — V6 Authentication Requirements**
   - URL: https://github.com/OWASP/ASVS/blob/master/5.0/en/0x15-V6-Authentication.md
   - Accessed: 2026-07-10
   - Summary: Full text of V6 authentication chapter. Extracted all 6.x.x requirement numbers and levels for password policy, rate limiting, error message requirements, account lifecycle.

2. **OWASP ASVS 5.0 — V7 Session Management Requirements**
   - URL: https://asvs.dev/v5.0.0/V7-Session-Management/
   - Accessed: 2026-07-10
   - Summary: Full V7 session management requirements with levels. Key finding: 7.2.4 (session token regeneration on login) is L1. V7.3 timeout requirements are L2.

3. **OWASP ASVS 5.0 — V16 Security Logging and Error Handling**
   - URL: https://github.com/OWASP/ASVS/blob/v5.0.0/5.0/en/0x25-V16-Security-Logging-and-Error-Handling.md
   - Accessed: 2026-07-10
   - Summary: All V16 requirements are L2. Covers structured log fields (who/what/when/where), what events to log, log protection, and error handling (fail-secure, generic errors, circuit breaker pattern).

4. **OWASP ASVS 5.0 — Table of Contents (all 17 chapters)**
   - URL: https://sentrixhub.com/owasp-asvs-5-0-table-of-contents/
   - Accessed: 2026-07-10
   - Summary: Complete chapter mapping. V1-V17 with titles and foci. Confirmed V7=Session Management, V16=Logging.

5. **OWASP ASVS Foundation Page**
   - URL: https://owasp.org/www-project-application-security-verification-standard/
   - Accessed: 2026-07-10
   - Summary: Confirms ASVS 5.0 released May 2025, ~350 requirements across 17 chapters, 3 levels.

6. **OWASP Password Storage Cheat Sheet**
   - URL: https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
   - Accessed: 2026-07-10
   - Summary: Canonical algorithm recommendations with exact parameters. Argon2id (m=19456, t=2, p=1) preferred; scrypt (N=2^17, r=8, p=1) second; PBKDF2-HMAC-SHA256 (600,000 iter) for FIPS; bcrypt (cost≥10) legacy only.

7. **OWASP CheatSheetSeries — Password Storage (GitHub source)**
   - URL: https://github.com/OWASP/CheatSheetSeries/blob/master/cheatsheets/Password_Storage_Cheat_Sheet.md
   - Accessed: 2026-07-10
   - Summary: Same content as above; cross-referenced for consistency.

8. **OWASP Session Management Cheat Sheet**
   - URL: https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
   - Accessed: 2026-07-10
   - Summary: Session ID entropy (≥64 bits), cookie attributes (Secure, HttpOnly, SameSite), `__Host-` prefix recommendation, session fixation defense (regenerate on privilege change), timeout values (2-5 min high-value, 15-30 min low-risk; absolute 4-8 hours).

9. **OWASP Authentication Cheat Sheet**
   - URL: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
   - Accessed: 2026-07-10
   - Summary: Generic error message requirements, account lockout principles (per-account counter, not per-IP), timing-safe comparison guidance. No specific numeric thresholds for lockout specified.

10. **OWASP Logging Cheat Sheet**
    - URL: https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
    - Accessed: 2026-07-10
    - Summary: Complete log field specification: when (timestamp, event_id), where (app name, address, endpoint), who (source IP, user identity, user type), what (event type, severity, result, reason, HTTP status). Full list of events requiring security logging.

11. **OWASP ASVS Index of Cheat Sheets**
    - URL: https://cheatsheetseries.owasp.org/IndexASVS.html
    - Accessed: 2026-07-10
    - Summary: Cross-reference between ASVS requirements and relevant cheat sheets.

12. **OWASP Session Fixation Protection**
    - URL: https://owasp.org/www-community/controls/Session_Fixation_Protection
    - Accessed: 2026-07-10
    - Summary: Confirms session ID regeneration as mandatory defense. References ASVS 5.0.0-3.2.2 (former numbering scheme; now 7.2.4 in v5).

---

## NIST Standards

13. **NIST SP 800-63B Rev 4 — Digital Identity Guidelines: Authentication**
    - URL: https://pages.nist.gov/800-63-4/sp800-63b.html
    - Accessed: 2026-07-10
    - Summary: Minimum 8 characters, support 64+ characters, no mandatory complexity rules, no periodic rotation unless compromised, screen against breached-password lists. Published July 2024, supersedes Rev 3.

14. **NIST SP 800-63B-4 — CSRC page**
    - URL: https://csrc.nist.gov/pubs/sp/800/63/b/4/final
    - Accessed: 2026-07-10
    - Summary: Final publication reference. Confirms publication date and authoritative source.

---

## Python Standard Library

15. **Python hashlib documentation**
    - URL: https://docs.python.org/3/library/hashlib.html
    - Accessed: 2026-07-10
    - Summary: `hashlib.scrypt(password, *, salt, n, r, p, maxmem=0, dklen=64)` — Python 3.6+, requires OpenSSL. `hashlib.pbkdf2_hmac(hash_name, password, salt, iterations, dklen=None)` — available when Python compiled with OpenSSL (Python 3.12+). Documented recommendation: 500k iterations for application-specific use; references NIST-SP-800-132.

16. **Python hmac documentation**
    - URL: https://docs.python.org/3/library/hmac.html
    - Accessed: 2026-07-10
    - Summary: `hmac.compare_digest(a, b)` — constant-time comparison to prevent timing attacks. Available since Python 3.3. Use instead of `==` for all cryptographic comparisons.

---

## Framework Security References

17. **Rails Security Guide — Ruby on Rails official docs**
    - URL: https://guides.rubyonrails.org/security.html
    - Accessed: 2026-07-10
    - Summary: Comprehensive guide to Rails built-in security features. CSRF via `protect_from_forgery`, session cookie encryption, parameter filtering, SQL injection prevention via parameterized queries.

18. **Rails Has Your Back: Security You Don't Have to Think About (Mario Chavez)**
    - URL: https://mariochavez.io/desarrollo/2026/04/13/rails-security-you-dont-have-to-think-about/
    - Accessed: 2026-07-10
    - Summary: Enumerated Rails 8 default security features: SQL injection prevention, XSS protection, CSRF token scoped per method/action (Rails 8.0+), Sec-Fetch-Site CSRF (Rails 8.2), encrypted sessions, HttpOnly+SameSite=Lax+Secure cookies, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, HSTS, ReDoS 1s timeout, built-in auth generator with bcrypt.

19. **Rails CSRF Protection Best Practices**
    - URL: https://blog.saeloun.com/2026/04/28/rails-security-best-practices-a-comprehensive-guide/
    - Accessed: 2026-07-10
    - Summary: Current Rails security best practices including Sec-Fetch-Site header check in Rails 8.2.

20. **Laravel Cloud Security Defaults**
    - URL: https://laravel.com/blog/laravel-cloud-security-defaults-behind-every-deploy
    - Accessed: 2026-07-10
    - Summary: Framework-level defaults (CSRF, Blade XSS escaping, Eloquent parameterized queries, Hash facade for bcrypt/Argon2, mass assignment protection) plus Cloud-level defaults (WAF, DDoS, rate limiting at 100 req/min, X-Frame-Options: DENY, X-Content-Type-Options: nosniff, HSTS, auto TLS, deploy-time dependency vulnerability scanning).

21. **Laravel Security Best Practices 2026**
    - URL: https://benjamincrozat.com/laravel-security-best-practices
    - Accessed: 2026-07-10
    - Summary: Practical enumeration of what Laravel ships and what requires explicit configuration.

22. **Django Security Cheat Sheet (OWASP)**
    - URL: https://cheatsheetseries.owasp.org/cheatsheets/Django_Security_Cheat_Sheet.html
    - Accessed: 2026-07-10
    - Summary: Django SECURE_* settings, AUTH_PASSWORD_VALIDATORS, CSRF middleware, session cookie settings.

23. **Django Password Management Docs**
    - URL: https://docs.djangoproject.com/en/3.2/topics/auth/passwords/
    - Accessed: 2026-07-10
    - Summary: Django uses PBKDF2-SHA256 by default with high iteration count; Argon2 available as optional backend. Built-in validators: MinimumLength, CommonPassword, UserAttributeSimilarity, Numeric.

24. **Phoenix Security Documentation**
    - URL: https://phoenix.hexdocs.pm/security.html (redirect from hexdocs.pm/phoenix/security.html)
    - Accessed: 2026-07-10
    - Summary: Phoenix defaults: `:protect_from_forgery` plug in browser pipeline, `put_secure_browser_headers` plug, `:fetch_session`, content auto-escaping. SameSite=Lax default for session cookies.

25. **Elixir and Phoenix Security: Complete Guide (Curiosum)**
    - URL: https://curiosum.com/blog/security-in-elixir
    - Accessed: 2026-07-10
    - Summary: Phoenix security feature enumeration including session management, CSRF, XSS escaping.

---

## Production Ops & Observability

26. **Kubernetes Liveness, Readiness, and Startup Probes (official docs)**
    - URL: https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/
    - Accessed: 2026-07-10
    - Summary: Canonical definitions. Liveness = should pod restart? Readiness = should pod receive traffic? Startup = is pod done initializing?

27. **Configure Liveness, Readiness and Startup Probes (Kubernetes tasks)**
    - URL: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
    - Accessed: 2026-07-10
    - Summary: Endpoint naming conventions: `/healthz` (liveness), `/readyz` (readiness). Configuration examples.

28. **How to Build Health Checks and Readiness Probes in Python for Kubernetes**
    - URL: https://oneuptime.com/blog/post/2025-01-06-python-health-checks-kubernetes/view
    - Accessed: 2026-07-10
    - Summary: Python-specific implementation patterns. Liveness never checks external dependencies; readiness does.

29. **How to Implement Health Checks That Distinguish Liveness and Readiness**
    - URL: https://oneuptime.com/blog/post/2026-02-09-health-checks-liveness-vs-readiness/view
    - Accessed: 2026-07-10
    - Summary: Design rule: failing liveness due to DB down causes crash-loop restarts. Readiness is the correct place for dependency checks.

30. **OpenTelemetry Logging Specification**
    - URL: https://opentelemetry.io/docs/specs/otel/logs/
    - Accessed: 2026-07-10
    - Summary: OTel log data model includes TraceId, SpanId, TraceFlags as first-class fields. JSON structure for correlation.

31. **Trace Context in Non-OTLP Log Formats (OpenTelemetry)**
    - URL: https://opentelemetry.io/docs/specs/otel/compatibility/logging_trace_context/
    - Accessed: 2026-07-10
    - Summary: `trace_id` and `span_id` should appear as top-level fields in JSON log entries for correlation.

32. **W3C TraceContext & OpenTelemetry Context Propagation (Uptrace)**
    - URL: https://uptrace.dev/opentelemetry/context-propagation
    - Accessed: 2026-07-10
    - Summary: `traceparent` header structure, propagation mechanics. Relation between X-Request-ID (simple correlation) and trace IDs (distributed tracing).

33. **Correlation ID vs Trace ID (Last9)**
    - URL: https://last9.io/blog/correlation-id-vs-trace-id/
    - Accessed: 2026-07-10
    - Summary: Practical distinction: correlation ID = single identifier across one request end-to-end; trace ID = root identifier for distributed trace tree with span hierarchy.

34. **flask-log-request-id (GitHub: Workable)**
    - URL: https://github.com/Workable/flask-log-request-id
    - Accessed: 2026-07-10
    - Summary: Flask extension for X-Request-ID propagation. Supports X-Request-ID, X-Correlation-ID, AWS X-Amzn-Trace-Id. Includes Celery worker propagation.

35. **flask-request-id-header (PyPI: antarctica)**
    - URL: https://github.com/antarctica/flask-request-id-header
    - Accessed: 2026-07-10
    - Summary: Simpler Flask middleware that accepts or generates UUID-based request IDs on each request.

---

## Rate Limiting & Anti-Automation

36. **Flask Security Best Practices 2025 (Corgea)**
    - URL: https://corgea.com/learn/flask-security-best-practices-2025
    - Accessed: 2026-07-10
    - Summary: Flask-specific rate limiting guidance. Per-IP + per-user combined approach. Redis backend required for multi-process.

37. **Flask-Limiter (PyPI)**
    - URL: https://pypi.org/project/Flask-Limiter/
    - Accessed: 2026-07-10
    - Summary: De-facto standard Flask rate limiting library. Supports fixed window, sliding window, token bucket. Redis/memcached backends for production.

38. **Preventing Brute Force Attacks with Flask-Limiter (Qadr Labs)**
    - URL: https://qadrlabs.com/post/preventing-brute-force-attacks-rate-limiting-your-flask-api-with-flask-limiter
    - Accessed: 2026-07-10
    - Summary: Practical patterns for login endpoint protection. Per-IP strict limit; per-user more lenient.

39. **Django-axes (Jazzband)**
    - URL: https://github.com/jazzband/django-axes
    - Accessed: 2026-07-10
    - Summary: Django's canonical failed-login tracking plugin. Supports per-IP, per-username, per-user-agent tracking; configurable lockout thresholds; cache or DB backend.

---

## Additional References

40. **Password Hashing: Argon2 vs Bcrypt vs Scrypt vs PBKDF2 (guptadeepak.com)**
    - URL: https://guptadeepak.com/the-complete-guide-to-password-hashing-argon2-vs-bcrypt-vs-scrypt-vs-pbkdf2-2026/
    - Accessed: 2026-07-10
    - Summary: Practitioner comparison with 2026 recommended parameters. Confirms OWASP guidance; adds context on attack costs (GPU hashrates per algorithm).

41. **Password Hashing in 2026: bcrypt vs Argon2 vs scrypt vs PBKDF2 (Toolsana)**
    - URL: https://toolsana.com/blog/password-hashing-2026-bcrypt-argon2-scrypt-pbkdf2-guide/
    - Accessed: 2026-07-10
    - Summary: Updated 2026 practitioner guide. Notes bcrypt's 72-byte truncation limit as active risk; recommends Argon2id or scrypt as replacements.

42. **Timing Attacks Against String Comparison in Python (Sqreen)**
    - URL: https://sqreen.github.io/DevelopersSecurityBestPractices/timing-attack/python
    - Accessed: 2026-07-10
    - Summary: Explains the timing oracle from `==` comparison. Demonstrates `hmac.compare_digest()` as the fix. Applicable to any secret comparison (tokens, hashes, API keys).

43. **OWASP ASVS v5: Raising the Bar for Application Security (Cyber Chief)**
    - URL: https://www.cyberchief.ai/2025/10/owasp-asvs-v5-raising-bar-for.html
    - Accessed: 2026-07-10
    - Summary: Overview of what changed in ASVS 5.0 vs 4.0. New V17 (WebRTC), reorganized chapters, ~350 total requirements.

44. **Building authentication in Rails web applications (WorkOS)**
    - URL: https://workos.com/blog/rails-authentication-guide-2026
    - Accessed: 2026-07-10
    - Summary: Rails 8 built-in authentication generator details, bcrypt integration, session management defaults.

45. **Flask app ships with no upload size limit (GitHub issue)**
    - URL: https://github.com/EAPD-DRB/MUIOGO/issues/270
    - Accessed: 2026-07-10
    - Summary: Real-world security issue report confirming Flask's default of no MAX_CONTENT_LENGTH is a security vulnerability.
