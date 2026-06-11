# MFAU — World-Class Improvements

15 targeted improvements ranked by security impact and implementation leverage.

---

## 1. Passkey / WebAuthn Ceremony (FIDO2)

Replace the stub `webauthn` method type with a full FIDO2 registration and assertion ceremony using `py_webauthn`. This eliminates shared secrets entirely — the server never stores a credential it can transmit to an attacker. Phishing-resistance becomes structural rather than policy.

**Gap**: Current enrollment stores a raw serial/secret string for hardware tokens and treats WebAuthn as an enum value with no cryptographic ceremony.

---

## 2. Persistent Storage Adapter Contract

The in-memory class-level dicts (`_profiles`, `_methods`, `_auth_events`, etc.) are shared across all instances and leak between test runs. Introduce a `MFAStorageAdapter` abstract base with async `get`/`set`/`delete`/`query` and ship concrete adapters for PostgreSQL (via asyncpg) and Redis. The constructor accepts an adapter; tests inject an `InMemoryAdapter`. This removes the class-variable anti-pattern and enables horizontal scaling.

---

## 3. TOTP Secret Encryption At Rest

`enrol_totp` receives `encrypted_secret` from `token_service`, but `enrol_sms` and `enrol_email` store plaintext phone numbers and email addresses directly in `encrypted_secret`. All secret material should pass through the `encryption_key`-backed `token_service` before persistence. Add `encrypt_value` / `decrypt_value` helpers and audit every `_store_mfa_method` call site.

---

## 4. Replay-Protected Challenge Nonces

TOTP verification calls `token_service.verify_totp_code` but there is no nonce store to prevent the same valid OTP window from being accepted twice (replay within the 30-second window). Add an async `_mark_otp_used(secret_hash, otp, window_ts)` guard backed by the storage adapter with a TTL of 90 seconds (3 windows).

---

## 5. Structured Audit Events with Integrity Chain

`_log_auth_event` stores plain dicts. Security-sensitive events should be immutable, signed (HMAC-SHA256 over the previous event hash + payload), and queryable by event type, time range, and actor. This makes the audit log forensically useful and satisfies SOC 2 / ISO 27001 event-integrity requirements.

---

## 6. Risk Signal Fan-Out (IP Reputation + Geo-Velocity)

`RiskAnalyzer.assess_authentication_risk` is called with a `context` dict but the implementation detail is opaque. Add first-class async signal sources: `IPReputationSignal` (AbuseIPDB / local threat-intel feed), `GeoVelocitySignal` (impossible travel detector using haversine between last known location and current), and `DeviceFingerprintSignal`. Aggregate via a weighted Bayesian model. Expose the signal breakdown in the risk response so step-up decisions are explainable.

---

## 7. Continuous Authentication (Behavioral Biometrics)

Add `score_continuous_session(user_id, tenant_id, behavioral_sample)` that accepts keystroke-dynamics or pointer-movement vectors and updates a rolling session trust score. When the score drops below `require_step_up_threshold`, emit a `session_trust_degraded` event that the UI can react to without terminating the session. This shifts MFA from a point-in-time gate to a continuous assurance posture.

---

## 8. Policy-Driven Method Requirements

The `_determine_required_methods` logic is hard-coded by risk score thresholds. Replace it with a `MFAPolicy` model (stored per tenant, optionally per role) that specifies: minimum factor count, allowed method types, phishing-resistance requirement, step-up triggers (e.g., `ip_change`, `new_device`, `privilege_escalation`), and max token TTL. Policy evaluation should be a pure function so it can be unit-tested without service dependencies.

---

## 9. WebAuthn Authenticator Attestation Validation

When a FIDO2 device is registered, verify the attestation statement against the FIDO MDS3 metadata service to confirm the authenticator model, firmware version, and certification level. Reject authenticators with known vulnerabilities (e.g., AAGUID blocklist). Log the AAGUID and attestation result in the audit chain.

---

## 10. Machine-to-Machine (M2M) MFA via Client Assertions

Service accounts and CI/CD pipelines cannot complete interactive MFA challenges. Add `issue_m2m_assertion(service_id, tenant_id, signed_jwt, context)` that validates a short-lived signed JWT (RS256/ES256) from a pre-registered service identity, then issues a scoped bearer token. This closes the gap where automated pipelines bypass MFA entirely.

---

## 11. Rate Limiting with Token Bucket per User

The lockout mechanism (N failures → lock for M minutes) is blunt. Replace with a token-bucket rate limiter per `(tenant_id, user_id, method_type)` with configurable `capacity` and `refill_rate`. This allows legitimate users who fail once to retry quickly while still throttling brute force. Expose `get_rate_limit_status(user_id, tenant_id)` for UI feedback.

---

## 12. Structured Recovery Channel Verification

`initiate_account_recovery` delegates to `RecoveryService` whose internals are opaque. Define a recovery channel model: `RecoveryChannel(type: email|phone|backup_code|admin_approval, verified_at, last_used_at)`. Recovery flows should require proof-of-possession on at least one out-of-band channel before resetting MFA methods. Add `verify_recovery_channel(user_id, tenant_id, channel_id, proof)` with the same replay protection as OTP verification.

---

## 13. Session Binding to Device Fingerprint

Authentication tokens should be bound to the TLS client hello fingerprint (JA3) or a browser-side device fingerprint hash. Add `device_binding_hash: str | None` to the token model. Verification middleware should reject tokens presented from a different device fingerprint, preventing token theft attacks where the attacker operates from a different network stack.

---

## 14. Observability — OpenTelemetry Traces and Metrics

`_auth_metrics` is a plain in-memory dict with no cardinality or time-series structure. Instrument all public methods with OpenTelemetry spans (`opentelemetry-sdk`) and emit Prometheus counters/histograms for: authentication latency (p50/p95/p99), method-type breakdown, risk-score distribution, lockout events, and recovery invocations. This makes the service production-observable without requiring log parsing.

---

## 15. Secrets Rotation Without Downtime

TOTP secrets and encryption keys are written once and never rotated. Add `rotate_totp_secret(user_id, tenant_id, method_id, context)` that generates a new secret, stores it alongside the old one under a `pending_rotation` flag, accepts verification of the new secret, then atomically promotes it and tombstones the old. Similarly, add `rotate_encryption_key(old_key, new_key)` that re-encrypts all method secrets in a single async batch with rollback on partial failure. Both operations emit structured audit events.
