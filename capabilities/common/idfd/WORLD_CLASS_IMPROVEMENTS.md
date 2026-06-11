# Identity Federation (IDFD) — World-Class Improvements

15 high-impact improvements to elevate IDFD to production-grade identity federation infrastructure.

---

## 1. Persistent Storage Backend Abstraction

**Current state**: All stores are plain in-memory `dict` instances — data is lost on restart.

**Improvement**: Introduce a `FederationRepository` abstract base with `PostgreSQLFederationRepository` and `InMemoryFederationRepository` implementations. Service wires to the repo at construction time. Enables zero-downtime rolling restarts, cross-replica session consistency, and point-in-time audit recovery.

**Priority**: Critical — required before any production deployment.

---

## 2. Async-Native I/O Throughout

**Current state**: Async method wrappers call synchronous inner logic. No actual I/O suspension occurs; `await` buys nothing.

**Improvement**: Convert all store operations to async repository calls (`await repo.get(...)`, `await repo.put(...)`). Use `asyncio.gather` in `health_report` and `federation_analytics` to parallelize independent reads. This keeps event-loop time per request under 5 ms even with external DB latency.

**Priority**: High — needed for correct async behaviour under load.

---

## 3. Structured Federation Policy Engine with Hot-Reload

**Current state**: Policy rules are baked into `capability_contract.py` and evaluated synchronously without versioning.

**Improvement**: Replace with a rule DAG stored in a versioned JSON/YAML policy file. Support hot-reload via `inotify`/`watchfiles` so rule changes take effect without service restart. Expose `policy_version` on every `evaluate()` response. Enables canary policy rollouts and instant incident response.

**Priority**: High.

---

## 4. JWKS Endpoint Caching and Key Rotation Automation

**Current state**: Certificate records are registered manually; there is no automated rotation lifecycle or JWKS cache.

**Improvement**: Add `async jwks_refresh(tenant_id, provider_id)` that fetches the IdP's JWKS URI (via `httpx` with connection pooling), validates each JWK against the registered `signing_key_id`, caches results in a TTL-bounded `BoundedCache`, and triggers `certificate_expiry_check` pre-emptively. Reduces MTTR for key rollover incidents from hours to seconds.

**Priority**: High.

---

## 5. SAML Assertion Signature Verification

**Current state**: `saml_sp_metadata` generates metadata structure but performs no cryptographic validation.

**Improvement**: Add `async saml_assertion_verify(tenant_id, provider_id, raw_assertion_b64)` using `xmlsec1`/`pysaml2` to verify the assertion's XML-DSIG signature against the registered certificate, validate `NotBefore`/`NotOnOrAfter` conditions, and check `Recipient` ACS URL against `redirect_allowlist`. Prevents assertion replay and IdP impersonation.

**Priority**: Critical for SAML security posture.

---

## 6. OIDC Token Introspection and Revocation

**Current state**: `token_exchange` issues token refs but never validates downstream token validity.

**Improvement**: Add `async oidc_token_introspect(tenant_id, provider_id, token_ref)` (RFC 7662) and `async oidc_token_revoke(tenant_id, provider_id, token_ref, hint)` (RFC 7009) with provider-side HTTP calls. Tie revocation to `trust_revoke` so provider suspension cascades to token invalidation at the IdP level.

**Priority**: High.

---

## 7. SCIM 2.0 Bulk Operations and Patch Support

**Current state**: `user_provision` and `user_deprovision` are single-subject operations with flat attribute dicts.

**Improvement**: Add `async scim_bulk_op(tenant_id, provider_id, operations, actor)` implementing RFC 7644 §3.7 bulk request semantics (method, path, data, bulkId). Support `PATCH` with `add`/`remove`/`replace` operations on individual attributes. Return RFC-compliant bulk response with per-operation status codes. This brings the SCIM surface to enterprise IAM compatibility.

**Priority**: High.

---

## 8. Risk-Based Adaptive Authentication Steps

**Current state**: `risk_score` is recorded but never triggers step-up challenges.

**Improvement**: Add `async evaluate_session_risk(tenant_id, session_id, signals)` where `signals` includes IP reputation, device fingerprint, login velocity, and geo-distance. Return a decision: `allow | step_up | block` with `challenge_type` (TOTP, FIDO2, biometric). Integrate the decision into `issue_session` so high-risk scores automatically require `reauth_completed=True`.

**Priority**: High — directly reduces account takeover risk.

---

## 9. Federation Event Streaming to External Sinks

**Current state**: Audit events are stored only in the in-process `_audit_events` dict; no external sink integration exists.

**Improvement**: Add a pluggable `FederationEventSink` protocol with `KafkaFederationSink`, `WebhookFederationSink`, and `NullFederationSink` implementations. `_audit()` becomes async and publishes to the configured sink. Enables SIEM integration, real-time alerting, and compliance export without modifying service logic.

**Priority**: High for regulated environments.

---

## 10. Multi-Tenant Provider Discovery and Metadata Caching

**Current state**: Providers must be registered manually; no automatic discovery from well-known endpoints.

**Improvement**: Add `async discover_provider(tenant_id, issuer_url, protocol)` that fetches `/.well-known/openid-configuration` or SAML metadata URL, parses the response, auto-populates `signing_key_id` and `metadata_url`, and registers the provider atomically. Cache discovery results with configurable TTL to avoid redundant network round-trips.

**Priority**: Medium-High — dramatically reduces operator onboarding friction.

---

## 11. Session Binding to Device Fingerprints

**Current state**: Sessions bind only to `subject_id` and `provider_id`; no device-level binding.

**Improvement**: Add optional `device_fingerprint` and `client_ip` fields to `FederatedSession`. `issue_session` records fingerprint at issuance; subsequent session validation calls can assert fingerprint continuity. Prevents session token theft across devices.

**Priority**: Medium.

---

## 12. Comprehensive Observability with OpenTelemetry

**Current state**: No tracing spans, metrics, or structured log fields beyond audit events.

**Improvement**: Instrument every public service method with `opentelemetry-sdk` spans carrying `tenant_id`, `provider_id`, and operation outcome as span attributes. Export `federation.session.issued`, `federation.provider.stale`, and `federation.trust.revoked` counters via OTLP. Enables SLO alerting and latency attribution without log scraping.

**Priority**: Medium.

---

## 13. Claim Transformation Pipeline with DSL

**Current state**: `transform` field accepts a string label (`copy`, etc.) but no actual transformation logic is executed.

**Improvement**: Replace the single-string transform with a typed `ClaimTransform` DSL supporting `copy`, `regex_extract`, `jmespath`, `static_value`, `join`, and `split` operations. Add `async apply_claim_pipeline(tenant_id, provider_id, raw_claims)` that runs the registered mapping chain against an incoming assertion claims dict and returns resolved local claims. Eliminates manual claim adapter code in each consuming app.

**Priority**: Medium.

---

## 14. Certificate Chain Validation and Pinning

**Current state**: `register_certificate` stores certificate metadata but performs no chain validation.

**Improvement**: Add `async certificate_chain_validate(tenant_id, certificate_id, pem_chain)` using `cryptography` library to: verify issuer chain to a configured trust anchor, check OCSP/CRL revocation status, validate key usage extensions (digitalSignature for SAML signing), and enforce minimum RSA-2048 / ECDSA P-256 key size. Reject registration if chain validation fails.

**Priority**: High for PKI-heavy enterprise deployments.

---

## 15. Tenant-Scoped Rate Limiting and Abuse Detection

**Current state**: No rate limiting on session issuance, token exchange, or provider registration.

**Improvement**: Add a `FederationRateLimiter` backed by a sliding-window counter in Redis (or in-process `BoundedCache` for single-node). Apply per-tenant-per-operation quotas configurable via capability contract (`max_sessions_per_hour`, `max_token_exchanges_per_minute`). Expose `async rate_limit_status(tenant_id)` returning current consumption against quota. Block operations that exceed quota with `429`-style `PermissionError("rate_limit_exceeded")`.

**Priority**: Medium — required for multi-tenant SaaS hardening.
