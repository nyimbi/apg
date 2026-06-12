# MFAU - Multi-Factor Authentication

MFAU provides adaptive multi-factor authentication for APG applications. It is a composable security capability for enrolling factors, assessing risk, issuing challenges, binding devices, governing account recovery, managing backup codes, composing first-class MFA security agents, validating Bytewax lifecycle batches, and exposing UI surfaces that generated applications can assemble into complete authentication flows.

The capability is designed to be dependency-light at the generated-application boundary. `mfa_runtime.py` is the deterministic runtime used by compiler output, examples, package tests, and capability composition. The existing production modules remain available as deeper adapters for deployments that need Flask, Flask-AppBuilder, biometric services, notification services, external AI-agent runtimes, or external security infrastructure.

## What MFAU Provides

- Tenant-scoped MFA profiles with policy assignment and lockout state.
- Method enrollment for TOTP, WebAuthn (FIDO2), push, email OTP, SMS OTP, backup codes, hardware keys, and biometrics.
- Risk-aware challenges with step-up, phishing-resistant factor requirements, low-trust device review, replay protection, and failed-attempt lockout.
- Adaptive risk evaluation returning challenge levels: `none | low | medium | high | block`.
- Device trust registration and revocation.
- Account recovery with verified recovery channels, admin approval, and audit evidence.
- Backup/recovery code generation (10 single-use codes per set) and consumption.
- Admin-issued time-limited MFA bypass grants.
- Bulk enrollment for tenant-wide MFA rollouts.
- Tenant-level analytics over configurable time windows.
- Policy management with audit requirements.
- First-class MFA security-agent registration for Codex, Claude Code, opencode, and Pi style assistants behind provider-neutral AICR adapter contracts.
- Bytewax lifecycle batch validation for profile, method, device, risk, challenge, recovery, backup-code, policy, biometric, and agent changes.
- Deterministic rule evaluation for generated apps and tests.
- UI route metadata, component names, permissions, and theme tokens.
- Bytewax event-stream contract for batch MFA mutations and generated-app pipelines.

## Package Structure

| File | Purpose |
|---|---|
| `SPECIFICATION.md` | Expected functional surface |
| `PLAN.md` | Implementation plan and review checklist |
| `capability_contract.py` | Configuration, rule engine, UI routes, theme, adapters |
| `mfa_runtime.py` | Generated-app runtime (deterministic, dependency-light) |
| `service.py` | Full production service (`MFAService`) |
| `api.py` | Dependency-light API helper functions around the runtime |
| `views.py` | Dependency-light UI model helpers |
| `app.py` | Builds semantic model and component manifest from live contract |
| `tests/` | Unit, integration, composition, and APG tests |

## API Reference

### `MFAService` — public methods

| Method | Description |
|---|---|
| `authenticate_user(user_id, tenant_id, authentication_methods, context)` | Main authentication orchestrator; handles lockout, risk, step-up |
| `enroll_mfa_method(user_id, tenant_id, method_type, enrollment_data, context)` | Enroll any method type via the unified entry point |
| `enrol_totp(user_id, tenant_id, context)` | Enroll TOTP; returns secret + QR-code URI |
| `enrol_sms(user_id, tenant_id, phone_number, context)` | Enroll SMS OTP channel |
| `enrol_email(user_id, tenant_id, email, context)` | Enroll email OTP channel |
| `enrol_hardware_key(user_id, tenant_id, key_serial, context)` | Enroll FIDO2/TOTP hardware token |
| `enrol_push(user_id, tenant_id, device_token, platform, context)` | Register push-notification MFA channel |
| `verify_mfa(user_id, tenant_id, method_id, otp_code, context)` | Verify OTP against enrolled method; returns `trust_score` |
| `step_up_auth(user_id, tenant_id, operation, context)` | Initiate step-up challenge for sensitive operation |
| `verify_step_up_authentication(user_id, tenant_id, step_up_token, additional_methods, context)` | Complete a pending step-up and receive elevated token |
| `adaptive_mfa_risk(user_id, tenant_id, context)` | Return risk score and challenge level (`none/low/medium/high/block`) |
| `mfa_bypass_admin(admin_id, tenant_id, target_user_id, reason, duration_minutes, context)` | Grant time-limited MFA bypass (admin only) |
| `mfa_recovery_code_gen(user_id, tenant_id, context)` | Generate 10 single-use recovery codes; invalidates prior set |
| `mfa_recovery_validate(user_id, tenant_id, code, context)` | Consume a recovery code; single-use, replay-safe |
| `bulk_enrol(tenant_id, user_ids, method_type, actor, context)` | Batch-enroll a list of users; returns `batch_id`, enrolled/failed counts |
| `mfa_status(user_id, tenant_id)` | Concise MFA enablement status for a user |
| `mfa_analytics(tenant_id, days)` | Tenant-level analytics over last N days |
| `get_user_mfa_status(user_id, tenant_id)` | Full status including methods, trust score, lockout, recent events |
| `trusted_device_register(user_id, tenant_id, device_info, context)` | Register a trusted device; bypasses MFA for low-risk sessions |
| `trusted_device_revoke(user_id, tenant_id, device_id, context)` | Revoke trust for a specific device |
| `generate_backup_codes(user_id, tenant_id, context)` | Generate backup codes (delegates to `token_service`) |
| `remove_mfa_method(user_id, tenant_id, method_id, context)` | Remove an enrolled method; blocks if it is the last one |
| `initiate_account_recovery(user_id, tenant_id, recovery_type, context)` | Start recovery flow (mfa_reset / account_unlock) |
| `start_biometric_enrollment(user_id, tenant_id, biometric_types, context)` | Begin guided biometric enrollment session |
| `get_service_metrics()` | Authentication counts, success rate, risk and biometric metrics |

## Basic Usage

```python
from capabilities.common.mfau.mfa_runtime import MfauService

service = MfauService()
tenant_id = "tenant-a"

profile = service.create_user_profile(
    profile_id="profile-alice",
    tenant_id=tenant_id,
    user_id="alice",
    policy_id="standard-mfa",
    primary_channel="alice@example.com",
)

device = service.bind_device(
    device_id="device-alice-laptop",
    tenant_id=tenant_id,
    user_id="alice",
    trust_score=0.82,
)

method = service.enroll_method(
    method_id="method-alice-webauthn",
    tenant_id=tenant_id,
    user_id="alice",
    method_type="webauthn",
    device_id=device["id"],
    phishing_resistant=True,
)

risk = service.assess_risk(
    assessment_id="risk-alice-login",
    tenant_id=tenant_id,
    user_id="alice",
    risk_score=0.41,
    device_trust_score=device["trust_score"],
)

challenge = service.create_challenge(
    challenge_id="challenge-alice-login",
    tenant_id=tenant_id,
    user_id="alice",
    method_id=method["id"],
    assessment_id=risk["id"],
)

service.complete_challenge(
    challenge_id=challenge["id"],
    tenant_id=tenant_id,
    verification_evidence=True,
)
```

## New Methods

### Adaptive Risk Evaluation

```python
result = await service.adaptive_mfa_risk(
    user_id="alice",
    tenant_id="tenant-a",
    context={"ip": "203.0.113.5", "device_id": "laptop-001"},
)
# {"user_id": "alice", "risk_score": 0.55, "challenge_level": "medium", "factors_required": 1}
```

### Recovery Code Lifecycle

```python
# Generate (invalidates any prior set)
gen = await service.mfa_recovery_code_gen("alice", "tenant-a", ctx)
# {"success": True, "codes": ["A1B2C", ...], "count": 10}

# Consume (single-use)
val = await service.mfa_recovery_validate("alice", "tenant-a", "A1B2C", ctx)
# {"success": True, "remaining_codes": 9}
```

### Admin MFA Bypass

```python
bypass = await service.mfa_bypass_admin(
    admin_id="sre-bot",
    tenant_id="tenant-a",
    target_user_id="alice",
    reason="emergency account recovery — ticket INC-4421",
    duration_minutes=30,
    context=ctx,
)
# {"success": True, "bypass_key": "tenant-a:alice", "expires_in_minutes": 30}
```

### Bulk Enrollment

```python
batch = await service.bulk_enrol(
    tenant_id="tenant-a",
    user_ids=["alice", "bob", "carol"],
    method_type=MFAMethodType.TOTP,
    actor="onboarding-automation",
    context=ctx,
)
# {"batch_id": "...", "enrolled": 3, "failed": 0, "failures": []}
```

### Trusted Device Management

```python
reg = await service.trusted_device_register("alice", "tenant-a", device_info, ctx)
# {"success": True, "device_id": "uuid...", "trusted": True}

await service.trusted_device_revoke("alice", "tenant-a", reg["device_id"], ctx)
```

### Tenant Analytics

```python
stats = await service.mfa_analytics("tenant-a", days=7)
# {tenant_id, window_days, total_events, auth_successes, auth_failures,
#  mfa_bypass_grants, recovery_code_sets, service_metrics}
```

## Rule Engine

Rules are deterministic and data-driven. Call `evaluate_capability_rules(context)` with a context dictionary. The engine returns `allow`, `require_review`, or `deny`, plus matched rules and required actions.

```python
from capabilities.common.mfau.capability_contract import evaluate_capability_rules

result = evaluate_capability_rules({
    "operation": "validate_mfa_lifecycle_batch",
    "tenant_context_present": True,
    "event_stream": "legacy_queue",
})

assert result["decision"] == "deny"
assert result["matched_rules"] == ["bytewax_mfa_stream_required"]
```

## UI Composition

Generated applications should consume `views.py` helpers rather than importing Flask or Flask-AppBuilder directly. The helpers return route-ready models for dashboards, profile registries, enrollment flows, challenge consoles, risk views, device trust, recovery, backup codes, policies, biometrics, MFA security-agent rosters, lifecycle-batch monitors, audit, and settings.

The UI contract includes stable permissions and a theme named `mfau_adaptive_auth_console`. APG builders can override tokens per tenant while retaining the same component contract.

## World-Class Enhancements (v2.0)

The following 15 improvements are tracked in `WORLD_CLASS_IMPROVEMENTS.md`. They address the primary security, scalability, and observability gaps in the current implementation.

| # | Improvement | Impact |
|---|---|---|
| 1 | **Passkey / WebAuthn Ceremony (FIDO2)** | Full FIDO2 registration + assertion via `py_webauthn`; phishing-resistance becomes structural |
| 2 | **Persistent Storage Adapter Contract** | `MFAStorageAdapter` ABC with PostgreSQL (asyncpg) and Redis backends; eliminates class-variable shared state |
| 3 | **TOTP Secret Encryption At Rest** | All secret material passes through `encrypt_value`/`decrypt_value`; closes plaintext SMS/email storage gap |
| 4 | **Replay-Protected Challenge Nonces** | `_mark_otp_used` nonce store (90-second TTL) prevents same OTP window being accepted twice |
| 5 | **Structured Audit Events with Integrity Chain** | HMAC-SHA256 chained events; satisfies SOC 2 / ISO 27001 event-integrity requirements |
| 6 | **Risk Signal Fan-Out (IP Reputation + Geo-Velocity)** | `IPReputationSignal`, `GeoVelocitySignal`, `DeviceFingerprintSignal` aggregated via weighted Bayesian model |
| 7 | **Continuous Authentication (Behavioral Biometrics)** | `score_continuous_session` accepts keystroke/pointer vectors; emits `session_trust_degraded` for step-up without session termination |
| 8 | **Policy-Driven Method Requirements** | `MFAPolicy` model per tenant/role; factor count, allowed types, phishing-resistance flag, step-up triggers, max token TTL |
| 9 | **WebAuthn Authenticator Attestation Validation** | FIDO MDS3 metadata verification; AAGUID blocklist for known-vulnerable authenticators |
| 10 | **Machine-to-Machine (M2M) MFA via Client Assertions** | `issue_m2m_assertion` validates RS256/ES256 JWTs from service identities; closes CI/CD MFA bypass gap |
| 11 | **Rate Limiting with Token Bucket per User** | Per `(tenant_id, user_id, method_type)` token bucket; `get_rate_limit_status` exposes feedback to UI |
| 12 | **Structured Recovery Channel Verification** | `RecoveryChannel` model with proof-of-possession; `verify_recovery_channel` with same replay protection as OTP |
| 13 | **Session Binding to Device Fingerprint** | `device_binding_hash` on token model (JA3 / browser fingerprint); tokens rejected if device fingerprint changes |
| 14 | **Observability — OpenTelemetry Traces and Metrics** | OTel spans on all public methods; Prometheus counters/histograms for latency (p50/p95/p99), method breakdown, lockout events |
| 15 | **Secrets Rotation Without Downtime** | `rotate_totp_secret` (pending_rotation flag + atomic promotion) and `rotate_encryption_key` (async batch re-encrypt with rollback) |

## Composition Notes

MFAU depends on `auth`, `secu`, `encr`, `aicr`, `conf`, and `audl`. Optional adapters include `ntfy`, `cvsn`, `biop`, `cach`, and `moni`. Batch mutation and event-driven integration should use Bytewax through the `event_stream` and lifecycle stream contracts.

Use MFAU when an APG application needs first-class authentication assurance rather than local per-screen password checks. Compose MFAU with authorization, audit, notification, computer vision, biometric, and risk capabilities through the contract rather than by importing private service internals.

---

Copyright © 2025 Datacraft — www.datacraft.co.ke
