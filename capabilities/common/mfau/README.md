# MFAU - Multi-Factor Authentication

MFAU provides adaptive multi-factor authentication for APG applications. It is a composable security capability for enrolling factors, assessing risk, issuing challenges, binding devices, governing account recovery, managing backup codes, and exposing UI surfaces that generated applications can assemble into complete authentication flows.

The capability is designed to be dependency-light at the generated-application boundary. `mfa_runtime.py` is the deterministic runtime used by compiler output, examples, package tests, and capability composition. The existing production modules remain available as deeper adapters for deployments that need Flask, Flask-AppBuilder, biometric services, notification services, or external security infrastructure.

## What MFAU Provides

- Tenant-scoped MFA profiles with policy assignment and lockout state.
- Method enrollment for TOTP, WebAuthn, push, email OTP, SMS OTP, backup codes, hardware keys, and biometrics.
- Risk-aware challenges with step-up, phishing-resistant factor requirements, low-trust device review, replay protection, and failed-attempt lockout.
- Device trust records and review guardrails.
- Account recovery with verified recovery channels, admin approval, and audit evidence.
- Backup code generation and single-use verification.
- Policy management with audit requirements.
- Deterministic rule evaluation for generated apps and tests.
- UI route metadata, component names, permissions, and theme tokens.
- Bytewax event-stream contract for batch MFA mutations and generated-app pipelines.

## Package Structure

- `SPECIFICATION.md` defines the expected functional surface.
- `PLAN.md` records the implementation plan and review checklist.
- `capability_contract.py` declares configuration, rule engine, UI routes, theme, and adapters.
- `mfa_runtime.py` implements the generated-app runtime.
- `api.py` exposes dependency-light API helper functions around the runtime.
- `views.py` exposes dependency-light UI model helpers.
- `app.py` builds the semantic model and component manifest from the live contract.
- `test_capability_contract.py` and `tests/test_package_contract.py` provide focused verification.

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

## Rule Engine

Rules are deterministic and data-driven. Call `evaluate_capability_rules(context)` with a context dictionary. The engine returns `allow`, `require_review`, or `deny`, plus matched rules and required actions.

```python
from capabilities.common.mfau.capability_contract import evaluate_capability_rules

result = evaluate_capability_rules({
    "operation": "batch_mfa_mutation",
    "tenant_context_present": True,
    "event_stream": "kafka",
})

assert result["decision"] == "deny"
assert result["matched_rules"] == ["batch_mfa_mutation_requires_bytewax"]
```

## UI Composition

Generated applications should consume `views.py` helpers rather than importing Flask or Flask-AppBuilder directly. The helpers return route-ready models for dashboards, profile registries, enrollment flows, challenge consoles, risk views, device trust, recovery, backup codes, policies, biometrics, audit, and settings.

The UI contract includes stable permissions and a theme named `mfau_adaptive_auth_console`. APG builders can override tokens per tenant while retaining the same component contract.

## Composition Notes

MFAU depends on `auth`, `secu`, and `encr`. Optional adapters include `audl`, `ntfy`, `cvsn`, `biop`, `cach`, and `moni`. Batch mutation and event-driven integration should use Bytewax through the `event_stream` adapter.

Use MFAU when an APG application needs first-class authentication assurance rather than local per-screen password checks. Compose MFAU with authorization, audit, notification, computer vision, biometric, and risk capabilities through the contract rather than by importing private service internals.
