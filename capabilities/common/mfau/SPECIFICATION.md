# MFAU Capability Specification

## Purpose

MFAU makes multi-factor authentication a first-class APG capability. It must let generated applications compose strong authentication, adaptive step-up, factor lifecycle management, and account recovery without binding the generated app to a specific web framework or external identity provider.

## Users

- Application builders compose MFAU into generated APG applications.
- Security administrators manage policies, allowed factors, device trust, recovery, and audit evidence.
- End users enroll methods, respond to challenges, recover access, and manage backup codes.
- Auditors inspect authentication state changes and recovery events.

## Functional Requirements

1. Register tenant-scoped user MFA profiles with a user identifier, policy identifier, status, and primary recovery channel.
2. Enroll authentication methods with type validation, channel verification, device binding, biometric consent, template encryption evidence, and factor-secret encryption evidence.
3. Bind and score trusted devices, including low-trust device review.
4. Assess risk per authentication attempt with device trust, external risk signals, and review requirements.
5. Create authentication challenges that require active methods, profile presence, unexpired single-use tokens, step-up for high risk, phishing-resistant factors for privileged actions, and lockout behavior.
6. Complete challenges only when verification evidence is present and replay checks pass.
7. Recover accounts only when a profile exists, a verified recovery channel is present, required admin approval is recorded, and audit evidence is captured.
8. Generate backup codes and enforce single-use code consumption.
9. Disable or rotate methods only when safety prerequisites are met.
10. Change MFA policies only with audit evidence.
11. Deny cross-tenant access.
12. Route batch MFA mutation through Bytewax.

## Configuration

The capability contract must define configuration sections for tenant context, profiles, methods, enrollment, challenges, risk, devices, recovery, backup codes, policies, biometrics, security, governance, observability, adapters, UI, and theme.

Required adapters:

- `generated_app_runtime`: `mfa_runtime.MfauService`
- `production_runtime`: `service.MFAService`
- `helper_runtime`: `mfa_runtime.py`
- `http_api`: `api.py`
- `event_stream`: `bytewax`
- `auth_provider`: `auth`
- `security_framework`: `secu`
- `encryption`: `encr`
- `audit_sink`: `audl`
- `notification`: `ntfy`
- `biometric`: `biop`
- `computer_vision`: `cvsn`
- `cache`: `cach`
- `metrics_sink`: `moni`

## Rule Engine

The rule engine must be deterministic and executable without network access. Rules must return:

- `allow` when no guardrail matches.
- `require_review` when the operation can continue only after review.
- `deny` when a safety, tenant, verification, or governance condition fails.

The minimum rule coverage is:

- tenant context
- profile registration
- method enrollment
- biometric consent and encryption
- device binding
- active method limits
- challenge creation and verification
- risk-based step-up
- privileged action phishing resistance
- device trust review
- token expiry and replay
- failed-attempt lockout
- account recovery
- backup codes
- method disable and rotation
- policy audit
- external risk signals
- Bytewax batch mutation
- cross-tenant denial
- state-change audit

## Runtime Contract

`MfauService` must expose deterministic methods for generated applications:

- `describe`
- `evaluate`
- `create_user_profile`
- `enroll_method`
- `bind_device`
- `assess_risk`
- `create_challenge`
- `complete_challenge`
- `recover_account`
- `generate_backup_codes`
- `use_backup_code`
- `disable_method`
- `rotate_method`
- `create_policy`
- list helpers and `dashboard_summary`
- `package`

The runtime must keep in-memory state for generated apps and examples. Production deployments can replace the runtime through adapters while preserving the same contract.

## UI Contract

The UI must expose route metadata for:

- dashboard
- profiles
- methods
- enrollment
- challenges
- risk
- devices
- recovery
- backup codes
- policies
- biometrics
- governance
- audit
- settings

Each route must include a path, component, permission, and navigation group. Theme components must cover factor stacks, profile cards, method cards, challenge panels, risk meters, device trust, enrollment, recovery, backup codes, policy editing, biometric consent, and audit timelines.

## Security and Governance Requirements

- All operations are tenant-scoped.
- Cross-tenant access is denied by default.
- Authentication state changes require audit evidence.
- Sensitive methods require encryption evidence.
- Biometric enrollment requires explicit consent.
- Recovery actions require verified channels and audit evidence.
- Admin-assisted recovery requires approval.
- Batch state changes use Bytewax.

## Acceptance Criteria

- The capability contract exposes at least 30 rules and at least 12 UI routes.
- The generated semantic model is built from the live contract.
- `app.self_test()` fails if routes, rules, adapters, or runtime metadata become stale.
- Focused tests exercise the contract, rule engine, runtime lifecycle, API helpers, UI helpers, and package entrypoint.
- Capability documentation explains usage, configuration, UI composition, and integration boundaries.
