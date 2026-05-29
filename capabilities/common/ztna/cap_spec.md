# Zero Trust Network Access Capability Specification

- **Capability Name**: Zero Trust Network Access
- **Capability ID**: `ztna`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`ztna` provides an executable zero-trust network access runtime for APG
applications. It owns tenant identity verification state, device posture,
resource policy state, access decisions, high-risk access review, continuous
session verification, session revocation, UI route metadata, theme metadata,
and publish-plan evidence.

The package is dependency-light and deterministic. Live identity providers,
MFA providers, endpoint posture collectors, policy engines, service meshes,
network gateways, packet capture, SIEM feeds, and session recording systems are
adapter boundaries around the local runtime, not prerequisites for package
proof.

## Provided Services

- `zero_trust_policies`
- `device_posture`
- `resource_access_broker`
- `continuous_verification`
- `risk_based_access`
- `ztna_operations`

## Required Services

- `tenant_context`
- `auth` for identity context
- `secu` for security policy integration
- `mfau` for privileged MFA evidence
- `moni` for runtime monitoring and risk signals

Optional adapters may integrate `audl`, `idfd`, `anom`, and `mqeb` when a
capacity requires audit fanout, identity fraud detection, anomaly scoring, or
event-bus delivery.

## Runtime Behavior

The current package runtime is implemented in `zero_trust_runtime.py`,
`service.py`, `api.py`, and `views.py`.

Executable lifecycles:

- register and verify tenant identities, including privileged identities and
  MFA completion state;
- register devices with posture, attestation, management, compliance, trust
  score, and quarantine/trusted status;
- register resources with access level, sensitivity, network segment, and
  policy attachment state;
- attach resource policies and clear `policy_required` state;
- request access by evaluating tenant context, identity, device posture,
  resource policy, privileged MFA, and risk review rules;
- approve high-risk access requests before session start;
- start approved sessions;
- continuously reevaluate sessions and either keep them active, require
  reauthentication, or revoke them;
- close sessions;
- expose dashboard, policy console, device posture, resource map, access
  requests, session monitor, risk console, and settings view models;
- append audit events for identities, devices, resources, access requests,
  approvals, sessions, reevaluations, and closures.

Compatibility helpers `create_record()` and `list_records()` remain available
for generic package tooling. `create_record()` creates a protected resource and
`list_records()` returns protected resources rather than storing generic
records.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`.

Required configuration sections:

- `tenant_id`
- `identities`
- `devices`
- `resources`
- `governance`
- `ui`
- `theme`

Important default controls:

- verified identity required;
- MFA required for privileged access;
- continuous identity checks enabled;
- device posture required;
- minimum device trust score set to `0.7`;
- resource policy required;
- least privilege default enabled;
- session recording for privileged resources enabled;
- microsegmentation enabled;
- tenant context required;
- access decision audit enabled;
- deny by default enabled;
- high-risk review threshold set to `0.8`.

## Rules

The deterministic rule engine exposes these rule IDs:

- `tenant_context_required`
- `identity_must_be_verified`
- `device_posture_required`
- `resource_policy_required`
- `privileged_access_requires_mfa`
- `high_risk_access_requires_review`

Service guardrails enforce the same decisions:

- missing tenant context raises `tenant_context_required`;
- unverified identity raises `identity_verification_required`;
- missing device posture raises `device_posture_required`;
- resource without attached policy raises `resource_policy_required`;
- privileged access without MFA raises `privileged_mfa_required`;
- access risk above `0.8` creates a `review_required` request with required
  action `review_access_request` until approval is recorded.

Continuous session verification uses the same rule engine. A deny decision
revokes the session. A review decision marks the session `review_required` and
sets `reauth_required`.

## UI

The package exposes eight APG Python UI routes:

- `/ztna/dashboard` via `ZTNADashboard`
- `/ztna/policies` via `ZeroTrustPolicies`
- `/ztna/devices` via `DevicePosture`
- `/ztna/resources` via `ResourceMap`
- `/ztna/access` via `AccessRequests`
- `/ztna/sessions` via `SessionMonitor`
- `/ztna/risk` via `AccessRiskConsole`
- `/ztna/settings` via `ZTNASettings`

`views.py` returns dependency-light view models for these routes. The view
models include route names, tenant context, relevant records, available
actions, summary counts, risk signals, and theme/configuration metadata.

## Theme

The package uses the `ztna_zero_trust_ops` APG theme contract. Current
component theme metadata covers access decisions, device posture, resource
maps, and session monitors.

## Adapter Boundaries

Keep these integrations behind APG composition adapters:

- identity providers and directory synchronization;
- MFA push, token, and biometric providers;
- endpoint posture and attestation collectors;
- network access gateways and service meshes;
- policy decision points and policy administration points;
- session recording systems;
- SIEM, audit, anomaly, and incident-response feeds;
- packet inspection and microsegmentation enforcement.

Local package proof must remain deterministic without those providers.

## Focused Verification

Use these battery-conscious commands after ZTNA package changes:

```bash
./.venv/bin/python -m py_compile capabilities/common/ztna/__init__.py capabilities/common/ztna/models.py capabilities/common/ztna/zero_trust_runtime.py capabilities/common/ztna/service.py capabilities/common/ztna/api.py capabilities/common/ztna/views.py capabilities/common/ztna/capability_contract.py capabilities/common/ztna/app.py capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ztna/test_capability_contract.py capabilities/common/ztna/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ztna --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ztna --json
```

When global readiness changes, also run:

```bash
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```
