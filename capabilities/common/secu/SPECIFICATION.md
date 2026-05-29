# APG SECU Capability Specification

## Purpose

`secu` is the APG Security Framework capability. It provides the executable
security control plane for generated APG applications: zero-trust policy,
device posture, threat indicators, access/risk decisions, compliance evidence,
policy exceptions, incident response, containment, and audit evidence.

The package must remain usable without live identity providers, SIEM, EDR/MDM,
SOAR, compliance platforms, AI detectors, FastAPI, Flask-AppBuilder, or policy
providers. Those systems are adapters behind the package contract.

## Capability Outcomes

- Register tenant-scoped security policies and required controls.
- Record device posture and quarantine compromised devices.
- Register threat indicators and use them in risk posture.
- Assess access with deterministic deny, quarantine, challenge, and allow
  decisions.
- Record compliance controls and require audit evidence for gaps.
- Govern policy exceptions with independent approval and expiry evidence.
- Open, contain, resolve, and audit security incidents.
- Expose API helpers, view models, UI routes, theme components, rules,
  semantic evidence, and focused tests for APG composition.

## First-Class Domain Concepts

### Security Policy

A tenant-scoped policy specifying security level, owner, required controls, and
scope.

### Device Posture

Trust and risk state for a device. Compromised or high-risk devices are
quarantined.

### Threat Indicator

Tenant-scoped indicator of compromise or threat intelligence item.

### Risk Assessment

Access decision evidence for a subject, device, network, compliance state, and
risk score.

### Compliance Control

Framework control status, owner, compliance state, and audit evidence.

### Policy Exception

Time-bound reviewed exception for a policy or control.

Required evidence:

- `id`
- `tenant_id`
- `policy_id`
- `requested_by`
- `reason`
- `expires_at`
- `status`
- `decision`
- `reviewer`
- `notes`

### Security Incident

Incident response lifecycle record.

Required evidence:

- `id`
- `tenant_id`
- `title`
- `severity`
- `opened_by`
- `status`
- `containment_action`
- `containment_evidence`
- `resolution`
- `resolved_by`

## Lifecycle Requirements

### Policy And Posture

- Tenant context is required.
- Policies require owner and security level.
- Devices require user and supported trust state.
- Compromised or high-risk devices are quarantined.
- Threat indicators require name, value, severity, source, and TTL.

### Access Assessment

- Known malicious network origins deny access.
- Compromised devices quarantine the device.
- Critical risk denies access.
- High risk requires challenge unless challenge evidence is present.
- Compliance violations require attached audit evidence.

### Policy Exception

- Exception requests require requester, reason, expiry, and target policy.
- Exception approval requires independent reviewer and notes.
- Expired exceptions cannot be approved.
- Rejected exceptions cannot be used as bypass evidence.

### Incident Response

- Critical incidents require a containment plan.
- Containment requires actor and evidence.
- Resolution requires containment evidence, resolver, resolution, and notes.
- Incidents and containment emit security audit events.

## Rules

The deterministic rule engine must enforce at least:

- `known_malicious_network_denied`
- `compromised_device_quarantined`
- `critical_risk_denied`
- `high_risk_requires_challenge`
- `compliance_violation_alert`
- `policy_exception_requires_independent_reviewer`
- `expired_policy_exception_denied`
- `critical_incident_requires_containment`
- `incident_resolution_requires_containment`

## UI Surfaces

SECU must expose routes and theme components for:

- Security dashboard
- Risk console
- Threat console
- Policy workbench
- Policy exception queue
- Incident response queue
- Device quarantine console
- Compliance console
- Rule workbench
- Audit timeline
- Settings

## Adapter Boundaries

The executable package must not require live SIEM, EDR, IAM, GRC, DLP, SOAR,
policy-engine, AI-detection, or web-server integrations to satisfy its package
contract.

Production adapters must preserve the same guardrails:

- Do not accept high-risk or malicious contexts without SECU decision evidence.
- Do not unquarantine compromised devices without evidence.
- Do not approve policy exceptions without independent review.
- Do not resolve critical incidents without containment evidence.
- Do not satisfy compliance controls without audit evidence.

## Focused Proof

Battery-conscious proof for this slice:

```bash
./.venv/bin/python -m py_compile capabilities/common/secu/__init__.py capabilities/common/secu/security_runtime.py capabilities/common/secu/service.py capabilities/common/secu/api.py capabilities/common/secu/views.py capabilities/common/secu/capability_contract.py capabilities/common/secu/app.py capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/secu --json
./.venv/bin/apg capabilities publish-plan capabilities/common/secu --json
```
