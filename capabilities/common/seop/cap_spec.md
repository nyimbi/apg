# Security Operations Capability Specification

- **Capability Name**: Security Operations
- **Capability ID**: `seop`
- **Category**: common
- **Version**: 1.0.0

## Purpose

SEOP provides the dependency-light security-operations runtime for APG. It
turns alert detections, anomaly triage, incident response, playbook approvals,
response actions, posture controls, and audit evidence into deterministic
package behavior that generated applications can compose immediately.

## Current Executable Runtime

The package exposes `SeopService`, API helpers, and UI view models behind the
existing SEOP capability contract.

Current behavior:

- creates detections from trusted alert sources with anomaly confidence,
  severity, signal references, rule decisions, and required triage actions;
- opens incidents with owner, severity, linked detections, escalation evidence,
  and critical-incident guardrails;
- approves response playbooks with owner, ordered steps, and approver identity;
- executes response actions against incidents only through approved playbooks;
- records posture controls with coverage bands for operations readiness;
- closes incidents only when closure evidence is attached;
- records audit events for detections, incidents, playbooks, responses, and
  closures;
- exposes dashboard, detection console, incident queue, triage, playbook,
  response, posture, settings, and list APIs.

The compatibility `create_record` and `list_records` helpers map older generic
package calls to detection behavior while new code uses the domain-specific
methods.

## Provided Services

- `detection_pipeline`
- `incident_response`
- `threat_triage`
- `response_playbooks`
- `security_posture`
- `seop_operations`

## Required Services

- `tenant_context`
- `secu` for security policy and posture integration
- `anom` for anomaly detection inputs
- `moni` for telemetry and alert feeds
- optional `logt`, `ztna`, `dlpd`, and compliance adapters

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. Response playbooks require explicit approval before execution.

## Rules

- `tenant_context_required`
- `detection_requires_alert_source`
- `incident_requires_owner`
- `critical_incident_requires_escalation`
- `response_requires_playbook_approval`
- `high_confidence_anomaly_requires_review`

## UI

The package exposes eight APG Python UI routes through `views.py` and the
package semantic model:

- `/seop/dashboard`
- `/seop/detections`
- `/seop/incidents`
- `/seop/triage`
- `/seop/playbooks`
- `/seop/responses`
- `/seop/posture`
- `/seop/settings`

## Theme

The package uses the `seop_security_ops` APG theme contract with compact
security-operations density, severity indicators, incident priority lists,
playbook approval chips, and posture coverage panels.

## Adapter Boundaries

The dependency-light runtime deliberately does not connect to live SIEM, SOAR,
EDR, MDM, ZTNA, DLP, case-management, ticketing, compliance, threat-intel, or
telemetry systems. Those systems should be added as adapters around the current
service methods so local package behavior remains deterministic and testable.

## Focused Verification

Use these commands for battery-conscious package proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/seop/__init__.py capabilities/common/seop/models.py capabilities/common/seop/ops_runtime.py capabilities/common/seop/service.py capabilities/common/seop/api.py capabilities/common/seop/views.py capabilities/common/seop/capability_contract.py capabilities/common/seop/app.py capabilities/common/seop/test_capability_contract.py capabilities/common/seop/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/seop/test_capability_contract.py capabilities/common/seop/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/seop --json
./.venv/bin/apg capabilities publish-plan capabilities/common/seop --json
```
