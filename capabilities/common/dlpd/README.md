# Data Loss Prevention (DLPD)

DLPD is APG's generated-application capability for tenant-scoped data loss
prevention. It gives composed applications a deterministic, dependency-light
surface for data classification, policy enforcement, egress inspection,
quarantine, incident response, legal hold, review, audit, and UI composition.

The generated-app runtime stores hashes and metadata, not raw sensitive content.
It is intended to make APG applications executable quickly while leaving live
network controls, storage engines, classifier model providers, and compliance
systems behind explicit adapter boundaries.

## What The Capability Provides

- Policy lifecycle for owned, tenant-scoped DLP controls.
- Built-in and custom classifier lifecycle with review guardrails.
- Deterministic pattern classification for PII, PHI, PCI, secrets, financial
  records, and source code.
- Egress inspection for email, API exports, file sharing, chat, clipboard, and
  object storage.
- High-severity blocking/quarantine rules, large-export review, and restricted
  destination review.
- Encrypted quarantine metadata with legal-hold flags.
- Incident creation, resolution, notification evidence, and digest-backed audit.
- Contract-derived UI routes, view payloads, visual theme tokens, and Bytewax
  event-stream adapter evidence.

## Generated-App Usage

```python
from capabilities.common.dlpd.service import DlpdService

service = DlpdService()
tenant_id = "tenant-dlp"

classifier = service.register_classifier(
    classifier_id="cls-secrets",
    tenant_id=tenant_id,
    name="Secrets",
    classifier_type="built_in",
    sensitivity_label="restricted",
    pattern_keys=["secrets"],
)

policy = service.register_policy(
    policy_id="pol-email",
    tenant_id=tenant_id,
    name="Email egress",
    owner="security-ops",
    channels=["email"],
    classifiers=[classifier["id"]],
    default_action="quarantine",
)

inspection = service.inspect_egress(
    inspection_id="insp-1",
    tenant_id=tenant_id,
    policy_id=policy["id"],
    channel="email",
    subject_id="user-1",
    destination="external@example.com",
    content="api_key='SECRET123456789'",
)
```

## Composition Contract

Use `get_capability_contract()` when a compiler, generator, or larger APG
application needs to inspect DLPD.

```python
from capabilities.common.dlpd.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-dlp")
routes = contract["ui"]["routes"]
rules = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Important adapter evidence:

- `generated_app_runtime`: `service.DlpdService`
- `event_stream`: `bytewax`
- `security_framework`: `secu`
- `encryption`: `encr`
- `nlp_core`: `nlpc`
- `anomaly_detection`: `anom`
- `audit_sink`: `audl`
- `message_bus`: `mqeb`
- `compliance`: `comp`

## Screens

The contract exposes route metadata for dashboard, policies, classifiers,
channels, inspections, incidents, quarantine, reviews, legal hold, analytics,
audit, and settings. The view helpers in `views.py` return dependency-light
payloads for generated Python applications.

## Guardrails

DLPD includes deterministic rules for tenant context, policy ownership, policy
channels, classifiers, active policies, covered channels, destinations,
classifier labels, custom classifier review, classifier confidence, sensitive
classification labels, source-code review, secret/high-severity blocking,
large-export review, external/restricted destinations, quarantine encryption,
quarantine content hashes, legal hold, incident ownership/resolution,
independent review, raw-content retention denial, Bytewax batch mutation,
tenant isolation, and required audit evidence.

## Verification

Focused package checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/dlpd/__init__.py capabilities/common/dlpd/capability_contract.py capabilities/common/dlpd/dlp_engine.py capabilities/common/dlpd/models.py capabilities/common/dlpd/service.py capabilities/common/dlpd/api.py capabilities/common/dlpd/views.py capabilities/common/dlpd/app.py capabilities/common/dlpd/test_capability_contract.py capabilities/common/dlpd/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/dlpd/test_capability_contract.py capabilities/common/dlpd/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dlpd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dlpd --json
```
