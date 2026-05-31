# Digital Forms and eSign Capability

`esgn` provides APG's common capability for governed digital forms and electronic signatures. It composes form-template authoring, schema validation, publication approval, submissions, signature envelopes, ordered signing ceremonies, cancellation/rejection, tamper sealing, encrypted evidence packages, first-class provider-neutral signing agents, UI route metadata, visual theming, and Bytewax lifecycle guardrails.

## What It Provides

- Governed form templates with owner, schema, compliance framework, DLP policy, retention policy, publication approval, and audit events.
- Validated form submissions with tenant-scoped evidence references and deterministic validation hashes.
- Signature envelopes with required subject, recipient consent, delegated signing policy checks, document hash, expiry, ordered routing, and tamper seal.
- Signing ceremonies that require identity verification, explicit signature intent, active envelope state, valid seal, non-expired envelope, correct signer order, and one signature per recipient.
- Envelope cancellation and rejection with mandatory reason and audit event.
- Evidence packages with encrypted seal digest, certificate ID, retention policy, and audit trail reference.
- First-class signing agents for form guidance, clause review, routing coordination, evidence audit, compliance review, signer-experience review, lifecycle review, and signing stewardship, with runtime, role, scope, owner, purpose, disclosure, and privileged-role approval controls.
- Bytewax lifecycle stream metadata for template, submission, envelope, signature, evidence, signing-agent, and audit batches.
- Dependency-light API helpers, UI view models, package manifest, semantic model, and release evidence.

## Runtime Shape

The generated runtime is `service.EsgnService`. It is deterministic and in-memory so generated applications can exercise the complete form and signature lifecycle without external identity, encryption, document storage, notification, workflow, audit, or AI-agent services.

Primary methods:

- `create_template(...)`
- `publish_template(...)`
- `submit_form(...)`
- `create_envelope(...)`
- `sign_envelope(...)`
- `cancel_envelope(...)`
- `reject_envelope(...)`
- `create_evidence_package(...)`
- `register_signing_agent(...)`
- `validate_lifecycle_batch(...)`
- `verify_tamper_seal(...)`
- `validate_batch_mutation(...)`
- `dashboard_summary(...)`

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rules
- UI route contracts
- theme tokens
- APG adapter map
- Bytewax streaming contract
- first-class signing-agent manifest
- Bytewax lifecycle-batch manifest

The rule engine returns `allow`, `require_review`, or `deny` decisions with matched rules and required actions. Runtime methods enforce the same guardrails used by the contract.

## UI Surfaces

The package exposes route contracts for:

- dashboard
- forms
- builder
- submissions
- envelopes
- signing
- agents
- lifecycle
- evidence
- audit
- analytics
- settings

`views.py` provides dependency-light models for these screens.

## How To Use

```python
from capabilities.common.esgn.service import EsgnService

service = EsgnService()
service.create_template(
    "tpl-nda",
    "tenant-1",
    "Mutual NDA",
    "legal-ops",
    ["counterparty", "effective_date"],
    "esign-act",
    "regulated-field-scan",
    "legal-7y",
)
service.publish_template("tpl-nda", "tenant-1", "legal-approver", True)
service.submit_form(
    "sub-1",
    "tenant-1",
    "tpl-nda",
    "sales-ops",
    {"counterparty": "Acme Ltd", "effective_date": "2026-05-30"},
    "audit:sub-1",
)
expires_at = service.default_expiry()
service.create_envelope(
    "env-1",
    "tenant-1",
    "sub-1",
    "Mutual NDA signature",
    [{"id": "rcp-1", "name": "Ada", "email": "ada@example.com", "routing_order": 1, "consent_recorded": True}],
    "sales-ops",
    "approve_nda",
    document_hash="sha256:document-v1",
    expires_at=expires_at,
)
service.sign_envelope("cer-1", "tenant-1", "env-1", "rcp-1", "approve_nda", True)
evidence = service.create_evidence_package("evd-1", "tenant-1", "env-1", True, "legal-7y", "audit:env-1")
service.register_signing_agent(
    "agent-1",
    "tenant-1",
    "Signing Steward",
    "codex",
    "signing_steward",
    "env-1",
    "legal-ops",
    True,
    purpose="Govern signing evidence and lifecycle controls.",
    human_approval_required=True,
)
service.validate_lifecycle_batch("tenant-1", "bytewax", 1, "signing_agent_batch")
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/esgn/__init__.py capabilities/common/esgn/capability_contract.py capabilities/common/esgn/models.py capabilities/common/esgn/signing_engine.py capabilities/common/esgn/service.py capabilities/common/esgn/api.py capabilities/common/esgn/views.py capabilities/common/esgn/app.py capabilities/common/esgn/test_capability_contract.py capabilities/common/esgn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/esgn/test_capability_contract.py capabilities/common/esgn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/esgn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/esgn --json
```
