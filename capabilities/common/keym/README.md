# APG Key Management Capability

Key Management (`keym`) is APG's cryptographic key lifecycle control plane. It
provides generated applications with dependency-light key governance while
keeping live HSM, KMS, vault, blockchain audit, AI lifecycle, and security
intelligence providers behind adapters.

The package includes a large async production-facing key-management service.
Generated APG applications should use `KeymService` for portable package
behavior: key creation, operation decisions, export approval, rotation
evidence, compromise response, UI view models, and package proof.

## What KEYM Provides

- Tenant-scoped managed keys with owner, algorithm, key class, policy reference,
  HSM attestation, lifecycle state, compromise state, and rotation age.
- Deterministic key operation decisions for create, use, export, rotate,
  disable, and destroy operations.
- Dual-control export approval with independent reviewer and reviewer notes.
- Rotation exception review for overdue keys.
- Rotation scheduling and completion with evidence.
- Compromise response that blocks cryptographic use until rotation evidence is
  recorded.
- First-class key-agent composition for policy, lifecycle, custody, export,
  rotation-exception, compromise-response, and HSM-attestation review.
- Bytewax lifecycle stream enforcement for grouped key mutations.
- API helpers and UI view models for generated applications.
- Contract, theme, semantic model, and release evidence for APG composition
  tooling.

## Service Usage

```python
from capabilities.common.keym.service import KeymService

service = KeymService()
key = service.create_managed_key(
	tenant_id="tenant-a",
	key_id="finance-root",
	name="Finance Root",
	owner="security-admin",
	algorithm="AES-256",
	key_class="root",
	policy_ref="policy://finance-root",
	hsm_attested=True,
)

decision = service.evaluate_key_operation(
	tenant_id="tenant-a",
	operation_id="use-finance-root",
	key_id=key["id"],
	operation="use_key",
)
```

Export requires KEYM-owned dual-control approval:

```python
approval = service.request_export_approval(
	tenant_id="tenant-a",
	approval_id="export-1",
	key_id=key["id"],
	requested_by="integration-owner",
	reason="Partner encrypted migration.",
)

approved = service.decide_export_approval(
	tenant_id="tenant-a",
	approval_id=approval["id"],
	reviewer="key-custodian",
	decision="approved",
	notes="Approved wrapped export only.",
)
```

Rotations require evidence:

```python
rotation = service.schedule_rotation(
	tenant_id="tenant-a",
	rotation_id="finance-root-rotation",
	key_id=key["id"],
	requested_by="soc-analyst",
	reason="Compromise signal.",
)

completed = service.complete_rotation(
	tenant_id="tenant-a",
	rotation_id=rotation["id"],
	actor="key-admin",
	evidence="audit://keym/finance-root/rotation",
)
```

Register accountable key agents before allowing AI participation in KEYM
governance workflows:

```python
agent = service.register_key_agent(
	tenant_id="tenant-a",
	agent_id="compromise-agent",
	name="Compromise Reviewer",
	runtime="opencode",
	role="compromise-responder",
	scope="compromised key response review",
	owner="secops",
	purpose="review key compromise evidence and rotation readiness",
	human_approval_required=True,
)
```

Batch lifecycle mutations must be accepted through Bytewax:

```python
batch = service.validate_key_lifecycle_batch(
	tenant_id="tenant-a",
	event_stream="bytewax",
	mutation_count=4,
)
```

## API Helpers

`api.py` exposes:

- `capability_status`
- `create_managed_key`
- `evaluate_key_operation`
- `request_export_approval`
- `decide_export_approval`
- `request_rotation_exception`
- `decide_rotation_exception`
- `schedule_rotation`
- `complete_rotation`
- `mark_key_compromised`
- `register_key_agent`
- `validate_key_lifecycle_batch`
- `list_key_posture`

## UI View Models

`view_models.py` provides package-ready models for:

- dashboard
- inventory
- lifecycle workbench
- export approval queue
- rotation exception queue
- HSM attestation console
- compromise console
- audit timeline
- analytics
- key-agent roster
- settings

## Agent Guardrails

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `key_policy_reviewer`, `key_lifecycle_reviewer`,
`key_custody_reviewer`, `export_reviewer`,
`rotation_exception_reviewer`, `compromise_responder`, and
`hsm_attestation_reviewer`. Privileged roles require human approval:
`export_reviewer`, `rotation_exception_reviewer`, `compromise_responder`, and
`hsm_attestation_reviewer`.

Every key agent must declare owner, purpose, scope, and contribution
disclosure. The service rejects unsupported runtimes, unsupported roles, missing
scope, missing disclosure, and privileged registrations without human approval.

## Adapter Boundaries

The dependency-light runtime intentionally avoids direct live dependencies on
HSM, KMS, cloud key stores, vaults, APG ENCR/SECU services, blockchain audit
ledgers, AI lifecycle managers, SIEM, SOAR, DLP, GRC, monitoring, and
notification providers. Add those systems as adapters that call the current
service methods and preserve fail-closed guardrails.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/keym/__init__.py capabilities/common/keym/models.py capabilities/common/keym/service.py capabilities/common/keym/api.py capabilities/common/keym/capability_contract.py capabilities/common/keym/app.py capabilities/common/keym/view_models.py capabilities/common/keym/tests/test_capability_contract.py capabilities/common/keym/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/keym/tests/test_capability_contract.py capabilities/common/keym/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/keym --json
./.venv/bin/apg capabilities publish-plan capabilities/common/keym --json
```
