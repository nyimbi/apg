# APG Encryption Services Capability

Encryption Services (`encr`) is APG's cryptographic governance capability for
generated applications. It gives application builders a dependency-light
runtime for key-domain posture, crypto operation decisions, legacy algorithm
review, threat-adaptive key rotation, UI composition, and audit evidence.

The package includes advanced async encryption engines, but generated APG
applications should use the synchronous `EncrService` when they need portable
package behavior without live HSM, KMS, post-quantum SDK, zero-knowledge,
homomorphic, or KEYM integrations.

## What ENCR Provides

- Tenant-scoped key domains with owner, algorithm, classification, entropy
  quality, quantum-safety state, and rotation state.
- Crypto operation decisions for encrypt, decrypt, export, compute, and
  generate-key workflows.
- Deterministic guardrails for tenant context, restricted-data quantum safety,
  plaintext export denial, entropy thresholds, legacy algorithm review, and
  threat-adaptive rotation.
- Crypto exception review with independent reviewer and reviewer notes.
- Key rotation scheduling and completion with evidence.
- API helpers and UI view models for generated APG applications.
- Contract, theme, semantic model, and release evidence for APG composition
  tooling.

## Service Usage

```python
from capabilities.common.encr.service import EncrService

service = EncrService()
domain = service.register_key_domain(
	tenant_id="tenant-a",
	domain_id="finance-pii",
	name="Finance PII",
	owner="security-admin",
	algorithm="CRYSTALS-Kyber-768",
	data_classification="restricted",
	entropy_quality=0.99,
)

operation = service.evaluate_crypto_operation(
	tenant_id="tenant-a",
	operation_id="encrypt-invoice",
	operation_type="encrypt",
	key_domain_id=domain["id"],
	data_classification="restricted",
)
```

Legacy algorithms require explicit review:

```python
legacy = service.evaluate_crypto_operation(
	tenant_id="tenant-a",
	operation_id="legacy-partner",
	operation_type="encrypt",
	key_domain_id=domain["id"],
	algorithm="RSA-2048",
	algorithm_family="legacy",
	data_classification="internal",
)

review = service.request_crypto_exception(
	tenant_id="tenant-a",
	review_id="legacy-partner-review",
	operation_id=legacy["id"],
	requested_by="integration-owner",
	reason="Partner migration window.",
)

approved = service.decide_crypto_exception(
	tenant_id="tenant-a",
	review_id=review["id"],
	reviewer="crypto-reviewer",
	decision="approved",
	notes="Approved for 30-day migration window.",
)
```

Threat-adaptive rotation requires evidence before sensitive operations proceed:

```python
rotation = service.schedule_key_rotation(
	tenant_id="tenant-a",
	rotation_id="finance-pii-rotation",
	key_domain_id=domain["id"],
	requested_by="soc-analyst",
	reason="Active compromise signal.",
)

completed = service.complete_key_rotation(
	tenant_id="tenant-a",
	rotation_id=rotation["id"],
	actor="key-admin",
	evidence="audit://encr/finance-pii/rotation",
)
```

## API Helpers

`api.py` exposes a shared dependency-light service:

- `capability_status`
- `register_key_domain`
- `evaluate_crypto_operation`
- `request_crypto_exception`
- `decide_crypto_exception`
- `schedule_key_rotation`
- `complete_key_rotation`
- `list_crypto_posture`
- compatibility `create_record` and `list_records`

## UI View Models

`views.py` provides models for:

- dashboard
- operations console
- key-domain console
- policy designer
- entropy console
- crypto exception queue
- key rotation console
- homomorphic workspace
- analytics
- audit timeline
- settings

## Adapter Boundaries

The local runtime intentionally avoids direct dependencies on live HSM, KMS,
cloud KMS, vault, APG KEYM, post-quantum SDK, entropy hardware, ZK prover,
homomorphic computation, SIEM, SOAR, DLP, GRC, or AI policy services. Add those
systems as adapters that call the current service methods and preserve the
fail-closed guardrails.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/encr/__init__.py capabilities/common/encr/models.py capabilities/common/encr/service.py capabilities/common/encr/api.py capabilities/common/encr/views.py capabilities/common/encr/capability_contract.py capabilities/common/encr/app.py capabilities/common/encr/tests/test_capability_contract.py capabilities/common/encr/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/encr/tests/test_capability_contract.py capabilities/common/encr/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/encr --json
./.venv/bin/apg capabilities publish-plan capabilities/common/encr --json
```
