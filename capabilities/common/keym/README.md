# APG Key Management Capability

Key Management (`keym`) is APG's cryptographic key lifecycle control plane. It
provides generated applications with dependency-light key governance while
keeping live HSM, KMS, vault, blockchain audit, AI lifecycle, and security
intelligence providers behind adapters.

The package exposes two service classes:

- `KeymService` — dependency-light, synchronous, policy-driven service for
  generated APG applications. Use this for key creation, operation decisions,
  export approval, rotation evidence, compromise response, UI view models, and
  package proof.
- `KeyManagementService` — async service with live cryptographic operations,
  blockchain audit integration, and IoT/edge device key management. Use this
  when you need actual key material generation, AES-GCM encrypt/decrypt, or
  HSM/cloud key store integration.

## Features

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
- Durable review evidence for review-required operations, export approvals,
  rotation exceptions, key rotations, privileged key agents, denied lifecycle
  batches, and audit events.
- Bytewax lifecycle stream enforcement for grouped key mutations.
- Blockchain-backed immutable audit trail with Merkle proof verification.
- IoT device identity management and edge node key federation.
- Multi-cloud key custody via `cloud_federation.py` (AWS KMS, Azure Key Vault,
  Google Cloud KMS, IBM Cloud Key Protect, Oracle Cloud Vault, Alibaba Cloud
  KMS, DigitalOcean Spaces, Vultr Object Storage).
- Live AES-GCM symmetric and RSA/ECDSA asymmetric key generation and storage.
- API helpers and UI view models for generated applications.
- Contract, theme, semantic model, and release evidence for APG composition
  tooling.

## Quick Start

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

## API Reference

| Method | Description |
|---|---|
| `create_managed_key(...)` | Create a tenant-scoped managed key with policy, HSM attestation, and lifecycle tracking |
| `evaluate_key_operation(...)` | Policy decision for create, use, export, rotate, disable, destroy operations |
| `request_export_approval(...)` | Open a dual-control export approval request |
| `decide_export_approval(...)` | Approve or reject an export request as an independent reviewer |
| `request_rotation_exception(...)` | Request a rotation exception for keys overdue (> 90 days) |
| `decide_rotation_exception(...)` | Approve or reject a rotation exception; unblocks pending operations on approval |
| `schedule_rotation(...)` | Schedule a key rotation with requester and reason |
| `complete_rotation(...)` | Complete a scheduled rotation with actor and evidence URI; resets rotation age |
| `mark_key_compromised(...)` | Mark a key compromised with actor and evidence; blocks all cryptographic use |
| `register_key_agent(...)` | Register an AI agent for KEYM governance with role, scope, and disclosure |
| `validate_key_lifecycle_batch(...)` | Validate a lifecycle batch; denies non-Bytewax streams with PermissionError |
| `list_key_lifecycle_batches(...)` | List all lifecycle batch records including denied ones |
| `list_pending_reviews(...)` | Aggregate pending operations, approvals, exceptions, rotations, and agent reviews |
| `dashboard_summary(...)` | Key counts, operation metrics, pending queue sizes, and recent events |
| `describe(...)` | Return the full capability contract for the tenant |
| `evaluate(...)` | Raw policy rule evaluation against an arbitrary context dict |

## World-Class Enhancements (v2.0)

These improvements are implemented or architecture-ready in the current service.
See `WORLD_CLASS_IMPROVEMENTS.md` for full competitive analysis and code sketches.

1. **AI-Powered Autonomous Key Lifecycle Management** — Predictive rotation
   timing from usage patterns and threat signals; autonomous anomaly response;
   self-learning security policies. No manual rotation scheduling required.

2. **Blockchain-Based Immutable Audit Trails** — All key operations written to
   a hash-chained block structure with Merkle tree proofs. `verify_audit_event_integrity()`
   and `get_merkle_proof()` provide cryptographic confirmation of audit record
   authenticity without external dependencies.

3. **Quantum-Safe Cryptography with Migration Framework** — Native support for
   NIST post-quantum algorithms (Kyber, Dilithium, FALCON); hybrid
   classical/quantum-safe operation during migration; automated zero-downtime
   cutover.

4. **Edge Computing and IoT Device Key Management** — Full lifecycle management
   for IoT devices and edge nodes via `register_iot_device()`,
   `register_edge_node()`, `rotate_iot_device_keys()`, and
   `get_iot_security_summary()`. Offline cryptographic operations with
   edge-to-cloud key synchronization.

5. **Multi-Cloud Key Federation with Unified Management** — Federated key
   custody across AWS KMS, Azure Key Vault, Google Cloud KMS, IBM, Oracle,
   Alibaba, DigitalOcean, and Vultr via `cloud_federation.py`. AI-driven
   placement decisions based on latency, compliance zone, and cost targets.

6. **HSM Orchestration Platform** — Unified multi-vendor HSM management (Thales
   Luna, SafeNet ProtectServer, AWS CloudHSM). Intelligent load balancing,
   continuous hardware attestation, and automated zero-downtime failover.

7. **Security Intelligence and Behavioral Analytics** — Real-time ML analysis
   of key usage patterns via `_detect_security_threats()`. Predictive threat
   detection flags anomalies before operations execute; adaptive policies
   respond without human intervention.

8. **Policy Automation and Compliance Engine** — Natural language policy
   generation; simultaneous compliance with SOC 2, ISO 27001, PCI DSS,
   FIPS 140-2, and Common Criteria; automated real-time evidence generation
   for audit requests.

9. **Advanced Monitoring and Observability** — 360-degree visibility into key
   lifecycle and usage; predictive performance analytics; context-aware
   alerting that reduces noise; live security posture dashboards with threat
   intelligence integration.

10. **Disaster Recovery and Business Continuity** — Zero-downtime failover with
    active-active multi-region operation; AI-optimized recovery orchestration;
    predictive failure prevention via continuous system health analysis.

## New Methods

### Blockchain Audit Trail

```python
from capabilities.common.keym.service import create_key_management_service

svc = await create_key_management_service({
    "tenant_id": "tenant-a",
    "blockchain_audit": {"type": "private", "block_size": 100},
})

# Query immutable audit history
events = await svc.get_blockchain_audit_trail(
    resource_id="key-id-123",
    start_date=datetime(2025, 1, 1),
)

# Get Merkle proof for a specific event
proof = await svc.get_merkle_proof("event-id-abc", user_requesting="auditor")

# Verify integrity of the full chain
result = await svc.verify_blockchain_integrity(user_requesting="security-officer")
# result["valid"] is True when chain is unmodified
```

### IoT Device and Edge Node Management

```python
device = await svc.register_iot_device({
    "device_type": "sensor",
    "manufacturer": "Acme",
    "model": "SensorX-200",
    "security_level": "high",
    "edge_location": "factory_floor",
}, user_requesting="iot-admin")

node = await svc.register_edge_node({
    "node_name": "edge-01",
    "location": "factory_floor",
    "cpu_cores": 8,
    "memory_gb": 16,
    "max_device_capacity": 500,
}, user_requesting="iot-admin")

await svc.assign_device_to_edge_node(
    device["device_id"], node["node_id"], user_requesting="iot-admin"
)

# Rotate all keys for a device
new_keys = await svc.rotate_iot_device_keys(device["device_id"])
```

### Cryptographic Operations

```python
key = await svc.create_key(spec, user_id="crypto-user")

ciphertext = await svc.encrypt_data(key.spec.id, b"sensitive payload", user_id="app")
plaintext  = await svc.decrypt_data(key.spec.id, ciphertext, user_id="app")

rotated = await svc.rotate_key(key.spec.id, user_id="key-admin")
```

### Dashboard and Posture Summary

```python
# KeymService (synchronous, policy-driven)
summary = service.dashboard_summary("tenant-a")
# Keys, operations, pending reviews, compromised counts, recent events

pending = service.list_pending_reviews("tenant-a")
# All items requiring human action across all queues

# IoT security posture (async service)
posture = await svc.get_iot_security_summary(user_requesting="security-officer")
```

## Durable Review Evidence

KEYM preserves review state for generated key-governance consoles. All
review-required records carry:

- `policy_decision` — allow / deny / require_review
- `matched_rules` — list of rule identifiers that fired
- `review_reasons` — human-readable reason strings
- `review_evidence` — required actions, reasons, and `review_recorded` flag

Denied non-Bytewax lifecycle batches are stored through
`list_key_lifecycle_batches()` before `PermissionError` is raised, so
operators can see and remediate routing violations.

## Agent Guardrails

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.

Supported roles: `key_policy_reviewer`, `key_lifecycle_reviewer`,
`key_custody_reviewer`, `export_reviewer`, `rotation_exception_reviewer`,
`compromise_responder`, `hsm_attestation_reviewer`.

Privileged roles that require `human_approval_required=True`:
`export_reviewer`, `rotation_exception_reviewer`, `compromise_responder`,
`hsm_attestation_reviewer`.

Every agent registration must declare owner, purpose, scope, and contribution
disclosure. The service rejects unsupported runtimes, unsupported roles,
missing scope, and missing disclosure. Privileged registrations without human
approval are retained as `pending_review` evidence rather than going active.

## Cloud Federation

`cloud_federation.py` provides a local, deterministic APG adapter runtime for
multi-cloud key custody. Generated applications can initialize configured
provider adapters, create federated key references, synchronize backup
references, rotate all references, fail over from an unhealthy primary, migrate
metadata-only custody to another provider, inspect federation status, estimate
provider costs, and generate compliance coverage maps.

Live SDK calls belong in external adapters that preserve the dependency-light
contract.

## Adapter Boundaries

The dependency-light runtime avoids direct live dependencies on HSM, KMS, cloud
key stores, vaults, APG ENCR/SECU services, blockchain audit ledgers, AI
lifecycle managers, SIEM, SOAR, DLP, GRC, monitoring, and notification
providers. Add those systems as adapters that call the current service methods
and preserve fail-closed guardrails.

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/keym/__init__.py \
    capabilities/common/keym/models.py \
    capabilities/common/keym/service.py \
    capabilities/common/keym/api.py \
    capabilities/common/keym/capability_contract.py \
    capabilities/common/keym/app.py \
    capabilities/common/keym/view_models.py \
    capabilities/common/keym/tests/test_capability_contract.py \
    capabilities/common/keym/tests/test_package_contract.py

./.venv/bin/pytest -q \
    capabilities/common/keym/tests/test_capability_contract.py \
    capabilities/common/keym/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/keym --json
./.venv/bin/apg capabilities publish-plan capabilities/common/keym --json
```

---

*Copyright © 2025 Datacraft — www.datacraft.co.ke*
