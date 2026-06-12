# APG Encryption Services Capability (v2.0)

Encryption Services (`encr`) is APG's cryptographic governance capability for
generated applications. It gives application builders a dependency-light
runtime for key-domain posture, crypto operation decisions, legacy algorithm
review, threat-adaptive key rotation, UI composition, and audit evidence.

Two service classes are provided:

- `APGEncryptionService` — async, 45+ methods, full feature set (AES-GCM,
  envelope encryption, field-level, database, transit, homomorphic stubs, ZK
  proofs, certificates, HSM stubs, bulk operations, audit, compliance).
- `EncrService` — synchronous shim wrapping `APGEncryptionService` via a
  private event loop; use this in generated APG applications that need portable
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
- First-class crypto-agent composition for policy, lifecycle, entropy,
  exception, threat-rotation, and homomorphic-compute review.
- Durable review evidence for review-required operations, crypto exception
  reviews, key rotations, privileged crypto agents, denied lifecycle batches,
  and audit events.
- Bytewax lifecycle stream enforcement for grouped crypto mutations.
- Envelope encryption (DEK + KEK pattern), field-level encryption,
  transparent database column encryption, and Vault-style transit encryption.
- Homomorphic computation stubs (BFV scheme, add/multiply/statistics ops).
- Fiat-Shamir ZK proof generation and verification.
- Certificate signing, verification, and revocation (stub CA).
- HSM operation stubs (generate, sign, verify, encrypt, decrypt).
- Bulk encrypt, bulk decrypt, and bulk key rotation via `asyncio.gather`.
- FIPS-140-2 and NIST-PQC compliance checks.
- Key usage analytics, CSV/JSON export, and dashboard KPIs.
- API helpers and UI view models for generated APG applications.
- Contract, theme, semantic model, and release evidence for APG composition
  tooling.

## Quick Start

```python
import asyncio
from capabilities.common.encr.service import APGEncryptionService

async def main():
    svc = APGEncryptionService(actor_id="dev", tenant_id="tenant-a")
    await svc.initialize()

    # Generate a key
    key = await svc.key_generate("tenant-a", "my-key-1",
                                  algorithm="AES-256-GCM",
                                  classification="confidential")

    # Encrypt data
    record = await svc.encrypt_data("tenant-a", b"Hello, World!", "my-key-1")

    # Decrypt data
    plaintext = await svc.decrypt_data("tenant-a", record["id"])
    assert plaintext == b"Hello, World!"

asyncio.run(main())
```

Synchronous usage (generated APG apps):

```python
from capabilities.common.encr.service import EncrService

service = EncrService()
service.key_generate("tenant-a", "my-key-1")
record = service.encrypt_data("tenant-a", b"secret", "my-key-1")
```

Governance workflow (key domain + exception handling):

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

# Legacy algorithm requires an exception review
legacy_op = service.evaluate_crypto_operation(
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
    operation_id=legacy_op["id"],
    requested_by="integration-owner",
    reason="Partner migration window.",
)

service.decide_crypto_exception(
    tenant_id="tenant-a",
    review_id=review["id"],
    reviewer="crypto-reviewer",
    decision="approved",
    notes="Approved for 30-day migration window.",
)
```

## API Reference

### Core Crypto Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `encrypt_data` | `(tenant_id, plaintext: bytes, key_id, algorithm, context)` | AES-256-GCM encryption with audit log |
| `decrypt_data` | `(tenant_id, ciphertext_id, context)` | Authenticated decryption |
| `re_encrypt` | `(tenant_id, ciphertext_id, new_key_id, context)` | Key rotation for existing ciphertexts |
| `bulk_encrypt` | `(tenant_id, items: list[{id, plaintext_b64}], key_id)` | Parallel encryption via `asyncio.gather` |
| `bulk_decrypt` | `(tenant_id, ciphertext_ids)` | Parallel decryption with per-item error handling |
| `envelope_encrypt` | `(tenant_id, plaintext, kek_id)` | One-time DEK encrypted under KEK |
| `envelope_decrypt` | `(tenant_id, envelope_id)` | Unwrap DEK and decrypt payload |
| `field_level_encrypt` | `(tenant_id, record_id, fields: dict[str,str], key_id)` | Per-field encryption, cleartext fields untouched |
| `database_encrypt` | `(tenant_id, table_name, column_name, row_id, value, key_id)` | Transparent database cell encryption |
| `transit_encrypt` | `(tenant_id, plaintext, context)` | Vault-style ephemeral key encryption |
| `secret_encrypt` | `(tenant_id, secret_name, secret_value, key_id)` | Named application secret storage |
| `secret_decrypt` | `(tenant_id, secret_name)` | Named secret retrieval |

### Key Lifecycle

| Method | Signature | Description |
|--------|-----------|-------------|
| `key_generate` | `(tenant_id, key_id, algorithm, classification, owner, expires_days)` | Generate and store a 256-bit key |
| `key_rotate` | `(tenant_id, key_id, reason)` | Re-key and notify owner via audit channel |
| `key_revoke` | `(tenant_id, key_id, reason)` | Mark key revoked; blocks future use |
| `key_delete` | `(tenant_id, key_id, confirmed=True)` | Permanent deletion (requires prior revocation) |
| `key_import` | `(tenant_id, key_id, algorithm, key_material_b64, ...)` | Import externally generated key |
| `key_export` | `(tenant_id, key_id, wrapping_key_id)` | Wrapped key export for transport |
| `key_wrap` | `(tenant_id, key_to_wrap_id, wrapping_key_id)` | KEK wrapping of a DEK |
| `key_list` | `(tenant_id)` | List keys without key material |
| `key_metadata_update` | `(tenant_id, key_id, updates)` | Mutate owner/classification/expires_at/tags |
| `key_schedule_rotation` | `(tenant_id, key_id, rotation_date, reason)` | Schedule future rotation |
| `bulk_rotate_keys` | `(tenant_id, key_ids, reason)` | Parallel rotation via `asyncio.gather` |
| `list_expiring_keys` | `(tenant_id, within_days=30)` | Keys expiring within window |

### Signing

| Method | Signature | Description |
|--------|-----------|-------------|
| `signing_key_generate` | `(tenant_id, key_id, algorithm, owner)` | Generate HMAC signing key |
| `data_sign` | `(tenant_id, signing_key_id, payload: bytes)` | HMAC-SHA256 signature |
| `data_verify_signature` | `(tenant_id, signature_id, payload: bytes)` | Constant-time signature verification |

### Certificates

| Method | Signature | Description |
|--------|-----------|-------------|
| `certificate_sign` | `(tenant_id, subject, public_key_pem, validity_days, ca_key_id)` | Issue a signed certificate record |
| `certificate_verify` | `(tenant_id, certificate_id)` | Expiry and status check |
| `certificate_revoke` | `(tenant_id, certificate_id, reason)` | Revoke and log at high severity |

### Advanced Crypto

| Method | Signature | Description |
|--------|-----------|-------------|
| `zero_knowledge_proof` | `(tenant_id, statement, witness)` | Fiat-Shamir ZK proof generation |
| `zk_proof_verify` | `(tenant_id, proof_id, statement, witness)` | ZK proof verification |
| `homomorphic_encrypt_stub` | `(tenant_id, plaintext_int, scheme)` | BFV stub (integrate SEAL/HElib in prod) |
| `hsm_integration` | `(tenant_id, hsm_slot, operation, key_label, payload)` | PKCS#11 stub (generate/sign/verify/encrypt/decrypt) |
| `encrypt_quantum_safe_with_session` | `(plaintext, tenant_id, session_id, user_context)` | AES-GCM with quantum-safe session context |

### Policy and Compliance

| Method | Signature | Description |
|--------|-----------|-------------|
| `policy_evaluate` | `(tenant_id, operation, classification, algorithm)` | Inline policy evaluation (deny legacy, require PQC for restricted/critical) |
| `encryption_policy_create` | `(tenant_id, policy_id, name, rules, owner)` | Define a named encryption policy |
| `encryption_policy_list` | `(tenant_id)` | List active policies |
| `compliance_check` | `(tenant_id, framework)` | FIPS-140-2 or NIST-PQC compliance scan |

### Audit and Analytics

| Method | Signature | Description |
|--------|-----------|-------------|
| `crypto_audit` | `(tenant_id, start_date, end_date)` | Operations audit report with event counts |
| `list_audit_events` | `(tenant_id, event_type)` | Filterable raw audit event list |
| `key_usage_analytics` | `(tenant_id)` | Key counts, operation counts, quantum-safe ratio |
| `dashboard_summary` | `(tenant_id)` | KPI dashboard: keys, certs, rotations, severity counts |
| `health_check` | `()` | Smoke test + collection size snapshot |
| `export_csv` | `(tenant_id, collection)` | CSV export of any store collection |
| `export_json` | `(tenant_id, collection)` | JSON export of any store collection |

### Governance (via `EncrService`)

| Method | Description |
|--------|-------------|
| `register_key_domain` | Tenant-scoped key domain with classification and entropy quality |
| `evaluate_crypto_operation` | Policy decision for encrypt/decrypt/export/generate |
| `request_crypto_exception` | Submit legacy algorithm exception for review |
| `decide_crypto_exception` | Approve or deny an exception |
| `schedule_key_rotation` | Threat-adaptive rotation with evidence |
| `complete_key_rotation` | Mark rotation complete with evidence URI |
| `register_crypto_agent` | Register AI agent with scope, role, and owner |
| `validate_crypto_lifecycle_batch` | Enforce Bytewax routing for batch mutations |
| `list_pending_reviews` | Active review queue for governance console |
| `list_crypto_posture` | Quantum-safety and algorithm posture across domains |

## World-Class Enhancements (v2.0)

These 15 improvements define the production upgrade path from the current
prototype to regulated-grade cryptographic infrastructure.

1. **Persistent StorageBackend Protocol** — `InMemoryBackend` / `PostgresBackend` / `RedisBackend`
   injected via `APGEncryptionService(backend=...)`. Production persistence with transaction
   semantics and horizontal scaling; existing call sites unchanged.

2. **Authenticated Envelope Format v2 (APG-AEv2)** — 16-byte header with magic bytes, algorithm
   ID, flags, and tenant hash prepended to every ciphertext. AAD binds tenant + record + header.
   Backward-compatible decoder auto-routes v1 blobs.

3. **Derived-Key Hierarchy (HKDF per Domain)** — Per-tenant `MasterKeyMaterial` in HSM stub or
   Vault; domain DEKs derived via `HKDF-SHA256(ikm=master, salt=tenant_id, info=domain+version)`.
   Rotation is an O(1) metadata version bump, not mass re-encrypt.

4. **Streaming / Chunked Encryption for Large Payloads** — `encrypt_stream` / `decrypt_stream`
   using AES-GCM with 64 KB chunks and per-chunk sequence number in AAD. Constant memory
   footprint; pipeline-ready for S3, GCS, PostgreSQL COPY.

5. **Real PKCS#11 / CloudHSM Adapter** — `PKCS11HsmAdapter` (`python-pkcs11`) and
   `CloudHSMAdapter` (`boto3`). Stub remains the default. Enables FIPS 140-3 Level 3 and
   PCI-DSS Level 1 key storage.

6. **X.509 Certificate Authority with CRL / OCSP** — Real DER/PEM X.509 v3 certificates via
   `cryptography.x509.CertificateBuilder`. In-memory CRL updated on revocation. OCSP responder
   at `GET /encr/ocsp`. Usable for mTLS and JWT verification without an external PKI service.

7. **Argon2id Passphrase Key Derivation** — `key_derive_from_passphrase(passphrase, salt_b64,
   memory_cost, time_cost, parallelism)` using `argon2-cffi`. Enables passphrase-protected export
   bundles and user-controlled encryption in zero-trust architectures.

8. **Async Key Expiry Daemon with Alerting** — `KeyExpiryMonitor` asyncio task (default: 1 h
   interval) that scans expiring keys, emits audit events, fires configured notification channels,
   and optionally calls `key_rotate` when `auto_rotate=True` is set in key metadata.

9. **Shamir's Secret Sharing for Key Escrow** — Proper (k, n) SSS over GF(2^8) replacing XOR
   shares. `key_split(key_id, k, n)` and `key_reconstruct(shares)` with enforcement that
   reconstruction requires exactly k distinct custodian shares.

10. **Post-Quantum KEM via liboqs** — `OQSAdapter` wrapping `liboqs-python` for real Kyber-768
    KEM (`encapsulate` / `decapsulate`) and Dilithium-3 signatures. Graceful fallback to stub
    when liboqs is absent. Ready for PQC hybrid TLS 1.3.

11. **Tamper-Evident Audit Hash Chain** — Each audit record carries `prev_hash: sha256(prev_record_json)`.
    `audit_chain_verify(tenant_id)` recomputes the chain. Deletion blocked on the audit collection.
    Merkle root export for OpenTimestamps notarization. Suitable for SOC 2 Type II.

12. **OPA / Rego Policy Engine** — `PolicyEngine` with `InlineBackend` (current Python
    conditionals) and `OPABackend` (HTTP POST to `/v1/data`). Policy rules are Rego bundles
    stored via `encryption_policy_create`. Changes deploy without code releases.

13. **Multi-Tenant Envelope-Per-Tenant Master Key (TMK)** — All per-tenant DEKs AES-GCM encrypted
    under the TMK before storage. TMK sealed by HSM or root KMS key. A full database dump cannot
    decrypt another tenant's data without their TMK.

14. **gRPC / Protobuf API** — `encr.proto` defines `KeyService`, `CryptoService`, `AuditService`.
    `EncrGRPCServer` adapts async service methods to gRPC handlers. Bidirectional
    `WatchKeyEvents` streaming RPC for real-time key lifecycle events.

15. **Cryptographic Benchmarking CI Gate** — `pytest-benchmark` baselines: `encrypt_data` < 2 ms
    (4 KB), `key_generate` < 1 ms, `bulk_encrypt` (100 items) < 50 ms. CI gate
    (`tests/ci/test_perf_gate.py`) fails if any target regresses by > 20%.

## New Methods

### Envelope Encryption (DEK + KEK)

```python
import asyncio
from capabilities.common.encr.service import APGEncryptionService

async def demo_envelope():
    svc = APGEncryptionService(actor_id="ops", tenant_id="acme")
    await svc.initialize()

    # Key-encryption key (KEK) protects one-time data-encryption keys (DEKs)
    await svc.key_generate("acme", "kek-primary", algorithm="AES-256-GCM", classification="restricted")

    # One call generates a random DEK, encrypts data, and wraps the DEK under the KEK
    envelope = await svc.envelope_encrypt("acme", b"sensitive payload", kek_id="kek-primary")

    # Decryption unwraps the DEK and authenticates the payload in one call
    plaintext = await svc.envelope_decrypt("acme", envelope["id"])
    assert plaintext == b"sensitive payload"

asyncio.run(demo_envelope())
```

### Field-Level Encryption

```python
async def demo_field_level():
    svc = APGEncryptionService(actor_id="app", tenant_id="acme")
    await svc.initialize()
    await svc.key_generate("acme", "pii-key", classification="confidential")

    # Only the nominated fields are encrypted; other columns remain plaintext
    result = await svc.field_level_encrypt(
        "acme",
        record_id="user-4821",
        fields={"ssn": "123-45-6789", "dob": "1985-03-12"},
        key_id="pii-key",
    )
    # result["encrypted_fields"] contains base64(nonce + ciphertext) per field
    # result["field_names"] == ["ssn", "dob"]

asyncio.run(demo_field_level())
```

### Zero-Knowledge Proof

```python
async def demo_zkp():
    svc = APGEncryptionService(actor_id="prover", tenant_id="acme")
    await svc.initialize()

    # Generate a Fiat-Shamir ZK proof
    proof = await svc.zero_knowledge_proof(
        "acme",
        statement="I know the API secret for service X",
        witness="secret-value-never-transmitted",
    )

    # Verify without revealing the witness
    result = await svc.zk_proof_verify(
        "acme",
        proof_id=proof["id"],
        statement="I know the API secret for service X",
        witness="secret-value-never-transmitted",
    )
    assert result["valid"] is True

asyncio.run(demo_zkp())
```

### Bulk Operations

```python
async def demo_bulk():
    import base64
    svc = APGEncryptionService(actor_id="etl", tenant_id="acme")
    await svc.initialize()
    await svc.key_generate("acme", "bulk-key")

    items = [
        {"id": f"row-{i}", "plaintext_b64": base64.b64encode(f"value-{i}".encode()).decode()}
        for i in range(100)
    ]

    # Encrypts all 100 items via asyncio.gather — typically < 50 ms
    records = await svc.bulk_encrypt("acme", items, "bulk-key")

    # Batch rotation of multiple keys
    await svc.key_generate("acme", "key-a")
    await svc.key_generate("acme", "key-b")
    rotation_results = await svc.bulk_rotate_keys("acme", ["key-a", "key-b"], reason="quarterly_rotation")

asyncio.run(demo_bulk())
```

### Compliance Check and Analytics

```python
async def demo_compliance():
    svc = APGEncryptionService(actor_id="auditor", tenant_id="acme")
    await svc.initialize()
    await svc.key_generate("acme", "pqc-key", algorithm="CRYSTALS-Kyber-768")

    # FIPS-140-2 check: fails if any active key uses a legacy algorithm
    fips = await svc.compliance_check("acme", framework="FIPS-140-2")
    print(fips["passed"], fips["issues"])

    # NIST-PQC check: fails if no post-quantum keys are present
    pqc = await svc.compliance_check("acme", framework="NIST-PQC")
    print(pqc["quantum_safe_key_count"])

    analytics = await svc.key_usage_analytics("acme")
    print(f"Quantum-safe ratio: {analytics['quantum_safe_ratio']:.0%}")

asyncio.run(demo_compliance())
```

## Durable Review Evidence

ENCR preserves review state for generated cryptographic governance consoles.
Review-required operations, exception reviews, scheduled rotations, privileged
crypto-agent registrations, lifecycle batch validations, and audit events carry
the same evidence fields:

- `policy_decision`
- `matched_rules`
- `review_reasons`
- `review_evidence`

Generated applications can compose the active review queue:

```python
pending = service.list_pending_reviews("tenant-a")
```

Denied non-Bytewax lifecycle batches are stored through
`list_crypto_lifecycle_batches()` before `PermissionError` is raised, so
operators can see and remediate the routing violation.

## Agent Guardrails

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.

Supported roles: `crypto_policy_reviewer`, `key_lifecycle_reviewer`,
`entropy_reviewer`, `exception_reviewer`, `threat_rotation_reviewer`,
`homomorphic_compute_reviewer`.

Privileged roles requiring human approval: `exception_reviewer`,
`threat_rotation_reviewer`, `homomorphic_compute_reviewer`.

Every crypto agent must declare `owner`, `purpose`, `scope`, and contribution
disclosure. The service rejects unsupported runtimes, unsupported roles, missing
scope, and missing disclosure. Privileged registrations without human approval
are retained as `pending_review` evidence.

## Adapter Boundaries

The local runtime intentionally avoids direct dependencies on live HSM, KMS,
cloud KMS, Vault, APG KEYM, post-quantum SDK, entropy hardware, ZK prover,
homomorphic computation, SIEM, SOAR, DLP, GRC, or AI policy services. Add those
systems as adapters that call the current service methods and preserve the
fail-closed guardrails.

## UI View Models

`views.py` provides Pydantic models for:

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
- crypto-agent roster
- settings

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/encr/__init__.py \
    capabilities/common/encr/models.py \
    capabilities/common/encr/service.py \
    capabilities/common/encr/api.py \
    capabilities/common/encr/views.py \
    capabilities/common/encr/capability_contract.py \
    capabilities/common/encr/app.py \
    capabilities/common/encr/tests/test_capability_contract.py \
    capabilities/common/encr/tests/test_package_contract.py

./.venv/bin/pytest -q \
    capabilities/common/encr/tests/test_capability_contract.py \
    capabilities/common/encr/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/encr --json
./.venv/bin/apg capabilities publish-plan capabilities/common/encr --json
```

---

© 2025 Datacraft — www.datacraft.co.ke
