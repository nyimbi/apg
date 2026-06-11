# ENCR — World-Class Improvement Roadmap

15 concrete improvements to elevate `APGEncryptionService` from robust prototype
to production-grade cryptographic infrastructure.

---

## 1. Persistent Backend Abstraction (StorageBackend Protocol)

**Problem**: `_Store` is a pure in-memory dict. Data is lost on restart; no
horizontal scaling; no transaction semantics.

**Improvement**: Define a `StorageBackend` Protocol with `put / get / list /
delete / transaction()`. Ship three implementations:
`InMemoryBackend` (current, for tests), `PostgresBackend` (asyncpg +
table-per-collection), `RedisBackend` (hash per tenant). Inject via
`APGEncryptionService(backend=...)`. All existing call sites stay unchanged.

**Impact**: Production-ready persistence, zero-downtime deploys, multi-replica
deployments.

---

## 2. Authenticated Envelope Format v2 (APG-AEv2)

**Problem**: Ciphertexts are stored as raw base64 blobs with no version tag, no
algorithm negotiation, and no associated-data binding to tenant.

**Improvement**: Prepend a fixed 16-byte header: `APGAEv2 | alg_id(1) |
flags(1) | tenant_hash(8) | nonce(12)` before the AES-GCM tag+ciphertext. AAD
always includes `tenant_id + record_id + header`. Decoder is backward-compatible
with v1 (detects missing magic bytes, routes to legacy path).

**Impact**: Tenant isolation is cryptographically enforced, not just DB-filtered.
Ciphertext migration path is automated.

---

## 3. Derived-Key Hierarchy (HKDF per Key Domain)

**Problem**: All keys are independent 256-bit secrets. There is no hierarchy, no
domain isolation, and no cheap re-keying of domain subtrees.

**Improvement**: Introduce a `MasterKeyMaterial` root (per-tenant, stored in
HSM stub or Vault). Per-domain DEKs are derived via HKDF-SHA256:
`HKDF(ikm=master_key, salt=tenant_id, info=domain_id+version)`. Key rotation
only changes the domain version; historical ciphertexts remain decryptable via
their embedded version tag.

**Impact**: One compromised DEK cannot compromise sibling domains. Rotation is
O(1) metadata update, not mass re-encrypt.

---

## 4. Streaming / Chunked Encryption for Large Payloads

**Problem**: `encrypt_data` loads full plaintext into memory. A 2 GB database
dump will OOM the service.

**Improvement**: Add `encrypt_stream(tenant_id, key_id, reader: AsyncIterator[bytes]) -> AsyncIterator[bytes]`
using AES-GCM in 64 KB chunks with a per-chunk sequence number in AAD to prevent
reordering attacks. Matching `decrypt_stream`. Use `cryptography`'s
`AEADEncryptionContext` for incremental authentication.

**Impact**: Arbitrarily large payloads, constant memory footprint, pipeline
integration (S3, GCS, PostgreSQL COPY).

---

## 5. Real PKCS#11 / CloudHSM Driver Adapter

**Problem**: `hsm_integration` is a pure stub — it hashes the payload locally
and pretends to be an HSM.

**Improvement**: Implement `PKCS11HsmAdapter` using `python-pkcs11` (MIT). The
adapter wraps `pkcs11.Session.{generate_key, sign, verify, encrypt, decrypt}`.
For AWS CloudHSM, add `CloudHSMAdapter` using `boto3.client('cloudhsm')`.
Service picks the adapter from config; stub remains as default.

**Impact**: Real tamper-evident hardware key storage for regulated workloads
(PCI-DSS Level 1, FIPS 140-3 Level 3).

---

## 6. Certificate Authority with CRL / OCSP

**Problem**: `certificate_sign` generates a fingerprint + HMAC signature stored
in a dict. No X.509, no chain of trust, no revocation distribution.

**Improvement**: Use `cryptography.x509.CertificateBuilder` to emit real
DER/PEM X.509 v3 certificates. Maintain an in-memory CRL updated on
`certificate_revoke`. Expose `GET /encr/ocsp` as an OCSP responder using
`cryptography.x509.ocsp`. CA key hierarchy: root CA (offline stub), intermediate
CA (service-held).

**Impact**: Certificates are usable in TLS mutual-auth and JWT verification
flows without a separate PKI service.

---

## 7. Argon2id-Based Key Derivation from Passphrases

**Problem**: No password-based key derivation. Users who want passphrase-derived
keys must roll their own.

**Improvement**: Add `async key_derive_from_passphrase(tenant_id, passphrase, salt_b64, key_id, memory_cost=65536, time_cost=3, parallelism=4)`.
Uses `argon2-cffi` Argon2id. Derived key is stored identically to `key_generate`
output. Add matching `key_verify_passphrase` for re-derivation checks.

**Impact**: Enables passphrase-protected export bundles, user-controlled
encryption in zero-trust architectures.

---

## 8. Async Key Expiry Daemon with Alerting

**Problem**: `list_expiring_keys` is a passive query. Nothing proactively
triggers rotation when keys expire.

**Improvement**: Add `KeyExpiryMonitor` — an `asyncio.Task` that wakes every
configurable interval (default 1 hour), calls `list_expiring_keys`, and for each
expiring key: emits an `audit_log` event, fires `_Notify.send` on configured
channels, and optionally calls `key_rotate` if `auto_rotate=True` was set in
key metadata. Lifecycle: `start_expiry_monitor() / stop_expiry_monitor()`.

**Impact**: Eliminates manual key hygiene. Feeds SIEM/SOAR via audit channel.

---

## 9. Shamir's Secret Sharing for Key Escrow

**Problem**: `ZeroKnowledgeEncryptionEngine.threshold_encrypt` uses XOR
shares — information-theoretically insecure for small share counts.

**Improvement**: Implement proper (k, n) Shamir Secret Sharing over GF(2^8)
(matches the `cryptography` library's internals). Add
`async key_split(tenant_id, key_id, k, n) -> list[ShareRecord]` and
`async key_reconstruct(tenant_id, shares) -> dict`. Enforce that reconstruction
requires exactly k distinct shares from different custodians.

**Impact**: Cryptographically sound M-of-N key escrow for break-glass scenarios
and disaster recovery.

---

## 10. Full Post-Quantum Key Exchange via liboqs / PQClean Adapter

**Problem**: `_PostQuantumCrypto.get_or_create_keypair` uses SHA-256 to fake key
bytes. No actual Kyber or Dilithium operations are performed.

**Improvement**: Add `OQSAdapter` wrapping the `liboqs-python` bindings
(`pip install liboqs`). Falls back gracefully to stub when liboqs is absent
(test environments). Real Kyber-768 KEM: `encapsulate(pk) -> (ciphertext, shared_secret)`,
`decapsulate(sk, ciphertext) -> shared_secret`. Real Dilithium-3: `sign / verify`.

**Impact**: Actual NIST PQC Round 4 algorithms. Ready for post-quantum TLS 1.3
hybrid key exchange.

---

## 11. Structured Audit Log with Tamper Evidence (Append-Only Hash Chain)

**Problem**: Audit records are mutable dicts in `_Store`. An attacker with store
access can delete or alter events.

**Improvement**: Each audit record includes `prev_hash: sha256(prev_record_json)`.
On append, the chain is extended. `audit_chain_verify(tenant_id)` recomputes and
validates the chain. Export the chain as a Merkle tree root for external
notarization (OpenTimestamps stub). Deletion is never allowed on audit collection.

**Impact**: Tamper-evident audit trail suitable for SOC 2 Type II, ISO 27001, and
PCI-DSS audit requirements.

---

## 12. Encryption Policy Engine with OPA / Rego Integration

**Problem**: `policy_evaluate` uses hardcoded Python conditionals. Policy changes
require code deploys.

**Improvement**: Add `PolicyEngine` with two backends: `InlineBackend` (current),
`OPABackend` (HTTP POST to OPA /v1/data endpoint). Policy rules are Rego bundles
stored via `encryption_policy_create`. `policy_evaluate` compiles the active
policy bundle and evaluates in <5 ms using `rego-python` or OPA sidecar.

**Impact**: Security teams can change encryption policy without engineering
involvement. Policy-as-code with version history and rollback.

---

## 13. Multi-Tenant Key Isolation via Envelope-Per-Tenant Master Key

**Problem**: `key_generate` stores keys in a flat dict keyed by `key_id`.
Cross-tenant key confusion is prevented only by Python dict lookup — not
cryptographically.

**Improvement**: Each tenant has a `TenantMasterKey` (TMK) generated once on
first write. All per-tenant DEKs are AES-GCM encrypted under the TMK before
storage. The TMK itself is sealed by the HSM or a root KMS key.
`_require_key` unwraps the DEK at access time. Tenant data in the same
`_Store` backend is therefore cryptographically isolated.

**Impact**: Even a full database dump cannot decrypt another tenant's data without
their TMK.

---

## 14. gRPC / Protobuf API Layer for Microservice Consumption

**Problem**: Only a Flask-AppBuilder REST API is documented. No typed RPC
interface, no streaming, no server-side push for key events.

**Improvement**: Define `encr.proto` with services `KeyService`, `CryptoService`,
`AuditService`. Generate Python stubs via `grpcio-tools`. Add
`EncrGRPCServer(APGEncryptionService)` that adapts async service methods to gRPC
handlers. Include a bidirectional `WatchKeyEvents` streaming RPC for real-time
key lifecycle events.

**Impact**: Language-agnostic microservice consumption. Zero-copy binary
serialization. Native support for service mesh mTLS.

---

## 15. Cryptographic Benchmarking and Regression CI Gate

**Problem**: There is no performance baseline. A change to `encrypt_data`
could silently regress throughput by 10x.

**Improvement**: Add `tests/bench/bench_encr.py` using `pytest-benchmark`.
Baseline targets: `encrypt_data` < 2 ms for 4 KB payload, `key_generate` < 1 ms,
`bulk_encrypt` (100 items) < 50 ms. CI gate (`tests/ci/test_perf_gate.py`) fails
if any target regresses by >20%. Benchmarks run in GitHub Actions on a
`c5.xlarge`-equivalent runner with pinned CPU affinity.

**Impact**: Performance regressions are caught at PR time, not production. Gives
operators a published SLA for encryption throughput.
