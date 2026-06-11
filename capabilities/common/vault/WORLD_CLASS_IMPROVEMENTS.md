# World-Class Improvements: PCI DSS Vault Capability

© 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke

---

## 1. Replace XOR Cipher with AES-256-SIV (Deterministic AEAD)

**Current state**: `_xor_encrypt` is a repeating-key XOR — trivially broken with two known plaintexts.
**Improvement**: Use `cryptography` library's `AESSIV` (RFC 5297). SIV mode is deterministic, so the same PAN always maps to the same ciphertext, enabling indexed lookups without a separate index table. Nonce is derivable, no nonce management headaches.
**PCI DSS impact**: Satisfies Requirement 3.5.1 (strong cryptography) and 3.6 (key management).

---

## 2. Key Hierarchy with HKDF + Per-Tenant Key Derivation

**Current state**: All tenants share a single vault key derived from one env var.
**Improvement**: Derive per-tenant DEKs (Data Encryption Keys) from a master KEK using HKDF-SHA256 with `info = b"vault:" + tenant_id.encode()`. Store DEKs in memory only; rotate by re-deriving. Wraps neatly into a KMS envelope pattern.
**PCI DSS impact**: Satisfies Requirement 3.7 (key rotation) and enforces tenant isolation cryptographically, not just by table filter.

---

## 3. Vault Transit Backend Abstraction

**Current state**: Encryption logic is hardcoded in the service.
**Improvement**: Introduce a `VaultBackend` protocol with `encrypt(plaintext: bytes, key_id: str) -> bytes` / `decrypt(ciphertext: bytes, key_id: str) -> bytes`. Provide three implementations: `LocalAESBackend` (default), `HashiCorpVaultBackend` (calls Transit API), `AWSKMSBackend`. Switch via `APG_VAULT_BACKEND` env var. Zero-downtime key rotation via re-encrypt path.
**PCI DSS impact**: Enables use of a certified HSM/KMS — PCI DSS Requirement 3.6.1.

---

## 4. Token Aliasing + Re-Tokenization Without PAN Re-Entry

**Current state**: Each `tokenize_pan` call stores a new token; no concept of stable alias.
**Improvement**: Add `get_or_create_token(pan)` that returns a stable canonical token for a PAN (idempotent). Separately, expose `rekey_token(old_token, new_key_id)` for in-place re-encryption without surfacing the PAN. Enables zero-downtime key rotation across millions of tokens.
**PCI DSS impact**: Critical for Requirement 3.7.1 (cryptographic key retirement).

---

## 5. Token Metadata Enrichment: BIN Intelligence

**Current state**: `_detect_card_type` uses a 20-line heuristic.
**Improvement**: Replace with a BIN lookup service against a local `apg_bin_registry` table (loadable from public BIN lists). Expose `get_bin_info(bin_prefix)` returning issuer name, country, card scheme, prepaid flag, product type. Cache results in `BoundedCache`. This enables intelligent routing in `fintech_gwy` without detokenization.
**PCI DSS impact**: Routing decisions made on token BIN, never on PAN — keeps non-CDE systems PAN-free.

---

## 6. Token Expiry + Lifecycle Management

**Current state**: Tokens live forever; no expiration or revocation.
**Improvement**: Add optional `expires_at` timestamp to `TokenRecord` and the `apg_token_vault` DB table. Implement `expire_token(token)`, `revoke_token(token, reason)`, `is_token_active(token)`. Background task (APScheduler or asyncio) to purge expired tokens and publish `token_expired` NATS events. Supports PCI DSS data retention limits.
**PCI DSS impact**: Requirement 3.3.1 (SAD data not stored post-auth) analogue for tokens.

---

## 7. Bulk Tokenization with Streaming + Backpressure

**Current state**: `tokenize_batch` likely iterates sequentially (not yet implemented).
**Improvement**: `tokenize_pan_stream(pans: AsyncIterable[str]) -> AsyncGenerator[TokenRecord]` with configurable concurrency (`asyncio.Semaphore`), error isolation per PAN (partial failure doesn't abort batch), and structured result envelope `BatchResult(succeeded, failed, errors)`. DB writes batched with `executemany` for O(N) instead of O(N) round-trips.
**PCI DSS impact**: Enables bulk migration of legacy PAN stores to token stores.

---

## 8. OPA Policy Caching with TTL + Circuit Breaker

**Current state**: Every detokenization call hits OPA synchronously, with a simple `try/except` fallback.
**Improvement**: Cache OPA `allow/deny` decisions in `BoundedCache` with 60s TTL (configurable). Wrap OPA calls in a circuit breaker (3 failures → open, 30s cooldown). Record OPA call latency in `prometheus_client` counters. Fail-closed option: `OPA_FAIL_CLOSED=true` denies access if OPA is unreachable (PCI strict mode).
**PCI DSS impact**: Requirement 7.3 (access control system up-to-date); OPA failure should not silently grant access.

---

## 9. Audit Log with Tamper-Evident Chaining

**Current state**: Audit events are NATS publications — fire-and-forget, no persistence, no chain of custody.
**Improvement**: Persist audit events to `apg_vault_audit` table. Each row contains a `prev_hash` HMAC of the previous row's content — creating a hash chain detectable tampering (similar to Bitcoin blocks, but for compliance). Expose `verify_audit_chain(from_ts, to_ts)` returning integrity status.
**PCI DSS impact**: Requirement 10.3 (audit log protection against modification) — tamper-evident log satisfies QSA requirements.

---

## 10. Masked PAN Normalization + Display Tokenization

**Current state**: `masked_pan` format is hardcoded as `BIN + Xs + last4`.
**Improvement**: Support configurable display formats via `MaskingPolicy`: `LAST4_ONLY` ("****1111"), `BIN_LAST4` ("411111XXXXXX1111"), `FIRST1_LAST4` ("4***1111"), `FULL_MASK` ("****************"). Enables UI components to request the appropriate mask without service changes. Adds `format_masked_pan(token, policy)` method.
**PCI DSS impact**: Requirement 3.3 (mask PAN when displayed) — policy-driven masking enforced at service layer.

---

## 11. Cryptographic Token Binding to Device/Session

**Current state**: Tokens are pure PAN aliases with no binding to context.
**Improvement**: Add `network_token(pan, device_id, session_id)` that generates a token bound to a device fingerprint (HMAC of device_id into the middle section). A token issued for device A cannot be replayed on device B — the vault validates the binding on use. Aligns with EMVCo Network Tokenization specification.
**PCI DSS impact**: Reduces token replay attack surface — satisfies Requirement 3.5.1(b) advanced controls.

---

## 12. Prometheus Metrics + OpenTelemetry Traces

**Current state**: No observability beyond Python logging.
**Improvement**: Add `prometheus_client` counters/histograms: `vault_tokenizations_total`, `vault_detokenizations_total`, `vault_detokenization_denied_total`, `vault_operation_latency_seconds`. Wrap key operations in `opentelemetry.trace.get_tracer("vault")` spans with `token_prefix` and `card_type` attributes. Zero PAN leakage in trace attributes.
**PCI DSS impact**: Requirement 10.2 (audit log for all access to cardholder data) — metrics serve as real-time alerting layer.

---

## 13. Token Portability: Cross-Tenant Token Transfer Protocol

**Current state**: Tokens are strictly scoped to one `tenant_id`; no inter-tenant transfer.
**Improvement**: `transfer_token(token, source_tenant, dest_tenant, authorization_token)` re-encrypts the PAN under the destination tenant's DEK and creates a new record. `authorization_token` is a JWT signed by a root CA verifying the transfer is authorized. Immutable transfer log appended to audit chain.
**PCI DSS impact**: Enables payment processor acquisitions and partner data shares without PAN exposure.

---

## 14. FFX (Format-Preserving Encryption) as Token Generation Backend

**Current state**: Token middle section is `secrets.randbelow(10)` — purely random, no reversible mapping.
**Improvement**: Implement FF3-1 (NIST SP 800-38G Rev 1) format-preserving encryption over the PAN. Tokens become reversible without a vault lookup table — the "vault" becomes a key store only, not a token-to-PAN mapping store. Dramatically reduces DB footprint (no rows needed per token) and eliminates vault availability as a SPOF for detokenization.
**PCI DSS impact**: NIST-approved construction (FF3-1) satisfies Requirement 3.5.1 and eliminates the vault DB as a breach target.

---

## 15. Zero-Knowledge Token Validation (Token Attestation)

**Current state**: Any caller can attempt detokenization and learn whether a token exists via `KeyError`.
**Improvement**: Add `attest_token(token, commitment)` — caller proves they know the PAN without revealing it (using a HMAC commitment scheme). Vault verifies `HMAC(pan, challenge) == commitment` and returns a signed attestation. The PAN never leaves the vault; the caller proves knowledge. Useful for fraud checks where counterparty must prove card ownership without sharing raw PAN.
**PCI DSS impact**: Enables zero-knowledge proofs of card ownership — reduces PAN sharing across systems entirely, beyond the current tokenization model.
