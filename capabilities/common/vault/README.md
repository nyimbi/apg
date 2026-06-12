# APG Vault (PCI DSS Tokenization) (`vault`)

**Version**: 2.0.0 | **Domain**: common

## Overview

PCI DSS format-preserving tokenization for cardholder PANs. Tokens preserve BIN and last 4 digits; Luhn-valid. OPA-gated detokenization. Full token lifecycle management (expire, revoke, rekey). Zero-knowledge attestation for cross-system ownership proofs.

## Core API

```python
from capabilities.common.vault.service import TokenizationService, MaskingPolicy

svc = TokenizationService(tenant_id="payment_processor")

# Tokenize
record = await svc.tokenize_pan("4111111111111111")

# Idempotent — returns same token for same PAN
record2 = await svc.get_or_create_token("4111111111111111")
assert record.token == record2.token

# Detokenize (PCI-authorized roles only)
pan = await svc.detokenize_pan(record.token, requester_role="pci_authorized", requester_id="svc")

# Bulk detokenization with partial-failure isolation
result = await svc.detokenize_batch(tokens, requester_role="pci_authorized", requester_id="svc")
# result.succeeded / result.failed / result.success_rate

# Streaming bulk tokenization (backpressure-safe, bounded concurrency)
async for item in svc.tokenize_pan_stream(pan_async_generator):
    if isinstance(item, TokenRecord):
        store(item.token)

# Lifecycle management
await svc.expire_token(token, ttl_seconds=3600)
await svc.revoke_token(token, reason="card_reported_stolen")
active = await svc.is_token_active(token)

# Zero-downtime key rotation
await svc.rekey_token(token, new_vault_key="new-secure-key")

# Token inspection without PCI auth
meta = await svc.get_token_metadata(token)

# Configurable display masking
masked = await svc.format_masked_pan(token, MaskingPolicy.LAST4_ONLY)  # "************1111"

# Zero-knowledge ownership attestation
result = await svc.attest_token(token, commitment=challenge_hex + hmac_hex)

# Live compliance snapshot
status = await svc.get_compliance_status()
```

## Token Properties

- Same length as original PAN (13-19 digits)
- Preserves BIN (first 6 digits) for payment routing
- Preserves last 4 digits for cardholder display
- Luhn-valid (prevents PAN scanners from triggering on tokens)
- Random middle section — not reversible without vault lookup

## Token Lifecycle States

`ACTIVE` -> `EXPIRED` (TTL or manual) | `ACTIVE` -> `REVOKED` (permanent)

## Masking Policies

| Policy | Example | PCI DSS Reference |
|--------|---------|-------------------|
| `BIN_LAST4` | `411111XXXXXX1111` | Req 3.3 (default) |
| `LAST4_ONLY` | `************1111` | Req 3.3 (strict) |
| `FIRST1_LAST4` | `4***1111` | Req 3.3 |
| `FULL_MASK` | `****************` | Req 3.3 (max) |

## Governance Rules

- tenant_context_required
- operation_type_required
- audit_logged
- access_controlled

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `APG_VAULT_KEY` | _(dev key — DO NOT USE IN PROD)_ | Vault encryption key |
| `OPA_URL` | _(none)_ | OPA endpoint for PCI DSS authorization |
| `NATS_URL` | _(none)_ | NATS for audit events |

## License

(c) 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements: PCI DSS Vault Capability
- **I2.** Replace XOR Cipher with AES-256-SIV (Deterministic AEAD)
- **I3.** Key Hierarchy with HKDF + Per-Tenant Key Derivation
- **I4.** Vault Transit Backend Abstraction
- **I5.** Token Aliasing + Re-Tokenization Without PAN Re-Entry
- **I6.** Token Metadata Enrichment: BIN Intelligence
- **I7.** Token Expiry + Lifecycle Management
- **I8.** Bulk Tokenization with Streaming + Backpressure
- **I9.** OPA Policy Caching with TTL + Circuit Breaker
- **I10.** Audit Log with Tamper-Evident Chaining
- **I11.** Masked PAN Normalization + Display Tokenization
- **I12.** Cryptographic Token Binding to Device/Session
- **I13.** Prometheus Metrics + OpenTelemetry Traces
- **I14.** Token Portability: Cross-Tenant Token Transfer Protocol
- **I15.** FFX (Format-Preserving Encryption) as Token Generation Backend

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
