# Vault - PCI DSS Tokenization User Guide

## Overview

The `vault` capability implements format-preserving tokenization (FPT) for payment card Primary Account Numbers (PANs) per PCI DSS Requirement 3.5. Applications store tokens instead of PANs - card data never appears in application logs, databases, or analytics systems.

## Token Properties

- Same length as original PAN
- Preserves BIN (first 6 digits) for payment routing decisions
- Preserves last 4 digits for cardholder display
- Random middle section unique per tokenization
- Passes Luhn validation (prevents PAN-detection scanners from triggering on tokens)

## Tokenizing a Card

```python
from capabilities.common.vault.service import TokenizationService

svc = TokenizationService(tenant_id="payment_processor")
record = await svc.tokenize_pan("4111111111111111")

print(record.token)       # "411111XXXX1111" - Luhn-valid, same length
print(record.masked_pan)  # "411111XXXXXX1111"
print(record.card_type)   # "visa"
print(record.last_four)   # "1111"

# Store record.token in your DB - never the original PAN
```

## Idempotent Tokenization

`get_or_create_token` returns the same stable token for a PAN on repeated calls. Useful when multiple systems independently tokenize the same card.

```python
record1 = await svc.get_or_create_token("4111111111111111")
record2 = await svc.get_or_create_token("4111111111111111")
assert record1.token == record2.token  # same token returned
```

## Detokenizing (PCI-Authorized Only)

```python
pan = await svc.detokenize_pan(
    token=record.token,
    requester_role="pci_authorized",
    requester_id="payment_gateway_service",
)
```

Unauthorized detokenization raises `PermissionError`. When `OPA_URL` is configured, OPA enforces the authorization decision.

## Bulk Operations

### Streaming Tokenization (large datasets)

```python
async def pan_source():
    for pan in legacy_database.fetch_pans():
        yield pan

async for item in svc.tokenize_pan_stream(pan_source()):
    if isinstance(item, TokenRecord):
        store_token(item.token)
    else:
        handle_error(item["error"], item["index"])
```

Concurrency is bounded by `bulk_concurrency` (default 50). Per-item failures do not abort the stream.

### Bulk Detokenization

```python
result = await svc.detokenize_batch(
    tokens=["4111...1234", "5200...5678"],
    requester_role="pci_authorized",
    requester_id="fraud_service",
)
print(f"Success rate: {result.success_rate:.0%}")
for item in result.succeeded:
    process(item["token"], item["pan"])
for item in result.failed:
    log_error(item["token"], item["error"])
```

## Token Lifecycle Management

### Expiry

```python
# Expire immediately
await svc.expire_token(token)

# Expire after 1 hour
await svc.expire_token(token, ttl_seconds=3600)
```

### Revocation

```python
# Permanent, irreversible
await svc.revoke_token(token, reason="card_reported_stolen")
```

### Status Check (no PCI auth required)

```python
if await svc.is_token_active(token):
    proceed_with_payment()
```

## Zero-Downtime Key Rotation

```python
# Re-encrypt all tokens under new key, preserving token strings
for token in get_all_tokens():
    await svc.rekey_token(token, new_vault_key=os.environ["NEW_APG_VAULT_KEY"])
```

## Token Inspection (no PCI auth required)

```python
meta = await svc.get_token_metadata(token)
print(meta.card_type)           # "visa"
print(meta.status)              # TokenStatus.ACTIVE
print(meta.expires_at)          # None or ISO-8601 string
print(meta.revocation_reason)   # None or string
```

## Display Masking

```python
from capabilities.common.vault.service import MaskingPolicy

# PCI DSS compliant display
masked = await svc.format_masked_pan(token, MaskingPolicy.BIN_LAST4)
# "411111XXXXXX1111"

masked = await svc.format_masked_pan(token, MaskingPolicy.LAST4_ONLY)
# "************1111"
```

## Zero-Knowledge Ownership Attestation

Prove card ownership to a counterparty without sharing the PAN:

```python
# Caller side: compute commitment using a shared challenge
challenge_hex = secrets.token_hex(32)   # 64-char hex
hmac_hex = hmac.new(
    bytes.fromhex(challenge_hex),
    pan.encode(),
    hashlib.sha256,
).hexdigest()
commitment = challenge_hex + hmac_hex

# Vault side: verify without seeing the PAN
result = await svc.attest_token(token, commitment)
print(result.attested)       # True / False
print(result.signature)      # HMAC-signed by vault key
```

## Compliance Status

```python
status = await svc.get_compliance_status()
print(status["pci_dss_compliant"])                 # True/False
print(status["tokens_issued_this_session"])
print(status["pci_dss_requirements_addressed"])
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/vault/tokenize` | Tokenize a PAN |
| POST | `/api/vault/tokenize/batch` | Tokenize multiple PANs |
| POST | `/api/vault/detokenize` | Detokenize (PCI auth required) |
| POST | `/api/vault/detokenize/batch` | Bulk detokenize |
| POST | `/api/vault/validate/luhn` | Validate Luhn check digit |
| POST | `/api/vault/tokens/{token}/expire` | Expire a token |
| POST | `/api/vault/tokens/{token}/revoke` | Revoke a token |
| GET | `/api/vault/tokens/{token}/status` | Check token active status |
| GET | `/api/vault/tokens/{token}/metadata` | Token metadata (no PCI auth) |
| POST | `/api/vault/tokens/{token}/attest` | ZK ownership attestation |
| POST | `/api/vault/keys/rotate` | Rotate encryption key |
| GET | `/api/vault/compliance` | PCI DSS compliance status |
| GET | `/api/vault/health` | Health check |

## PCI DSS Scope Reduction

By using tokens in `fintech_gwy` and `fintech_trx`, those capabilities never persist actual PANs. The PCI DSS cardholder data environment (CDE) is limited to the `vault` capability alone.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `APG_VAULT_KEY` | _(dev key - DO NOT USE IN PROD)_ | AES encryption key for token store |
| `OPA_URL` | _(none)_ | OPA endpoint for authorization |
| `NATS_URL` | _(none)_ | NATS for audit event publishing |

## Production Deployment

Replace the XOR cipher with AES-256 (or inject HashiCorp Vault Transit / AWS KMS). Inject a PostgreSQL session for persistent token storage:

```python
svc = TokenizationService(
    tenant_id="acme",
    db=async_session,
    vault_key=os.environ["APG_VAULT_KEY"],
    bulk_concurrency=100,
)
```

The `apg_token_vault` table schema is in `0001_token_vault.sql`. Add columns `token_status`, `expires_at`, and `revocation_reason` to support lifecycle management in persistent storage.
