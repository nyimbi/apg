# Vault — PCI DSS Tokenization User Guide

## Overview

The `vault` capability implements format-preserving tokenization (FPT) for payment card Primary Account Numbers (PANs) per PCI DSS Requirement 3.5. Applications store tokens instead of PANs — card data never appears in application logs, databases, or analytics systems. The PCI DSS cardholder data environment scope is dramatically reduced.

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

print(record.token)       # "411111XXXX1111" — Luhn-valid, same length
print(record.masked_pan)  # "411111XXXXXX1111"
print(record.card_type)   # "visa"
print(record.last_four)   # "1111"

# Store record.token in your DB — never the original PAN
```

## Detokenizing (PCI-Authorized Only)

```python
# Only PCI-authorized roles can retrieve the original PAN
pan = await svc.detokenize_pan(
    token=record.token,
    requester_role="pci_authorized",
    requester_id="payment_gateway_service",
)
```

Unauthorized detokenization raises `PermissionError`. When `OPA_URL` is configured, OPA enforces the authorization decision.

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/vault/tokenize` | Tokenize a PAN |
| POST | `/api/vault/tokenize/batch` | Tokenize multiple PANs |
| POST | `/api/vault/detokenize` | Detokenize (PCI auth required) |
| POST | `/api/vault/validate/luhn` | Validate Luhn check digit |
| POST | `/api/vault/secrets` | Store encrypted secret |
| GET | `/api/vault/secrets/{key}` | Retrieve secret |
| POST | `/api/vault/keys/rotate` | Rotate encryption key |
| GET | `/api/vault/compliance` | PCI DSS compliance status |
| GET | `/api/vault/audit` | Tokenization audit log |
| GET | `/api/vault/health` | Health check |

## PCI DSS Scope Reduction

By using tokens in `fintech_gwy` and `fintech_trx`, those capabilities never persist actual PANs. The PCI DSS cardholder data environment (CDE) is limited to the `vault` capability alone — reducing audit scope significantly.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VAULT_ENCRYPTION_KEY` | _(auto-generated)_ | AES encryption key for token store |
| `OPA_URL` | _(none)_ | OPA endpoint for authorization |

## Production Deployment

In production, replace the XOR cipher with AES-256 (or inject HashiCorp Vault Transit / AWS KMS). Inject a PostgreSQL session for persistent token storage:

```python
svc = TokenizationService(tenant_id="acme", db=async_session)
```

The `apg_token_vault` table schema is in `0001_token_vault.sql`.
