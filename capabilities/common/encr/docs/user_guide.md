# Encryption Services

**Capability ID**: `encr` | **Domain**: `common` | **Version**: `1.0.0`

## Description

Encryption Services (`encr`) is APG's cryptographic governance capability for generated applications. It gives application builders a dependency-light runtime for key-domain posture, crypto operation decisions, legacy algorithm

## Installation

```bash
pip install apg-common-encr
```

## Provides

- `encr_operations`
- `crypto_governance`
- `crypto_agent_composition`
- `review_evidence`

## Requires

- `conf`
- `auth`
- `secu`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/encr/dashboard` | `encr:view` | Overview |
| `/encr/operations` | `encr:operate` | Operations |
| `/encr/keys` | `encr:view_keys` | Operations |
| `/encr/policies` | `encr:manage_policies` | Governance |
| `/encr/entropy` | `encr:view_entropy` | Governance |
| `/encr/exceptions` | `encr:review` | Governance |
| `/encr/rotations` | `encr:rotate` | Operations |
| `/encr/homomorphic` | `encr:compute` | Advanced |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `put()`
- `get()`
- `list()`
- `delete()`
- `log_event()`
- `send()`
- `encrypt_data()`
- `decrypt_data()`

_(See `service.py` for complete API.)_

## Interoperability

`encr` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use encr;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENCR_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
