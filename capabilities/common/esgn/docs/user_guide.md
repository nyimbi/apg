# Digital Forms and eSign

**Capability ID**: `esgn` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`esgn` provides APG's common capability for governed digital forms and electronic signatures. It composes form-template authoring, schema validation, publication approval, submissions, signature envelopes, ordered signing ceremonies, cancellation/rejection, tamper sealing, encrypted evidence packages, first-class provider-neutral signing agents, UI route metadata, visual theming, and Bytewax lifecycle guardrails.

## Installation

```bash
pip install apg-common-esgn
```

## Provides

- `digital_forms`
- `signature_envelopes`
- `signing_ceremonies`
- `evidence_packages`
- `signing_agent_composition`

## Requires

- `auth`
- `encr`
- `audl`
- `comp`
- `aicr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/esgn/dashboard` | `esgn:view` | Overview |
| `/esgn/forms` | `esgn:create_forms` | Forms |
| `/esgn/builder` | `esgn:create_forms` | Forms |
| `/esgn/submissions` | `esgn:view` | Forms |
| `/esgn/envelopes` | `esgn:send_envelopes` | Signatures |
| `/esgn/signing` | `esgn:sign` | Signatures |
| `/esgn/agents` | `esgn:send_envelopes` | Signatures |
| `/esgn/lifecycle` | `esgn:admin` | Operations |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `put()`
- `get()`
- `list()`
- `delete()`
- `log_event()`
- `send()`
- `form_create()`
- `form_publish()`

_(See `service.py` for complete API.)_

## Interoperability

`esgn` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use esgn;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ESGN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
