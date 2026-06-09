# Facial Recognition

**Capability ID**: `frec` | **Domain**: `common` | **Version**: `1.0.0`

## Description

FREC provides governed facial recognition for APG applications. It covers face consent, face-template enrollment, liveness evidence, one-to-one verification, one-to-many identification, watchlist policy, emotion-analysis governance, review queues, first-class facial-recognition governance agents, Bytewax lifecycle batch validation, audit evidence, UI metadata, and visual theming.

## Installation

```bash
pip install apg-common-frec
```

## Provides

- `facial_recognition`
- `face_identification`
- `facial_recognition_agent_composition`

## Requires

- `biop`
- `cvsn`
- `aicr`
- `encr`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/frec/dashboard` | `frec:view` | Overview |
| `/frec/subjects` | `frec:view` | Identity |
| `/frec/consents` | `frec:enroll` | Identity |
| `/frec/enrollment` | `frec:enroll` | Identity |
| `/frec/templates` | `frec:enroll` | Identity |
| `/frec/verification` | `frec:verify` | Identity |
| `/frec/identification` | `frec:identify` | Identity |
| `/frec/liveness` | `frec:verify` | Security |

## Key Service Methods

- `_create_audit_log()`
- `_simple_liveness_check()`
- `_extract_probe_features()`
- `initialize()`
- `close()`
- `create_user()`
- `get_user()`
- `get_user_by_external_id()`
- `update_verification_threshold()`
- `get_service_statistics()`

_(See `service.py` for complete API.)_

## Interoperability

`frec` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use frec;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FREC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
