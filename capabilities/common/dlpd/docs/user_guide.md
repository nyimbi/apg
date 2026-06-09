# Data Loss Prevention

**Capability ID**: `dlpd` | **Domain**: `common` | **Version**: `1.0.0`

## Description

DLPD is APG's generated-application capability for tenant-scoped data loss prevention. It gives composed applications a deterministic, dependency-light surface for data classification, policy enforcement, egress inspection,

## Installation

```bash
pip install apg-common-dlpd
```

## Provides

_(see capability contract)_

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/dlpd/dashboard` | `dlpd:view` | Overview |
| `/dlpd/policies` | `dlpd:manage_policies` | Policies |
| `/dlpd/classifiers` | `dlpd:manage_policies` | Policies |
| `/dlpd/channels` | `dlpd:inspect` | Monitoring |
| `/dlpd/inspections` | `dlpd:inspect` | Monitoring |
| `/dlpd/incidents` | `dlpd:respond` | Response |
| `/dlpd/quarantine` | `dlpd:respond` | Response |
| `/dlpd/reviews` | `dlpd:review` | Response |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_policy()`
- `create_policy()`
- `update_policy()`
- `policy_effectiveness()`
- `register_classifier()`
- `regex_pattern_library()`
- `ml_classifier_train()`
- `classify_content()`

_(See `service.py` for complete API.)_

## Interoperability

`dlpd` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use dlpd;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `DLPD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
