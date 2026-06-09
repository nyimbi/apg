# Social Media Intelligence

**Capability ID**: `intel_socint` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_socint` is an executable APG capability for lawful public or authorized social-source intelligence. It can be composed into generated APG applications that need social monitoring, public-safety alerting, fraud and disinformation

## Installation

```bash
pip install apg-intel-socint
```

## Provides

- `socint_authority_workflow`
- `socint_topic_workflow`
- `socint_source_workflow`
- `socint_post_workflow`
- `socint_signal_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-socint/dashboard` | `intel_socint:view` | Overview |
| `/intel-socint/authorities` | `intel_socint:authorities` | Governance |
| `/intel-socint/topics` | `intel_socint:topics` | Planning |
| `/intel-socint/sources` | `intel_socint:sources` | Collection |
| `/intel-socint/posts` | `intel_socint:posts` | Collection |
| `/intel-socint/signals` | `intel_socint:signals` | Analysis |
| `/intel-socint/influence` | `intel_socint:influence` | Analysis |
| `/intel-socint/networks` | `intel_socint:networks` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_topic()`
- `register_source()`
- `record_post()`
- `record_signal()`
- `record_influence()`
- `record_network()`
- `record_referral()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_socint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_socint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_SOCINT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
