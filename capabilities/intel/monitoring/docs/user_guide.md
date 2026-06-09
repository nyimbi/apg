# Real-Time Monitoring

**Capability ID**: `intel_monitoring` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_monitoring` is an executable APG capability for lawful, defensive real-time monitoring workflows. It can be composed into generated APG applications that need security monitoring, fraud monitoring, public-safety

## Installation

```bash
pip install apg-intel-monitoring
```

## Provides

- `monitoring_authority_workflow`
- `monitoring_policy_workflow`
- `monitoring_source_workflow`
- `monitoring_watch_workflow`
- `monitoring_event_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-monitoring/dashboard` | `intel_monitoring:view` | Overview |
| `/intel-monitoring/authorities` | `intel_monitoring:authorities` | Governance |
| `/intel-monitoring/policies` | `intel_monitoring:policies` | Planning |
| `/intel-monitoring/sources` | `intel_monitoring:sources` | Sources |
| `/intel-monitoring/watches` | `intel_monitoring:watches` | Detection |
| `/intel-monitoring/events` | `intel_monitoring:events` | Detection |
| `/intel-monitoring/signals` | `intel_monitoring:signals` | Analysis |
| `/intel-monitoring/incidents` | `intel_monitoring:incidents` | Response |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_policy()`
- `register_source()`
- `record_watch()`
- `record_event()`
- `record_signal()`
- `record_incident()`
- `record_referral()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_monitoring` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_monitoring;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_MONITORING_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
