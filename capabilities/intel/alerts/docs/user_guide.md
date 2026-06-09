# Alert Management

**Capability ID**: `intel_alerts` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_alerts` is an executable APG capability package for building governed alert-management applications. It gives generated APG apps a concrete runtime for lawful authority, alert workspaces, rules, signals, alerts, escalations,

## Installation

```bash
pip install apg-intel-alerts
```

## Provides

- `alert_authority_workflow`
- `alert_workspace_workflow`
- `alert_rule_workflow`
- `alert_signal_workflow`
- `alert_record_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-alerts/dashboard` | `intel_alerts:view` | Overview |
| `/intel-alerts/authorities` | `intel_alerts:authorities` | Governance |
| `/intel-alerts/workspaces` | `intel_alerts:workspaces` | Planning |
| `/intel-alerts/rules` | `intel_alerts:rules` | Configuration |
| `/intel-alerts/signals` | `intel_alerts:signals` | Signals |
| `/intel-alerts/alerts` | `intel_alerts:alerts` | Operations |
| `/intel-alerts/escalations` | `intel_alerts:escalations` | Operations |
| `/intel-alerts/notifications` | `intel_alerts:notifications` | Dissemination |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `record_rule()`
- `record_signal()`
- `record_alert()`
- `record_escalation()`
- `record_notification()`
- `record_assignment()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_alerts` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_alerts;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_ALERTS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
