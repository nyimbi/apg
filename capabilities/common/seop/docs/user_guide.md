# Security Operations

**Capability ID**: `seop` | **Domain**: `common` | **Version**: `1.0.0`

## Description

SEOP is the APG security-operations capability. It gives generated applications a composable runtime for detections, incident response, response playbooks, posture controls, audit evidence, governed AI agents, UI view models, visual theming, and Bytewax lifecycle events.

## Installation

```bash
pip install apg-common-seop
```

## Provides

- `detection_pipeline`
- `incident_response`
- `threat_triage`
- `response_playbooks`
- `security_posture`

## Requires

- `secu`
- `anom`
- `moni`
- `logt`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/seop/dashboard` | `seop:view` | Overview |
| `/seop/detections` | `seop:triage` | Detection |
| `/seop/incidents` | `seop:respond` | Incidents |
| `/seop/triage` | `seop:triage` | Detection |
| `/seop/playbooks` | `seop:manage_playbooks` | Response |
| `/seop/responses` | `seop:respond` | Response |
| `/seop/posture` | `seop:view` | Operations |
| `/seop/agents` | `seop:admin` | Automation |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_detection()`
- `open_incident()`
- `approve_playbook()`
- `execute_response()`
- `record_posture_control()`
- `close_incident()`
- `create_soc_alert()`
- `triage_alert()`

_(See `service.py` for complete API.)_

## Interoperability

`seop` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use seop;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `SEOP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
