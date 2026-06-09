# Mine Safety & Compliance

**Capability ID**: `mining_saf` | **Domain**: `mining` | **Version**: `1.0.0`

## Description

Manages mine safety operations including incident reporting and investigation, hazard identification and risk assessment, risk register maintenance, permit-to-work issuance, corrective action tracking, compliance obligation registers, safety audits, and safety statistics reporting. Enforces statutory requirements including mandatory investigation before closing LTI and above incidents, stop-work authority for extreme risks, and issuer qualification checks for high-risk permits.

## Installation

```bash
pip install apg-mining-saf
```

## Provides

- `incident_reporting_workflow`
- `hazard_identification_workflow`
- `risk_register_management`
- `permit_to_work_workflow`
- `corrective_action_tracking`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mining-saf/dashboard` | `mining_saf:view` | Overview |
| `/mining-saf/incidents` | `mining_saf:view` | Incidents |
| `/mining-saf/incidents/create` | `mining_saf:write` | Incidents |
| `/mining-saf/incidents/:id` | `mining_saf:view` | Incidents |
| `/mining-saf/hazards` | `mining_saf:view` | Hazards |
| `/mining-saf/hazards/create` | `mining_saf:write` | Hazards |
| `/mining-saf/risk-register` | `mining_saf:view` | Risk |
| `/mining-saf/permits` | `mining_saf:view` | Permits |

## Key Service Methods

- `report_incident()`
- `get_incident()`
- `send_regulatory_notification()`
- `open_investigation()`
- `close_incident()`
- `list_incidents()`
- `identify_hazard()`
- `get_hazard()`
- `close_hazard()`
- `list_hazards()`

_(See `service.py` for complete API.)_

## Interoperability

`mining_saf` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mining_saf;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MINING_SAF_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
