# Clinical Management

**Capability ID**: `healthcare_cli` | **Domain**: `healthcare` | **Version**: `1.0.0`

## Description

Clinical workflow orchestration capability providing care plan management, clinical protocol activation, workflow task tracking, clinical decision support (CDS) alerts, structured handoff management, and care team coordination. Enforces structured SBAR handoff format and requires team assignment before care plan activation.

## Installation

```bash
pip install apg-healthcare-cli
```

## Provides

- `care_plan_management`
- `clinical_workflow_orchestration`
- `protocol_adherence_tracking`
- `clinical_decision_support`
- `care_team_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-cli/dashboard` | `healthcare_cli:view` | Overview |
| `/healthcare-cli/care-plans` | `healthcare_cli:care_plans` | Care Plans |
| `/healthcare-cli/care-plans/new` | `healthcare_cli:care_plans_write` | Care Plans |
| `/healthcare-cli/care-plans/<id>` | `healthcare_cli:care_plans` | Care Plans |
| `/healthcare-cli/protocols` | `healthcare_cli:protocols` | Protocols |
| `/healthcare-cli/protocols/<id>` | `healthcare_cli:protocols` | Protocols |
| `/healthcare-cli/workflows` | `healthcare_cli:workflows` | Workflows |
| `/healthcare-cli/cds` | `healthcare_cli:cds` | Decision Support |

## Key Service Methods

- `describe()`
- `create_care_plan()`
- `activate_care_plan()`
- `complete_care_plan()`
- `get_care_plan()`
- `list_care_plans()`
- `add_intervention()`
- `create_care_pathway()`
- `enrol_patient_pathway()`
- `pathway_progress()`

_(See `service.py` for complete API.)_

## Interoperability

`healthcare_cli` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use healthcare_cli;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_CLI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
