# Risk and Compliance Management

**Capability ID**: `grc_rcm` | **Domain**: `grc` | **Version**: `2.2.0`

## Description

`grc_rcm` is the APG capability packet for governed risk, control, compliance, evidence, issue, exception, governance-decision, and AI-agent review lifecycles. It is intentionally dependency-light at the package boundary so APG applications

## Installation

```bash
pip install apg-grc-rcm
```

## Provides

- `risk_register_lifecycle`
- `control_library_lifecycle`
- `compliance_obligation_lifecycle`
- `control_assessment_workflow`
- `evidence_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-rcm/dashboard` | `grc_rcm:view` | Overview |
| `/grc-rcm/heatmap` | `grc_rcm:view` | Overview |
| `/grc-rcm/risks` | `grc_rcm:manage_risks` | Risk |
| `/grc-rcm/risks/:id` | `grc_rcm:view` | Risk |
| `/grc-rcm/controls` | `grc_rcm:manage_controls` | Controls |
| `/grc-rcm/obligations` | `grc_rcm:manage_obligations` | Compliance |
| `/grc-rcm/assessments` | `grc_rcm:assess_controls` | Controls |
| `/grc-rcm/evidence` | `grc_rcm:manage_evidence` | Compliance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_risk()`
- `register_control()`
- `register_obligation()`
- `assess_control()`
- `collect_evidence()`
- `open_issue()`
- `remediate_issue()`
- `record_governance_decision()`

_(See `service.py` for complete API.)_

## Interoperability

`grc_rcm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grc_rcm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_RCM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
