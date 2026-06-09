# FinTech Compliance Automation

**Capability ID**: `fintech_compliance` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

FinTech Compliance Automation provides a structured framework for managing regulatory obligations, control mappings, compliance checks, evidence collection, attestations, issues, remediation plans, reports, and governance reviews across all supported regulatory frameworks. It acts as the internal compliance layer that links every operational capability to its governing regulatory requirements.

## Installation

```bash
pip install apg-fintech-compliance
```

## Provides

- `compliance_obligation_workflow`
- `compliance_control_workflow`
- `compliance_check_workflow`
- `compliance_evidence_workflow`
- `compliance_attestation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-compliance/dashboard` | `fintech_compliance:view` | Overview |
| `/fintech-compliance/obligations` | `fintech_compliance:obligations` | Obligations |
| `/fintech-compliance/controls` | `fintech_compliance:controls` | Controls |
| `/fintech-compliance/checks` | `fintech_compliance:checks` | Testing |
| `/fintech-compliance/evidence` | `fintech_compliance:evidence` | Evidence |
| `/fintech-compliance/attestations` | `fintech_compliance:attestations` | Governance |
| `/fintech-compliance/issues` | `fintech_compliance:issues` | Issues |
| `/fintech-compliance/reports` | `fintech_compliance:reports` | Reporting |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_obligation()`
- `map_control()`
- `record_check()`
- `attach_evidence()`
- `record_attestation()`
- `open_issue()`
- `record_remediation()`
- `publish_report()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_compliance` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_compliance;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_COMPLIANCE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
